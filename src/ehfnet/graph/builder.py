"""
异构图构建器

将 ligand/protein 编码结果组装为 PyG HeteroData，并构建图拓扑与扭转约束。
"""

import torch
import numpy as np

from typing import Any, cast
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius, radius_graph

from ehfnet.encoders.feature_specs import (
    LIGAND_ATOM_CAT_SCHEMA,
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.graph.hetero_schema import (
    INTRA_EDGES,
    AGGREGATE_EDGES,
    INTER_EDGES,
    BROADCAST_EDGES,
)


class ESMEmbeddingFiller:
    """
    ESM embedding 填充器

    用于 residue 级别的 ESM embeddings 存在缺失时，提供一致的填充值。
    """

    def __init__(self, *, embed_dim: int = 1152, fill_strategy: str = "zeros") -> None:
        """
        Args:
            embed_dim: embedding 维度（ESM-3 默认 1152）
            fill_strategy: 缺失填充策略，支持 "zeros" 或 "learnable"
        """
        self.embed_dim = embed_dim
        self.fill_strategy = fill_strategy

        if fill_strategy != "zeros":
            raise ValueError(
                f"Unsupported fill_strategy='{fill_strategy}'. Use 'zeros' here and handle learnable missing embeddings in the model."
            )

    def process(
        self,
        embeddings: list[np.ndarray | None],
        *,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        将可能包含 None 的 embedding 列表转换为 Tensor。

        Args:
            embeddings: 长度为 N 的列表，每个元素为 (D,) numpy 向量或 None
            device: 输出 Tensor 的 device；默认使用 CPU

        Returns:
            (embeddings, missing_mask)，其中：
            - embeddings: 形状 [N, D] 的 float32 Tensor（缺失位置填 0）
            - missing_mask: 形状 [N] 的 bool Tensor（True 表示缺失）
        """

        output_device = device or torch.device("cpu")
        result: list[Tensor] = []
        missing: list[bool] = []

        for i, emb in enumerate(embeddings):
            if emb is None:
                missing.append(True)
                result.append(
                    torch.zeros(
                        (self.embed_dim,),
                        dtype=torch.float32,
                        device=output_device,
                    )
                )
                continue

            emb_array = np.asarray(emb)

            if emb_array.shape != (self.embed_dim,):
                raise ValueError(
                    f"ESM embedding must have shape ({self.embed_dim},), got {emb_array.shape} at index {i}."
                )
            missing.append(False)
            result.append(
                torch.from_numpy(emb_array).to(device=output_device, dtype=torch.float32)
            )

        if not result:
            empty_emb = torch.zeros(
                (0, self.embed_dim),
                dtype=torch.float32,
                device=output_device,
            )
            empty_mask = torch.zeros((0,), dtype=torch.bool, device=output_device)
            return empty_emb, empty_mask

        embeddings_tensor = torch.stack(result, dim=0)
        missing_mask = torch.tensor(missing, dtype=torch.bool, device=output_device)
        return embeddings_tensor, missing_mask


class GraphBuilder:
    """
    图构建器

    负责节点特征组装、边拓扑构建（intra/aggregate/inter/broadcast）以及扭转约束注入。
    """

    def __init__(
        self,
        *,
        r_cutoff_intra: float = 5.0,
        r_cutoff_inter: float = 6.0,
        max_neighbors_intra: int = 64,
        max_neighbors_inter: int = 32,
        esm_filler: ESMEmbeddingFiller | None = None,
    ) -> None:
        """
        Args:
            r_cutoff_intra: 图内边的距离阈值（原子/残基内部）
            r_cutoff_inter: 跨图边的距离阈值（配体-蛋白交互）
            max_neighbors_intra: 图内最大邻居数（PyG radius_graph）
            max_neighbors_inter: 跨图最大邻居数（PyG radius）
            esm_filler: ESM embedding 填充器（默认使用 ESMEmbeddingFiller）
        """

        self.r_cutoff_intra = r_cutoff_intra
        self.r_cutoff_inter = r_cutoff_inter
        self.max_neighbors_intra = max_neighbors_intra
        self.max_neighbors_inter = max_neighbors_inter
        self.esm_filler = esm_filler or ESMEmbeddingFiller()


    def build(self, ligand_data: dict[str, Any], protein_data: dict[str, Any]) -> HeteroData:
        """
        构建单个样本的异构图。

        Args:
            ligand_data: 配体编码结果字典
            protein_data: 蛋白编码结果字典

        Returns:
            PyG HeteroData 对象
        """

        data = HeteroData()

        # 1. 节点特征构建
        data = self._add_ligand_atoms(data, ligand_data)
        data = self._add_ligand_molecule(data, ligand_data)
        data = self._add_protein_atoms(data, protein_data)
        data = self._add_protein_residues(data, protein_data)
        data = self._add_protein_pocket(data, protein_data)

        # 2. 边拓扑构建
        data = self._build_graph_topology(data)

        # 3. 物理约束构建
        data = self._add_torsion_constraints(data, ligand_data)

        return data


    def _add_ligand_atoms(self, data: HeteroData, ligand_data: dict[str, Any]) -> HeteroData:
        """
        构建配体原子节点特征与坐标。

        Args:
            data: 待填充的 HeteroData
            ligand_data: 配体编码结果字典

        Returns:
            更新后的 HeteroData
        """

        atom_feat = cast(dict[str, Any], ligand_data["atom_features"])

        x_cat = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.long) for f in LIGAND_ATOM_CAT_SCHEMA],
            dim=1,
        )
        x_cont = torch.stack(
            [
                torch.tensor(atom_feat[f.name], dtype=torch.float32)
                for f in LIGAND_ATOM_CONT_SCHEMA
            ],
            dim=1,
        )

        pos = torch.tensor(ligand_data["positions"], dtype=torch.float32)
        masses = torch.tensor(atom_feat["atomic_weight"], dtype=torch.float32)

        data["ligand_atom"].x_cat = x_cat
        data["ligand_atom"].x_cont = x_cont
        data["ligand_atom"].pos = pos
        data["ligand_atom"].masses = masses
        data["ligand_atom"].num_nodes = int(pos.size(0))

        return data


    def _add_ligand_molecule(self, data: HeteroData, ligand_data: dict[str, Any]) -> HeteroData:
        """
        构建配体分子全局节点特征。

        Args:
            data: 待填充的 HeteroData
            ligand_data: 配体编码结果字典

        Returns:
            更新后的 HeteroData
        """

        mol_feat = cast(dict[str, Any], ligand_data["mol_features"])
        x_cont = torch.tensor(
            [[mol_feat[f.name] for f in LIGAND_MOLECULE_CONT_SCHEMA]],
            dtype=torch.float32,
        )

        data["ligand_molecule"].x_cont = x_cont
        data["ligand_molecule"].num_nodes = 1

        return data


    def _add_protein_atoms(self, data: HeteroData, protein_data: dict[str, Any]) -> HeteroData:
        """
        构建蛋白原子节点特征与坐标，并附加 atom->residue 映射索引。

        Args:
            data: 待填充的 HeteroData
            protein_data: 蛋白编码结果字典

        Returns:
            更新后的 HeteroData
        """

        atom_feat = cast(dict[str, Any], protein_data["atom_features"])

        x_cat = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.long) for f in PROTEIN_ATOM_CAT_SCHEMA],
            dim=1,
        )
        x_cont = torch.stack(
            [
                torch.tensor(atom_feat[f.name], dtype=torch.float32)
                for f in PROTEIN_ATOM_CONT_SCHEMA
            ],
            dim=1,
        )

        pos = torch.tensor(protein_data["atom_positions"], dtype=torch.float32)
        residue_idx = torch.tensor(protein_data["atom_to_residue_index"], dtype=torch.long)

        data["protein_atom"].x_cat = x_cat
        data["protein_atom"].x_cont = x_cont
        data["protein_atom"].pos = pos
        data["protein_atom"].residue_idx = residue_idx
        data["protein_atom"].num_nodes = int(pos.size(0))

        return data


    def _add_protein_residues(self, data: HeteroData, protein_data: dict[str, Any]) -> HeteroData:
        """
        构建蛋白残基节点特征与坐标，并附加辅助 mask（结构恢复/损失计算）。

        Args:
            data: 待填充的 HeteroData
            protein_data: 蛋白编码结果字典

        Returns:
            更新后的 HeteroData
        """

        res_feat = cast(dict[str, Any], protein_data["residue_features"])

        x_cat = torch.stack(
            [
                torch.tensor(res_feat[f.name], dtype=torch.long)
                for f in PROTEIN_RESIDUE_CAT_SCHEMA
            ],
            dim=1,
        )

        torsion_cont = torch.stack(
            [
                torch.tensor(res_feat[f.name], dtype=torch.float32)
                for f in PROTEIN_RESIDUE_CONT_SCHEMA
            ],
            dim=1,
        )

        esm_emb, esm_missing_mask = self.esm_filler.process(
            cast(list[np.ndarray | None], protein_data["residue_esm_embeddings"]),
            device=torsion_cont.device,
        )

        # 拼接几何特征和 ESM 特征
        x_cont = torch.cat([torsion_cont, esm_emb], dim=1)
        pos = torch.tensor(protein_data["residue_positions"], dtype=torch.float32)

        data["protein_residue"].x_cat = x_cat
        data["protein_residue"].x_cont = x_cont
        data["protein_residue"].pos = pos
        data["protein_residue"].esm_missing_mask = esm_missing_mask
        data["protein_residue"].num_nodes = int(pos.size(0))

        # 辅助 mask，用于后续 loss 计算或结构恢复
        auxiliary = cast(dict[str, Any], protein_data["auxiliary"])

        for key in ["atom14_mask", "atom14_symmetry_mask", "torsion_angle_mask", "chi_pi_periodic_mask"]:
            data["protein_residue"][key] = torch.tensor(auxiliary[key], dtype=torch.float32)

        return data


    def _add_protein_pocket(self, data: HeteroData, _protein_data: dict[str, Any]) -> HeteroData:
        """
        构建蛋白 pocket 全局节点特征。

        当前实现使用 residue 节点特征的均值作为 pocket 表征；当 residue 为空时返回同维度零向量。

        Args:
            data: 待填充的 HeteroData
            _protein_data: 蛋白编码结果字典（目前不直接使用，但保留以便后续扩展）

        Returns:
            更新后的 HeteroData
        """

        n_residues = int(data["protein_residue"].num_nodes)

        if n_residues > 0:
            pocket_cont = data["protein_residue"].x_cont.mean(dim=0, keepdim=True)

        else:
            feat_dim = data["protein_residue"].x_cont.size(1)
            pocket_cont = torch.zeros(
                (1, feat_dim),
                device=data["protein_residue"].x_cont.device,
                dtype=data["protein_residue"].x_cont.dtype,
            )

        data["protein_pocket"].x_cont = pocket_cont
        data["protein_pocket"].num_nodes = 1

        return data


    def _build_graph_topology(self, data: HeteroData) -> HeteroData:
        """
        按 schema 构建图的所有边类型拓扑。

        Args:
            data: 已包含节点特征与坐标的 HeteroData

        Returns:
            填充了 edge_index 的 HeteroData
        """

        data = self._build_intra_edges(data)
        data = self._build_aggregate_edges(data)
        data = self._build_inter_edges(data)
        data = self._build_broadcast_edges(data)

        return data


    def _build_intra_edges(self, data: HeteroData) -> HeteroData:
        """
        构建图内边（同类型节点内部的半径邻接）。

        Args:
            data: HeteroData

        Returns:
            更新后的 HeteroData
        """

        for src, rel, dst in INTRA_EDGES:

            if not hasattr(data[src], "pos") or data[src].pos.numel() == 0:
                continue

            pos = data[src].pos
            # 残基尺度通常比原子尺度大，给予更大的 cutoff
            r_cutoff = self.r_cutoff_intra * 2 if "residue" in src else self.r_cutoff_intra
            edge_index = self._radius_graph(
                pos,
                r_cutoff,
                max_num_neighbors=self.max_neighbors_intra,
            )
            data[src, rel, dst].edge_index = edge_index

        return data


    def _build_aggregate_edges(self, data: HeteroData) -> HeteroData:
        """
        构建聚合边（局部节点 -> 全局节点 或 原子->残基）。

        Args:
            data: HeteroData

        Returns:
            更新后的 HeteroData
        """

        for src, rel, dst in AGGREGATE_EDGES:

            if not hasattr(data[src], "pos") or data[src].pos.numel() == 0:
                continue

            device = data[src].pos.device
            n_src = int(data[src].pos.size(0))

            if src == "protein_atom" and dst == "protein_residue":
                # 原子 -> 残基：利用预计算好的 mapping
                atom_to_res = data["protein_atom"].residue_idx
                # edge_index: [source, target] -> [atom_idx, residue_idx]
                edge_index = torch.stack(
                    [torch.arange(len(atom_to_res), device=atom_to_res.device), atom_to_res],
                    dim=0,
                )

            else:
                # 聚合到 Molecule/Pocket (Global Node)
                # 所有 source 节点都连接到 index 为 0 的 target 节点
                edge_index = torch.stack(
                    [
                        torch.arange(n_src, device=device),
                        torch.zeros(n_src, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )

            data[src, rel, dst].edge_index = edge_index

        return data


    def _build_inter_edges(self, data: HeteroData) -> HeteroData:
        """
        构建配体与蛋白质之间的跨图交互边。

        规则：
        - atom/residue 之间使用 radius 构图；
        - protein_pocket 相关关系使用全连接（全局节点通常不具备可用的几何坐标）。

        为避免重复计算，若 schema 同时包含 (src, rel, dst) 与 (dst, rel, src)，则仅计算一次，
        并在 schema 允许的前提下用 flip(0) 补齐反向边。

        Args:
            data: HeteroData

        Returns:
            更新后的 HeteroData
        """

        inter_edge_types = set(INTER_EDGES)
        processed_edges: set[tuple[str, str, str]] = set()

        for src, rel, dst in INTER_EDGES:
            edge_key = (rel, *sorted([src, dst]))

            if edge_key in processed_edges:
                continue

            is_pocket_edge = (src == "protein_pocket") or (dst == "protein_pocket")

            if not is_pocket_edge:

                if not hasattr(data[src], "pos") or not hasattr(data[dst], "pos"):
                    continue

                if data[src].pos.numel() == 0 or data[dst].pos.numel() == 0:
                    continue

            if hasattr(data["ligand_atom"], "pos"):
                edge_device = data["ligand_atom"].pos.device
            else:
                edge_device = torch.device("cpu")

            edge_index = torch.zeros((2, 0), dtype=torch.long, device=edge_device)

            if is_pocket_edge:
                n_src_nodes = int(data[src].num_nodes)
                n_dst_nodes = int(data[dst].num_nodes)

                # 构建全连接边: [0...N] -> [0...M]
                src_idx = torch.arange(n_src_nodes, device=edge_device).repeat_interleave(
                    n_dst_nodes
                )
                dst_idx = torch.arange(n_dst_nodes, device=edge_device).repeat(n_src_nodes)
                edge_index = torch.stack([src_idx, dst_idx], dim=0)

            else:
                edge_index = self._bipartite_radius_graph(
                    data[src].pos,
                    data[dst].pos,
                    self.r_cutoff_inter,
                    max_num_neighbors=self.max_neighbors_inter,
                )

            data[src, rel, dst].edge_index = edge_index
            reverse_edge_type = (dst, rel, src)

            if reverse_edge_type in inter_edge_types:

                if edge_index.numel() > 0:
                    data[dst, rel, src].edge_index = edge_index.flip(0)

                else:
                    data[dst, rel, src].edge_index = edge_index

            processed_edges.add(edge_key)

        return data


    def _build_broadcast_edges(self, data: HeteroData) -> HeteroData:
        """
        构建广播边（全局节点 -> 局部节点）。

        Args:
            data: HeteroData

        Returns:
            更新后的 HeteroData
        """

        for src, rel, dst in BROADCAST_EDGES:

            if not hasattr(data[dst], "pos") or data[dst].pos.numel() == 0:
                continue

            device = data[dst].pos.device
            n_dst = int(data[dst].pos.size(0))
            # 广播：全局节点（global node，索引 0）-> 全部局部节点
            edge_index = torch.stack(
                [
                    torch.zeros(n_dst, dtype=torch.long, device=device),
                    torch.arange(n_dst, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            data[src, rel, dst].edge_index = edge_index

        return data


    def _add_torsion_constraints(self, data: HeteroData, ligand_data: dict[str, Any]) -> HeteroData:
        """
        向 HeteroData 注入配体扭转约束信息。

        Args:
            data: HeteroData
            ligand_data: 配体编码结果字典（可包含 torsion_indices/torsion_masks）

        Returns:
            更新后的 HeteroData
        """

        device = data["ligand_atom"].pos.device if hasattr(data["ligand_atom"], "pos") else None

        torsion_indices = cast(list[list[int]], ligand_data.get("torsion_indices", []))
        torsion_masks = cast(list[list[bool]], ligand_data.get("torsion_masks", []))

        if torsion_indices:
            data.torsion_indices = torch.tensor(torsion_indices, dtype=torch.long, device=device)

        else:
            data.torsion_indices = torch.zeros((0, 4), dtype=torch.long, device=device)

        if torsion_masks:
            data.torsion_moving_mask = torch.tensor(torsion_masks, dtype=torch.bool, device=device)

        else:
            n_lig = int(data["ligand_atom"].num_nodes)
            data.torsion_moving_mask = torch.zeros((0, n_lig), dtype=torch.bool, device=device)

        return data


    @staticmethod
    def _radius_graph(pos: Tensor, r_cutoff: float, *, max_num_neighbors: int = 64) -> Tensor:
        """
        基于 radius_graph 构建同集合内的邻接边。

        Args:
            pos: 节点坐标，形状 [N, 3]
            r_cutoff: 半径阈值
            max_num_neighbors: 每个节点的最大邻居数

        Returns:
            edge_index，形状 [2, E]
        """

        return radius_graph(pos, r=r_cutoff, loop=False, max_num_neighbors=max_num_neighbors)


    @staticmethod
    def _bipartite_radius_graph(
        pos_src: Tensor,
        pos_dst: Tensor,
        r_cutoff: float,
        *,
        max_num_neighbors: int = 32,
    ) -> Tensor:
        """
        构建二部图半径邻接边（src -> dst）。

        Args:
            pos_src: 源节点坐标，形状 [N_src, 3]
            pos_dst: 目标节点坐标，形状 [N_dst, 3]
            r_cutoff: 半径阈值
            max_num_neighbors: 每个目标节点的最大邻居数（PyG radius 约束）

        Returns:
            edge_index（source_index, target_index），形状 [2, E]
        """

        # PyG radius 返回 (target_index, source_index)
        edge_index = radius(
            pos_src,
            pos_dst,
            r=r_cutoff,
            batch_x=None,
            batch_y=None,
            max_num_neighbors=max_num_neighbors,
        )
        # 翻转为 (source_index, target_index) 以符合直觉
        if edge_index.numel() > 0:
            edge_index = edge_index.flip(0)

        return edge_index


def create_graph_tools(
    *,
    r_cutoff_intra: float = 5.0,
    r_cutoff_inter: float = 6.0,
    max_neighbors_intra: int = 64,
    max_neighbors_inter: int = 32,
    esm_fill_strategy: str = "zeros",
) -> tuple[GraphBuilder, "GraphCollator"]:
    """
    创建 GraphBuilder 与 GraphCollator。

    Args:
        r_cutoff_intra: 图内边构建半径阈值
        r_cutoff_inter: 跨图边构建半径阈值
        max_neighbors_intra: 图内最大邻居数
        max_neighbors_inter: 跨图最大邻居数
        esm_fill_strategy: ESM embedding 缺失时的填充策略

    Returns:
        (builder, collator)
    """
    
    # 局部导入避免循环依赖
    from ehfnet.graph.collate import GraphCollator

    esm_filler = ESMEmbeddingFiller(embed_dim=1152, fill_strategy=esm_fill_strategy)
    builder = GraphBuilder(
        r_cutoff_intra=r_cutoff_intra,
        r_cutoff_inter=r_cutoff_inter,
        max_neighbors_intra=max_neighbors_intra,
        max_neighbors_inter=max_neighbors_inter,
        esm_filler=esm_filler,
    )
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
    return builder, collator
