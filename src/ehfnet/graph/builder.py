"""
异构图构建器

将 ligand/protein 编码结果组装为 PyG HeteroData，并构建图拓扑与扭转约束。
"""

import torch
import numpy as np

from typing import Any, cast
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph
from ehfnet.encoders.ligand_encoder import LigandEncodingResult
from ehfnet.encoders.protein_encoder import ProteinEncodingResult
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
    STATIC_INTER_EDGES,
    BROADCAST_EDGES,
)
from ehfnet.graph.collate import GraphCollator
from ehfnet.graph.pocket_features import build_pocket_features


class ESMEmbeddingFiller:
    """
    ESM embedding 填充器

    用于 residue 级别的 ESM embeddings 存在缺失时，提供一致的填充值。
    """

    def __init__(self, *, embed_dim: int = 960, fill_strategy: str = "zeros") -> None:
        """
        Args:
            embed_dim: embedding 维度（ESMC-300M 默认 960）
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
        interaction_profile: str = "full",
    ) -> None:
        """
        Args:
            r_cutoff_intra: 图内边的距离阈值（原子/残基内部）
            r_cutoff_inter: 保留兼容参数；动态跨图边现由 encoder 侧控制
            max_neighbors_intra: 图内最大邻居数（PyG radius_graph）
            max_neighbors_inter: 保留兼容参数；动态跨图边现由 encoder 侧控制
            esm_filler: ESM embedding 填充器（默认使用 ESMEmbeddingFiller）
            interaction_profile: 跨图交互配置，支持：
                - "full": 保留全部跨图边（默认）
                - "atom_only": 仅保留 ligand_atom<->protein_atom（用于多尺度交互消融）
        """

        self.r_cutoff_intra = r_cutoff_intra
        self.r_cutoff_inter = r_cutoff_inter
        self.max_neighbors_intra = max_neighbors_intra
        self.max_neighbors_inter = max_neighbors_inter
        self.esm_filler = esm_filler or ESMEmbeddingFiller()
        self.interaction_profile = interaction_profile

        if self.interaction_profile not in {"full", "atom_only"}:
            raise ValueError(
                f"Unsupported interaction_profile='{self.interaction_profile}'. "
                "Use one of {'full', 'atom_only'}."
            )


    def _is_inter_edge_enabled(self, src: str, dst: str) -> bool:
        """
        判断当前跨图边是否在 interaction_profile 下启用。
        """

        if self.interaction_profile == "full":
            return True

        if self.interaction_profile == "atom_only":
            atom_pair = {"ligand_atom", "protein_atom"}
            return {src, dst} == atom_pair

        return True


    def build(self, ligand_data: LigandEncodingResult, protein_data: ProteinEncodingResult) -> HeteroData:
        """
        构建单个样本的异构图。

        Args:
            ligand_data: 配体编码结果
            protein_data: 蛋白编码结果

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


    def _add_ligand_atoms(self, data: HeteroData, ligand_data: LigandEncodingResult) -> HeteroData:
        """
        构建配体原子节点特征与坐标。

        Args:
            data: 待填充的 HeteroData
            ligand_data: 配体编码结果

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


    def _add_ligand_molecule(self, data: HeteroData, ligand_data: LigandEncodingResult) -> HeteroData:
        """
        构建配体分子全局节点特征。

        Args:
            data: 待填充的 HeteroData
            ligand_data: 配体编码结果

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


    def _add_protein_atoms(self, data: HeteroData, protein_data: ProteinEncodingResult) -> HeteroData:
        """
        构建蛋白原子节点特征与坐标，并附加 atom->residue 映射索引。

        Args:
            data: 待填充的 HeteroData
            protein_data: 蛋白编码结果

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


    def _add_protein_residues(self, data: HeteroData, protein_data: ProteinEncodingResult) -> HeteroData:
        """
        构建蛋白残基节点特征与坐标，并附加辅助 mask（结构恢复/损失计算）。

        Args:
            data: 待填充的 HeteroData
            protein_data: 蛋白编码结果

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


    def _add_protein_pocket(self, data: HeteroData, _protein_data: ProteinEncodingResult) -> HeteroData:
        """
        构建蛋白 pocket 全局节点特征。

        当前实现使用 residue 节点特征的均值作为 pocket 表征；当 residue 为空时返回同维度零向量。

        Args:
            data: 待填充的 HeteroData
            _protein_data: 蛋白编码结果（目前不直接使用，但保留以便后续扩展）

        Returns:
            更新后的 HeteroData
        """

        pocket_cont = build_pocket_features(
            residue_x_cont=data["protein_residue"].x_cont,
            residue_pos=data["protein_residue"].pos,
            protein_atom_pos=data["protein_atom"].pos,
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

            pos = data[src].pos
            # [修复] KNN 图构建
            # protein_residue 节点少但空间稀疏 -> 128 邻居汇聚全局上下文
            # protein_atom / ligand_atom 节点多且密集 -> 32 邻居已足够捕获局部信息
            # 旧逻辑 '"protein" in src' 误将 protein_atom 也匹配到 k=128，
            # 导致大口袋产生 O(N_pro_atom * 128) 条边，是显存 OOM 的元凶之一
            k = 128 if "residue" in src else 32
            actual_k = min(k, pos.size(0) - 1)
            
            if actual_k > 0:
                edge_index = knn_graph(
                    pos,
                    k=actual_k,
                    loop=False,
                )
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)

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
        构建静态跨图交互边。

        规则：
        - 仅构建 pocket 相关的全局静态边；
        - atom-atom / atom-residue 动态交互边由 encoder 在每个 block 重新生成。

        Args:
            data: HeteroData

        Returns:
            更新后的 HeteroData
        """

        inter_edge_types = set(STATIC_INTER_EDGES)
        processed_edges: set[tuple[str, str, str]] = set()

        for src, rel, dst in STATIC_INTER_EDGES:
            if not self._is_inter_edge_enabled(src, dst):
                continue

            sorted_nodes = sorted([src, dst])
            edge_key = (rel, sorted_nodes[0], sorted_nodes[1])

            if edge_key in processed_edges:
                continue

            if hasattr(data["ligand_atom"], "pos"):
                edge_device = data["ligand_atom"].pos.device
            else:
                edge_device = torch.device("cpu")

            n_src_nodes = int(data[src].num_nodes)
            n_dst_nodes = int(data[dst].num_nodes)

            # pocket 相关边使用全连接，全局节点不依赖显式局部半径拓扑。
            src_idx = torch.arange(n_src_nodes, device=edge_device).repeat_interleave(
                n_dst_nodes
            )
            dst_idx = torch.arange(n_dst_nodes, device=edge_device).repeat(n_src_nodes)
            edge_index = torch.stack([src_idx, dst_idx], dim=0)

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


    def _add_torsion_constraints(self, data: HeteroData, ligand_data: LigandEncodingResult) -> HeteroData:
        """
        向 HeteroData 注入配体扭转约束信息。

        Args:
            data: HeteroData
            ligand_data: 配体编码结果（可包含 torsion_indices/torsion_masks）

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


def create_graph_tools(
    *,
    r_cutoff_intra: float = 5.0,
    r_cutoff_inter: float = 6.0,
    max_neighbors_intra: int = 64,
    max_neighbors_inter: int = 32,
    esm_fill_strategy: str = "zeros",
    interaction_profile: str = "full",
) -> tuple[GraphBuilder, GraphCollator]:
    """
    创建 GraphBuilder 与 GraphCollator。

    Args:
        r_cutoff_intra: 图内边构建半径阈值
        r_cutoff_inter: 保留兼容参数；动态跨图边现由 encoder 侧控制
        max_neighbors_intra: 图内最大邻居数
        max_neighbors_inter: 保留兼容参数；动态跨图边现由 encoder 侧控制
        esm_fill_strategy: ESM embedding 缺失时的填充策略
        interaction_profile: 跨图交互配置（"full" 或 "atom_only"）

    Returns:
        (builder, collator)
    """

    esm_filler = ESMEmbeddingFiller(embed_dim=960, fill_strategy=esm_fill_strategy)
    builder = GraphBuilder(
        r_cutoff_intra=r_cutoff_intra,
        r_cutoff_inter=r_cutoff_inter,
        max_neighbors_intra=max_neighbors_intra,
        max_neighbors_inter=max_neighbors_inter,
        esm_filler=esm_filler,
        interaction_profile=interaction_profile,
    )
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
    return builder, collator
