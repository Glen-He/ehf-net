"""
异构图构建器。

负责组装节点特征、建立多类型边关系，
并产出模型直接使用的 HeteroData 图对象。
"""


from typing import Any, cast

import numpy as np
import torch

from torch import Tensor
from torch_geometric.data import HeteroData

from ehfnet.data.featurizers import (
    LIGAND_ATOM_CAT_SCHEMA,
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    LigandEncodingResult,
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
    ProteinEncodingResult,
)
from ehfnet.graph.collate import GraphCollator
from ehfnet.graph.features import build_context_features
from ehfnet.graph.schema import (
    AGGREGATE_EDGES,
    BROADCAST_EDGES,
    INTRA_EDGES,
    STATIC_INTER_EDGES,
)
from ehfnet.graph.topology import (
    build_aggregate_edges,
    build_broadcast_edges,
    build_intra_edges,
    build_static_inter_edges,
)


class ESMEmbeddingFiller:
    """
    ESM 嵌入填充器。

    负责在残基级 ESM 嵌入缺失时提供一致的替代表示，
    避免蛋白编码流程因局部缺失值而破坏特征维度约定。
    """

    def __init__(self, *, embed_dim: int, fill_strategy: str) -> None:
        """
        初始化 ESM 嵌入填充器。

        配置嵌入维度和缺失值填充策略，
        用于残基级 ESM 特征缺失时的统一补全。

        Args:
            embed_dim: ESM 嵌入维度。
            fill_strategy: ESM 缺失嵌入的填充策略。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
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
        处理残基 ESM 嵌入列表。

        将原始嵌入结果整理为固定维度张量，
        并按填充策略补齐缺失项。

        Args:
            embeddings: 待保存或处理的嵌入结果。
            device: 运行所用设备，如 CPU 或 CUDA 设备。

        Returns:
            tuple[Tensor, Tensor]: 补齐缺失值后的 ESM 嵌入张量与缺失掩码。

        Raises:
            ValueError: 当输入嵌入维度与配置不一致时抛出。
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
                torch.from_numpy(emb_array).to(
                    device=output_device,
                    dtype=torch.float32,
                )
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
    异构图构建器。

    负责节点特征组装、局部上下文特征生成、拓扑建立和约束注入，
    是预处理和运行时局部裁剪后重建图结构的核心组件。
    """

    def __init__(
        self,
        *,
        r_cutoff_intra: float,
        max_neighbors_intra: int,
        atom_neighbor_cap: int,
        residue_neighbor_cap: int,
        residue_radius_scale: float,
        residue_radius_bias: float,
        ligand_atom_fallback_k: int,
        protein_atom_fallback_k: int,
        protein_residue_fallback_k: int,
        esm_filler: ESMEmbeddingFiller | None = None,
        interaction_profile: str,
    ) -> None:
        """
        初始化图构建器。

        保存图内边、上下文节点和交互拓扑相关配置，
        为异构图构建和运行时局部裁剪重建提供统一规则。

        Args:
            r_cutoff_intra: 图内边构建的距离截断半径。
            max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
            atom_neighbor_cap: 原子层图内边的邻居上限。
            residue_neighbor_cap: 残基层图内边的邻居上限。
            residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
            residue_radius_bias: 残基层邻域半径的额外偏置。
            ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
            protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
            protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
            esm_filler: 负责填充缺失 ESM 嵌入的处理器。
            interaction_profile: 跨图交互拓扑配置。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """
        required_args = {
            "r_cutoff_intra": r_cutoff_intra,
            "max_neighbors_intra": max_neighbors_intra,
            "atom_neighbor_cap": atom_neighbor_cap,
            "residue_neighbor_cap": residue_neighbor_cap,
            "residue_radius_scale": residue_radius_scale,
            "residue_radius_bias": residue_radius_bias,
            "ligand_atom_fallback_k": ligand_atom_fallback_k,
            "protein_atom_fallback_k": protein_atom_fallback_k,
            "protein_residue_fallback_k": protein_residue_fallback_k,
            "interaction_profile": interaction_profile,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "GraphBuilder is missing required explicit configuration values: "
                f"{missing_args}."
            )
        if esm_filler is None:
            raise ValueError("GraphBuilder is missing required explicit configuration value: esm_filler.")
        self.r_cutoff_intra = r_cutoff_intra
        self.max_neighbors_intra = max_neighbors_intra
        self.esm_filler = esm_filler
        self.interaction_profile = interaction_profile
        self._residue_esm_feature_start = len(PROTEIN_RESIDUE_CONT_SCHEMA)
        atom_neighbor_cap = max(1, min(int(max_neighbors_intra), int(atom_neighbor_cap)))
        residue_neighbor_cap = max(
            1, min(int(max_neighbors_intra), int(residue_neighbor_cap))
        )
        atom_radius = float(r_cutoff_intra)
        residue_radius = max(
            atom_radius * float(residue_radius_scale),
            atom_radius + float(residue_radius_bias),
        )
        self._intra_edge_cfg: dict[str, dict[str, float | int]] = {
            "ligand_atom": {
                "radius": atom_radius,
                "max_neighbors": atom_neighbor_cap,
                "fallback_k": max(1, int(ligand_atom_fallback_k)),
            },
            "protein_atom": {
                "radius": atom_radius,
                "max_neighbors": atom_neighbor_cap,
                "fallback_k": max(1, int(protein_atom_fallback_k)),
            },
            "protein_residue": {
                "radius": residue_radius,
                "max_neighbors": residue_neighbor_cap,
                "fallback_k": max(1, int(protein_residue_fallback_k)),
            },
        }

        if self.interaction_profile not in {"full", "atom_only"}:
            raise ValueError(
                f"Unsupported interaction_profile='{self.interaction_profile}'. "
                "Use one of {'full', 'atom_only'}."
            )

    @property
    def residue_esm_feature_start(self) -> int:
        """
        residue 连续特征中 ESM 子向量的起始列。

        Returns:
            int: 返回残基连续特征中 ESM 子向量的起始列号。
        """

        return self._residue_esm_feature_start

    def _is_inter_edge_enabled(self, src: str, dst: str) -> bool:
        if self.interaction_profile == "full":
            return True
        if self.interaction_profile == "atom_only":
            atom_pair = {"ligand_atom", "protein_atom"}
            return {src, dst} == atom_pair
        return True

    def build(self, ligand_data: LigandEncodingResult, protein_data: ProteinEncodingResult) -> HeteroData:
        """
        构建单个异构图样本。

        接收蛋白和配体编码结果，组装节点特征、局部上下文和拓扑边，
        输出模型可直接使用的 `HeteroData`。

        Args:
            ligand_data: 配体侧编码结果或中间表示。
            protein_data: 蛋白侧编码结果或中间表示。

        Returns:
            HeteroData: 完整组装后的异构图对象。
        """
        data = HeteroData()
        data = self._add_ligand_atoms(data, ligand_data)
        data = self._add_ligand_molecule(data, ligand_data)
        data = self._add_protein_atoms(data, protein_data)
        data = self._add_protein_residues(data, protein_data)
        data = self._add_protein_context(data)
        data = self.build_graph_topology(data)
        data = self._add_torsion_constraints(data, ligand_data)
        return data

    def _add_ligand_atoms(self, data: HeteroData, ligand_data: LigandEncodingResult) -> HeteroData:
        atom_feat = cast(dict[str, Any], ligand_data["atom_features"])
        x_cat = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.long) for f in LIGAND_ATOM_CAT_SCHEMA],
            dim=1,
        )
        x_cont = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.float32) for f in LIGAND_ATOM_CONT_SCHEMA],
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
        mol_feat = cast(dict[str, Any], ligand_data["mol_features"])
        x_cont = torch.tensor(
            [[mol_feat[f.name] for f in LIGAND_MOLECULE_CONT_SCHEMA]],
            dtype=torch.float32,
        )
        data["ligand_molecule"].x_cont = x_cont
        data["ligand_molecule"].num_nodes = 1
        return data

    def _add_protein_atoms(self, data: HeteroData, protein_data: ProteinEncodingResult) -> HeteroData:
        atom_feat = cast(dict[str, Any], protein_data["atom_features"])
        x_cat = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.long) for f in PROTEIN_ATOM_CAT_SCHEMA],
            dim=1,
        )
        x_cont = torch.stack(
            [torch.tensor(atom_feat[f.name], dtype=torch.float32) for f in PROTEIN_ATOM_CONT_SCHEMA],
            dim=1,
        )
        pos = torch.tensor(protein_data["atom_positions"], dtype=torch.float32)
        residue_idx = torch.tensor(
            protein_data["atom_to_residue_index"], dtype=torch.long
        )
        data["protein_atom"].x_cat = x_cat
        data["protein_atom"].x_cont = x_cont
        data["protein_atom"].pos = pos
        data["protein_atom"].residue_idx = residue_idx
        data["protein_atom"].num_nodes = int(pos.size(0))
        return data

    def _add_protein_residues(self, data: HeteroData, protein_data: ProteinEncodingResult) -> HeteroData:
        res_feat = cast(dict[str, Any], protein_data["residue_features"])
        x_cat = torch.stack(
            [torch.tensor(res_feat[f.name], dtype=torch.long) for f in PROTEIN_RESIDUE_CAT_SCHEMA],
            dim=1,
        )
        torsion_cont = torch.stack(
            [torch.tensor(res_feat[f.name], dtype=torch.float32) for f in PROTEIN_RESIDUE_CONT_SCHEMA],
            dim=1,
        )
        esm_emb, esm_missing_mask = self.esm_filler.process(
            cast(list[np.ndarray | None], protein_data["residue_esm_embeddings"]),
            device=torsion_cont.device,
        )
        x_cont = torch.cat([torsion_cont, esm_emb], dim=1)
        pos = torch.tensor(protein_data["residue_positions"], dtype=torch.float32)
        data["protein_residue"].x_cat = x_cat
        data["protein_residue"].x_cont = x_cont
        data["protein_residue"].pos = pos
        data["protein_residue"].esm_missing_mask = esm_missing_mask
        data["protein_residue"].num_nodes = int(pos.size(0))

        metadata = cast(dict[str, Any], protein_data.get("residue_metadata", {}))
        for key in [
            "source_residue_ix",
            "source_resid",
            "source_chain_index",
            "source_icode_code",
            "source_segment_id",
            "source_segment_offset",
            "source_segment_length",
        ]:
            if key in metadata:
                data["protein_residue"][key] = torch.tensor(metadata[key], dtype=torch.long)

        auxiliary = cast(dict[str, Any], protein_data["auxiliary"])
        for key in [
            "type_atom14_mask",
            "observed_atom14_mask",
            "atom14_ambiguity_group",
            "type_torsion_mask",
            "observed_torsion_mask",
            "observed_backbone_mask",
            "chi_pi_periodic_mask",
        ]:
            data["protein_residue"][key] = torch.tensor(auxiliary[key], dtype=torch.float32)

        return data

    def build_context_node_features(
        self,
        data: HeteroData,
        *,
        center: Tensor | None = None,
    ) -> Tensor:
        """
        根据当前 residue/protein_atom 几何生成 protein_context 连续特征。

        Args:
            data: 当前处理的图数据对象。
            center: 局部裁剪或特征汇总所围绕的中心坐标。

        Returns:
            Tensor: `protein_context` 节点的连续特征张量。
        """

        return build_context_features(
            residue_x_cont=data["protein_residue"].x_cont,
            residue_pos=data["protein_residue"].pos,
            protein_atom_pos=data["protein_atom"].pos,
            residue_esm_missing_mask=getattr(
                data["protein_residue"], "esm_missing_mask", None
            ),
            esm_feature_start=self._residue_esm_feature_start,
            center=center,
        )

    def _add_protein_context(self, data: HeteroData) -> HeteroData:
        data["protein_context"].x_cont = self.build_context_node_features(data)
        data["protein_context"].num_nodes = 1
        return data

    def build_graph_topology(self, data: HeteroData) -> HeteroData:
        """
        按 schema 构建图的全部静态边拓扑。

        Args:
            data: 当前处理的图数据对象。

        Returns:
            HeteroData: 建好拓扑边并补齐约束字段后的图对象。
        """

        data = build_intra_edges(
            data,
            intra_edges=INTRA_EDGES,
            intra_edge_cfg=self._intra_edge_cfg,
        )
        data = build_aggregate_edges(data, aggregate_edges=AGGREGATE_EDGES)
        data = build_static_inter_edges(
            data,
            static_inter_edges=STATIC_INTER_EDGES,
            is_edge_enabled=self._is_inter_edge_enabled,
        )
        data = build_broadcast_edges(data, broadcast_edges=BROADCAST_EDGES)
        return data

    def _add_torsion_constraints(self, data: HeteroData, ligand_data: LigandEncodingResult) -> HeteroData:
        device = (
            data["ligand_atom"].pos.device
            if hasattr(data["ligand_atom"], "pos")
            else None
        )
        torsion_indices = cast(list[list[int]], ligand_data.get("torsion_indices", []))
        torsion_masks = cast(list[list[bool]], ligand_data.get("torsion_masks", []))

        if torsion_indices:
            data.torsion_indices = torch.tensor(
                torsion_indices, dtype=torch.long, device=device
            )
        else:
            data.torsion_indices = torch.zeros((0, 4), dtype=torch.long, device=device)

        if torsion_masks:
            data.torsion_moving_mask = torch.tensor(
                torsion_masks, dtype=torch.bool, device=device
            )
        else:
            n_lig = int(data["ligand_atom"].num_nodes)
            data.torsion_moving_mask = torch.zeros(
                (0, n_lig), dtype=torch.bool, device=device
            )

        return data


def create_graph_tools(
    *,
    esm_dim: int,
    r_cutoff_intra: float,
    max_neighbors_intra: int,
    atom_neighbor_cap: int,
    residue_neighbor_cap: int,
    residue_radius_scale: float,
    residue_radius_bias: float,
    ligand_atom_fallback_k: int,
    protein_atom_fallback_k: int,
    protein_residue_fallback_k: int,
    esm_fill_strategy: str,
    interaction_profile: str,
) -> tuple[GraphBuilder, GraphCollator]:
    """
    创建构图工具集。

    按统一参数实例化 `GraphBuilder` 与 `GraphCollator`，
    供预处理和运行时裁剪流程共享同一套构图约定。

    Args:
        esm_dim: ESM 残基嵌入维度。
        r_cutoff_intra: 图内边构建的距离截断半径。
        max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
        atom_neighbor_cap: 原子层图内边的邻居上限。
        residue_neighbor_cap: 残基层图内边的邻居上限。
        residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
        residue_radius_bias: 残基层邻域半径的额外偏置。
        ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
        protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
        protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
        esm_fill_strategy: 残基 ESM 缺失时的填充策略。
        interaction_profile: 跨图交互拓扑配置。

    Returns:
        tuple[GraphBuilder, GraphCollator]: 与当前配置匹配的一组构图工具。
    """

    esm_filler = ESMEmbeddingFiller(
        embed_dim=int(esm_dim),
        fill_strategy=esm_fill_strategy,
    )
    builder = GraphBuilder(
        r_cutoff_intra=r_cutoff_intra,
        max_neighbors_intra=max_neighbors_intra,
        atom_neighbor_cap=atom_neighbor_cap,
        residue_neighbor_cap=residue_neighbor_cap,
        residue_radius_scale=residue_radius_scale,
        residue_radius_bias=residue_radius_bias,
        ligand_atom_fallback_k=ligand_atom_fallback_k,
        protein_atom_fallback_k=protein_atom_fallback_k,
        protein_residue_fallback_k=protein_residue_fallback_k,
        esm_filler=esm_filler,
        interaction_profile=interaction_profile,
    )
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
    return builder, collator
