"""
EHFNet 编码器

分层 EGNN 网络，按四个阶段处理图与坐标信息。
"""

import logging
import torch
import torch.nn as nn

from typing import Any, cast
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean
from egnn_pytorch import EGNN_Sparse
from ehfnet.models.layers.frame_conv import FrameAwareConv, FrameAwareHeteroConv
from egnn_pytorch.egnn_pytorch import CoorsNorm

# [修复] Monkey Patch egnn_pytorch.CoorsNorm
# PyTorch 的 tensor.norm(dim=-1) 在输入为严格 0 时会产生 NaN 梯度。
# 在分子对接随机初始化或完全重合场景下，这是 grad_norm=nan 的直接原因。
def safe_coors_norm_forward(self, coors):
    # 在 sum 之后立即加 eps 再开根号，彻底杜绝开根号 0 的梯度问题
    norm = torch.sqrt((coors ** 2).sum(dim=-1, keepdim=True) + self.eps)
    normed_coors = coors / norm
    return normed_coors * self.scale
CoorsNorm.forward = safe_coors_norm_forward

from ehfnet.graph.hetero_schema import (
    NODE_TYPES,
    ATOM_NODE_TYPES,
    INTRA_EDGES,
    AGGREGATE_EDGES,
    DYNAMIC_INTER_EDGES,
    INTER_EDGES,
    BROADCAST_EDGES,
)
from ehfnet.graph.inter_edges import build_batched_radius_or_knn_edges
from ehfnet.graph.pocket_features import build_pocket_features, pocket_feature_dim
from ehfnet.encoders.feature_specs import PROTEIN_RESIDUE_CONT_SCHEMA
from ehfnet.models.layers.embeddings import (
    TimeEmbedding,
    LigandAtomEmbedding,
    LigandMoleculeEmbedding,
    ProteinAtomEmbedding,
    ProteinResidueEmbedding,
    ProteinPocketEmbedding,
)

logger = logging.getLogger(__name__)


class EHFEncoder(nn.Module):
    """
    EHFNet 编码器模块

    分层 EGNN 网络，按四个阶段处理图与坐标信息：
    1) 层内精化：对同一层级的节点做局部坐标与特征细化
    2) 自下而上：将低层信息聚合到高层节点以汇总上下文
    3) 层间交互：跨图层传播与融合特征与坐标信息
    4) 自上而下：将高层语义广播回低层以补偿并细化节点表示

    每个 GNN 块含残差更新与全局特征精化，用于稳定训练与保留原始信息。
    """

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        num_gnn_blocks: int,
        lig_atom_cont_count: int,
        lig_mol_cont_count: int,
        pro_atom_cont_count: int,
        pro_res_cont_count: int,
        *,
        m_dim_scalar: int = 16,
        num_rbf: int = 32,
        dropout_rate: float = 0.0,
        fix_protein: bool = True,
        stats: dict | None = None,
        interaction_profile: str = "full",
        dynamic_inter_cutoff: float = 10.0,
        dynamic_inter_knn_k: int = 8,
        dynamic_residue_cutoff: float = 14.0,
        dynamic_residue_knn_k: int = 6,
    ) -> None:
        """
        Args:
            hidden_dim: 隐藏层维度
            time_dim: 时间嵌入维度（必须是偶数）
            num_gnn_blocks: GNN 块数量
            lig_atom_cont_count: 配体原子连续特征数量
            lig_mol_cont_count: 配体分子连续特征数量
            pro_atom_cont_count: 蛋白原子连续特征数量
            pro_res_cont_count: 蛋白残基连续特征数量
            m_dim_scalar: EGNN 消息维度
            num_rbf: 帧感知卷的高斯 RBF 基函数数量
            dropout_rate: Dropout 比例
            fix_protein: 是否冻结蛋白坐标
            stats: 统计数据字典 (用于输入归一化)
            interaction_profile: 跨图交互配置，支持 "full" 或 "atom_only"
            dynamic_inter_cutoff: 动态跨图原子边半径
            dynamic_inter_knn_k: 半径为空时的 kNN 回退邻居数
            dynamic_residue_cutoff: 动态 ligand-residue 跨图边半径
            dynamic_residue_knn_k: ligand-residue 边为空时的 kNN 回退邻居数
        """
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.num_rbf     = num_rbf
        self.fix_protein = fix_protein
        self.interaction_profile = interaction_profile
        self.dynamic_inter_cutoff = float(dynamic_inter_cutoff)
        self.dynamic_inter_knn_k = max(1, int(dynamic_inter_knn_k))
        self.dynamic_residue_cutoff = float(dynamic_residue_cutoff)
        self.dynamic_residue_knn_k = max(1, int(dynamic_residue_knn_k))

        if self.interaction_profile not in {"full", "atom_only"}:
            raise ValueError(
                f"Unsupported interaction_profile='{self.interaction_profile}'. "
                "Use one of {'full', 'atom_only'}."
            )

        # 1. 特征嵌入
        self.ligand_atom_embedder = LigandAtomEmbedding(
            cont_feature_count=lig_atom_cont_count, 
            hidden_dim=hidden_dim,
            stats=stats.get("ligand_atom") if stats else None
        )
        self.ligand_molecule_embedder = LigandMoleculeEmbedding(
            cont_feature_count=lig_mol_cont_count, 
            hidden_dim=hidden_dim,
            stats=stats.get("ligand_molecule") if stats else None
        )
        self.protein_atom_embedder = ProteinAtomEmbedding(
            cont_feature_count=pro_atom_cont_count, 
            hidden_dim=hidden_dim,
            stats=stats.get("protein_atom") if stats else None
        )
        self.protein_residue_embedder = ProteinResidueEmbedding(
            cont_feature_count=pro_res_cont_count, 
            hidden_dim=hidden_dim,
            stats=stats.get("protein_residue") if stats else None
        )
        self.protein_pocket_embedder = ProteinPocketEmbedding(
            cont_feature_count=pocket_feature_dim(pro_res_cont_count),
            hidden_dim=hidden_dim,
        )
        self.pocket_refresh_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_embedder = TimeEmbedding(dim=time_dim, hidden_dim=hidden_dim)

        # 2. 边类型分类
        self.intra_atom_edges = [
            e for e in INTRA_EDGES if "atom" in e[0] and "atom" in e[2]
        ]
        self.intra_feat_edges = [e for e in INTRA_EDGES if e not in self.intra_atom_edges]
        self.agg_edges = AGGREGATE_EDGES
        self.inter_atom_edges = [
            e for e in DYNAMIC_INTER_EDGES if "atom" in e[0] and "atom" in e[2]
        ]
        if self.interaction_profile == "atom_only":
            self.inter_feat_edges = []
        else:
            self.inter_feat_edges = [e for e in INTER_EDGES if e not in self.inter_atom_edges]
        self.bcast_edges = BROADCAST_EDGES

        # 3. 构造 GNN 模块序列
        self.gnn_blocks: ModuleList = ModuleList()

        for _ in range(num_gnn_blocks):
            block = nn.ModuleDict()

            # 阶段 1：层内精化
            if self.intra_atom_edges:
                block["1_intra_egnn"] = self._build_egnn_block(
                    hidden_dim, m_dim_scalar, dropout_rate
                )

            if self.intra_feat_edges:
                block["1_intra_gnn"] = self._build_frame_conv_block(
                    self.intra_feat_edges, hidden_dim, num_rbf
                )
                block["1_intra_update"] = self._build_update_mlp(hidden_dim, dropout_rate)

            # 阶段 2：自下而上聚合
            if self.agg_edges:
                block["2_agg_gnn"] = self._build_frame_conv_block(
                    self.agg_edges, hidden_dim, num_rbf
                )
                block["2_agg_update"] = self._build_update_mlp(hidden_dim, dropout_rate)

            # 阶段 3：层间交互
            if self.inter_atom_edges:
                block["3_inter_egnn"] = self._build_egnn_block(
                    hidden_dim, m_dim_scalar, dropout_rate
                )

            if self.inter_feat_edges:
                block["3_inter_gnn"] = self._build_frame_conv_block(
                    self.inter_feat_edges, hidden_dim, num_rbf
                )
                block["3_inter_update"] = self._build_update_mlp(hidden_dim, dropout_rate)

            # 阶段 4：自上而下广播
            if self.bcast_edges:
                block["4_bcast_gnn"] = self._build_frame_conv_block(
                    self.bcast_edges, hidden_dim, num_rbf
                )
                block["4_bcast_update"] = self._build_update_mlp(hidden_dim, dropout_rate)

            # 模块末端的全局特征细化
            block["post_mlp"] = self._build_update_mlp(hidden_dim, dropout_rate)

            self.gnn_blocks.append(block)


    @staticmethod
    def _build_frame_conv_block(
        edges: list[tuple[str, str, str]], hidden_dim: int, num_rbf: int
    ) -> FrameAwareHeteroConv:
        """
        构建帧感知异构图消息传递模块（取代 SAGEConv / HeteroConv）。

        每种边类型分配独立的 FrameAwareConv，共享 hidden_dim 和 num_rbf。

        Args:
            edges:      边类型列表 [(src, rel, dst), ...]
            hidden_dim: 所有节点的特征维度 H
            num_rbf:    高斯 RBF 基函数数量

        Returns:
            FrameAwareHeteroConv 实例
        """
        convs: dict[tuple[str, str, str], FrameAwareConv] = {
            edge_key: FrameAwareConv(hidden_dim, num_rbf)
            for edge_key in edges
        }
        return FrameAwareHeteroConv(convs)


    @staticmethod
    def _build_egnn_block(
        hidden_dim: int, m_dim_scalar: int, dropout_rate: float
    ) -> EGNN_Sparse:
        """
        构建 EGNN 模块

        Args:
            hidden_dim: 节点特征维度
            m_dim_scalar: 消息维度
            dropout_rate: Dropout 比例

        Returns:
            EGNN 模块
        """

        return EGNN_Sparse(
            feats_dim=hidden_dim,
            pos_dim=3,
            m_dim=m_dim_scalar,
            aggr="mean",
            update_coors=True,    # 全部 block 启用坐标更新：每层 EGNN 输出的坐标增量
                                  # 通过多步迭代积累等变位移信号；
                                  # encoder 通过 displacement_dict = pos_final - pos_init 返回给 EHFNet。
            update_feats=True,
            dropout=dropout_rate,
            norm_feats=True,
            norm_coors=True,
        )


    @staticmethod
    def _build_update_mlp(hidden_dim: int, dropout_rate: float) -> nn.ModuleDict:
        """
        构建残差更新 MLP

        Args:
            hidden_dim: 隐藏层维度
            dropout_rate: Dropout 比例

        Returns:
            包含各节点类型 MLP 的字典
        """

        return nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout_rate),
                )
                for nt in NODE_TYPES
            }
        )


    def _embed_inputs(
        self, data: HeteroData
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """
        对输入进行特征嵌入

        Args:
            data: 异构图数据

        Returns:
            x_dict: 节点特征字典
            pos_dict: 节点坐标字典
            initial_lig_pos: 初始配体坐标
        """

        x_dict: dict[str, Tensor] = {}
        pos_dict: dict[str, Tensor] = {}

        # 配体原子
        x_all_lig = self.ligand_atom_embedder(
            data["ligand_atom"].x_cat,
            data["ligand_atom"].x_cont,
            data["ligand_atom"].pos,
        )
        pos_dict["ligand_atom"] = x_all_lig[:, :3]
        x_dict["ligand_atom"] = x_all_lig[:, 3:]
        initial_lig_pos = pos_dict["ligand_atom"].clone()

        # 蛋白原子
        x_all_pro = self.protein_atom_embedder(
            data["protein_atom"].x_cat,
            data["protein_atom"].x_cont,
            data["protein_atom"].pos,
        )
        pos_dict["protein_atom"] = x_all_pro[:, :3]
        x_dict["protein_atom"] = x_all_pro[:, 3:]

        # 其他节点
        x_dict["ligand_molecule"] = self.ligand_molecule_embedder(
            data["ligand_molecule"].x_cont
        )
        esm_missing_mask = getattr(data["protein_residue"], "esm_missing_mask", None)
        x_dict["protein_residue"] = self.protein_residue_embedder(
            data["protein_residue"].x_cat,
            data["protein_residue"].x_cont,
            esm_missing_mask=esm_missing_mask,
        )
        x_dict["protein_pocket"] = self.protein_pocket_embedder(
            data["protein_pocket"].x_cont
        )

        return x_dict, pos_dict, initial_lig_pos


    def _run_egnn_on_atoms(
        self,
        egnn_layer: EGNN_Sparse,
        x_dict: dict[str, Tensor],
        pos_dict: dict[str, Tensor],
        edge_dict: dict[tuple[str, str, str], Tensor],
        relevant_edge_keys: list[tuple[str, str, str]],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        在原子层节点上运行 EGNN

        Args:
            egnn_layer: EGNN 模块
            x_dict: 节点特征字典
            pos_dict: 节点坐标字典
            edge_dict: 边索引字典
            relevant_edge_keys: 相关边类型列表

        Returns:
            更新后的特征和坐标字典
        """

        atom_nodes: list[str] = ATOM_NODE_TYPES
        offsets: dict[str, int] = {}
        feats_list: list[Tensor] = []
        pos_list: list[Tensor] = []
        current_offset: int = 0

        for nt in atom_nodes:

            if nt in x_dict and x_dict[nt].shape[0] > 0:
                offsets[nt] = current_offset
                feats_list.append(x_dict[nt])
                pos_list.append(pos_dict[nt])
                current_offset += x_dict[nt].shape[0]

        if not feats_list:
            return x_dict, pos_dict

        total_atoms = sum(f.shape[0] for f in feats_list)
        feat_dim = feats_list[0].shape[1]
        device = feats_list[0].device
        dtype = feats_list[0].dtype

        homo_x_in = torch.zeros(total_atoms, 3 + feat_dim, device=device, dtype=dtype)
        start_idx = 0

        for pos, feat in zip(pos_list, feats_list, strict=True):
            n = pos.shape[0]
            homo_x_in[start_idx : start_idx + n, :3] = pos
            homo_x_in[start_idx : start_idx + n, 3:] = feat
            start_idx += n

        homo_edge_indices: list[Tensor] = []

        for src, rel, dst in relevant_edge_keys:

            if (src, rel, dst) not in edge_dict:
                logger.warning(
                    f"Edge type ({src}, {rel}, {dst}) is missing in input data; skipping."
                )
                continue

            if src not in offsets or dst not in offsets:
                logger.warning(
                    f"Nodes for edge type ({src}, {rel}, {dst}) are missing in input; skipping."
                )
                continue

            edge_index = edge_dict[(src, rel, dst)]
            edge_index_src = edge_index[0] + offsets[src]
            edge_index_dst = edge_index[1] + offsets[dst]
            homo_edge_indices.append(torch.stack([edge_index_src, edge_index_dst]))

        if not homo_edge_indices:
            return x_dict, pos_dict

        homo_edge_index = torch.cat(homo_edge_indices, dim=1)
        homo_x_out = egnn_layer(homo_x_in, homo_edge_index)

        out_pos, out_feats = homo_x_out[:, :3], homo_x_out[:, 3:]

        # 原地更新以优化内存
        for nt, offset in offsets.items():
            length = x_dict[nt].shape[0]
            pos_dict[nt] = out_pos[offset : offset + length]
            x_dict[nt] = out_feats[offset : offset + length]

        return x_dict, pos_dict


    @staticmethod
    def _apply_residual_update(
        x_dict: dict[str, Tensor],
        out_dict: dict[str, Tensor],
        update_mlp: nn.ModuleDict,
    ) -> dict[str, Tensor]:
        """
        应用残差更新

        Args:
            x_dict: 节点特征字典
            out_dict: 消息字典
            update_mlp: 更新 MLP 字典

        Returns:
            更新后的特征字典
        """

        for nt, msg in out_dict.items():

            if nt in x_dict and nt in update_mlp:
                update = update_mlp[nt](msg)
                x_dict[nt] = x_dict[nt] + update

        return x_dict


    def _run_block(
        self,
        block: nn.ModuleDict,
        x_dict: dict[str, Tensor],
        pos_dict: dict[str, Tensor],
        edge_dict: dict[tuple[str, str, str], Tensor],
        full_pos_dict: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        运行单个 GNN 块。

        Args:
            block:         当前 GNN 块的 ModuleDict
            x_dict:        节点特征字典
            pos_dict:      原子层节点坐标字典（EGNN 可更新）
            edge_dict:     边索密字典
            full_pos_dict: 包含所有 5 种节点类型坐标的字典
                           （为 FrameAwareHeteroConv 提供几何参考）
        """
        typed_block = cast(nn.ModuleDict, block)

        # 阶段 1: 层内精化
        if "1_intra_egnn" in typed_block:
            x_dict, pos_dict = self._run_egnn_on_atoms(
                cast(EGNN_Sparse, typed_block["1_intra_egnn"]),
                x_dict,
                pos_dict,
                edge_dict,
                self.intra_atom_edges,
            )

        if "1_intra_gnn" in typed_block:
            out = typed_block["1_intra_gnn"](x_dict, full_pos_dict, edge_dict)
            x_dict = self._apply_residual_update(
                x_dict, out, cast(nn.ModuleDict, typed_block["1_intra_update"])
            )

        # 阶段 2: 自下而上聚合
        if "2_agg_gnn" in typed_block:
            out = typed_block["2_agg_gnn"](x_dict, full_pos_dict, edge_dict)
            x_dict = self._apply_residual_update(
                x_dict, out, cast(nn.ModuleDict, typed_block["2_agg_update"])
            )

        # 阶段 3: 层间交互
        if "3_inter_egnn" in typed_block:
            x_dict, pos_dict = self._run_egnn_on_atoms(
                cast(EGNN_Sparse, typed_block["3_inter_egnn"]),
                x_dict,
                pos_dict,
                edge_dict,
                self.inter_atom_edges,
            )

        if "3_inter_gnn" in typed_block:
            out = typed_block["3_inter_gnn"](x_dict, full_pos_dict, edge_dict)
            x_dict = self._apply_residual_update(
                x_dict, out, cast(nn.ModuleDict, typed_block["3_inter_update"])
            )

        # 阶段 4: 自上而下广播
        if "4_bcast_gnn" in typed_block:
            out = typed_block["4_bcast_gnn"](x_dict, full_pos_dict, edge_dict)
            x_dict = self._apply_residual_update(
                x_dict, out, cast(nn.ModuleDict, typed_block["4_bcast_update"])
            )

        # 全局特征精化
        if "post_mlp" in typed_block:
            post_mlp = cast(nn.ModuleDict, typed_block["post_mlp"])

            for nt in x_dict:
                
                if nt in post_mlp:
                    update = post_mlp[nt](x_dict[nt])
                    x_dict[nt] = x_dict[nt] + update
        
        return x_dict, pos_dict


    @staticmethod
    def _get_node_batch(data: HeteroData, node_type: str, num_nodes: int, device: torch.device) -> Tensor:
        """
        获取节点 batch 索引；若缺失则默认为单图 batch=0。
        """

        if hasattr(data[node_type], "batch") and data[node_type].batch is not None:
            node_batch = data[node_type].batch

            if node_batch.numel() == num_nodes:
                return node_batch

        return torch.zeros(num_nodes, dtype=torch.long, device=device)


    def _build_dynamic_inter_atom_edges(
        self,
        *,
        data: HeteroData,
        pos_dict: dict[str, Tensor],
        edge_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[tuple[str, str, str], Tensor]:
        """
        动态重建 ligand_atom<->protein_atom 跨图边。
        半径图为空时回退 kNN，确保跨图信息不断链。
        """

        key_fw = ("ligand_atom", "inter_proximity", "protein_atom")
        key_bw = ("protein_atom", "inter_proximity", "ligand_atom")

        if "ligand_atom" not in pos_dict or "protein_atom" not in pos_dict:
            return edge_dict

        lig_pos = pos_dict["ligand_atom"]
        pro_pos = pos_dict["protein_atom"]

        if lig_pos.numel() == 0 or pro_pos.numel() == 0:
            return edge_dict

        device = lig_pos.device
        lig_batch = self._get_node_batch(data, "ligand_atom", lig_pos.size(0), device)
        pro_batch = self._get_node_batch(data, "protein_atom", pro_pos.size(0), device)

        # radius 返回 [dst(y), src(x)]，此处 y=ligand, x=protein
        # 输出正向边使用 [lig_idx, pro_idx]
        edge_fw = build_batched_radius_or_knn_edges(
            src_pos=lig_pos,
            src_batch=lig_batch,
            dst_pos=pro_pos,
            dst_batch=pro_batch,
            radius_cutoff=self.dynamic_inter_cutoff,
            knn_k=self.dynamic_inter_knn_k,
            ensure_src_coverage=True,
            max_num_neighbors=max(64, self.dynamic_inter_knn_k * 4),
        )

        edge_dict[key_fw] = edge_fw if edge_fw.numel() > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_dict[key_bw] = edge_dict[key_fw].flip(0) if edge_dict[key_fw].numel() > 0 else edge_dict[key_fw]

        return edge_dict


    def _build_dynamic_ligand_residue_edges(
        self,
        *,
        data: HeteroData,
        pos_dict: dict[str, Tensor],
        edge_dict: dict[tuple[str, str, str], Tensor],
        residue_pos: Tensor,
    ) -> dict[tuple[str, str, str], Tensor]:
        """
        动态重建 ligand_atom<->protein_residue 跨图边（Stage-3 多尺度交互）。
        """

        key_fw = ("ligand_atom", "inter_proximity", "protein_residue")
        key_bw = ("protein_residue", "inter_proximity", "ligand_atom")

        if "ligand_atom" not in pos_dict or "protein_residue" not in data.node_types:
            return edge_dict

        lig_pos = pos_dict["ligand_atom"]
        res_pos = residue_pos

        if lig_pos.numel() == 0 or res_pos.numel() == 0:
            return edge_dict

        device = lig_pos.device
        lig_batch = self._get_node_batch(data, "ligand_atom", lig_pos.size(0), device)
        res_batch = self._get_node_batch(data, "protein_residue", res_pos.size(0), device)

        edge_fw = build_batched_radius_or_knn_edges(
            src_pos=lig_pos,
            src_batch=lig_batch,
            dst_pos=res_pos,
            dst_batch=res_batch,
            radius_cutoff=self.dynamic_residue_cutoff,
            knn_k=self.dynamic_residue_knn_k,
            ensure_src_coverage=True,
            max_num_neighbors=max(64, self.dynamic_residue_knn_k * 6),
        )

        edge_dict[key_fw] = edge_fw if edge_fw.numel() > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_dict[key_bw] = edge_dict[key_fw].flip(0) if edge_dict[key_fw].numel() > 0 else edge_dict[key_fw]

        return edge_dict


    def _compute_current_residue_positions(
        self,
        *,
        data: HeteroData,
        pos_dict: dict[str, Tensor],
    ) -> Tensor:
        """
        从当前 protein_atom 坐标派生 residue 几何中心。

        rigid 模式直接复用输入 residue 坐标；
        flexible 模式下保持 atom / residue / pocket 几何一致。
        """

        if self.fix_protein:
            return data["protein_residue"].pos

        if "protein_atom" not in pos_dict or pos_dict["protein_atom"].numel() == 0:
            return data["protein_residue"].pos

        residue_idx = data["protein_atom"].residue_idx.long()
        num_residues = int(data["protein_residue"].num_nodes)
        residue_pos = scatter_mean(
            pos_dict["protein_atom"],
            residue_idx,
            dim=0,
            dim_size=num_residues,
        )
        counts = torch.bincount(residue_idx, minlength=num_residues).to(
            device=residue_pos.device
        )
        missing_mask = counts == 0
        if bool(missing_mask.any()):
            residue_pos[missing_mask] = data["protein_residue"].pos.to(residue_pos.device)[missing_mask]
        return residue_pos


    def _refresh_protein_context(
        self,
        *,
        data: HeteroData,
        x_dict: dict[str, Tensor],
        pos_dict: dict[str, Tensor],
        device: torch.device,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        构建当前 block 使用的 residue / pocket 几何与 pocket 隐状态。

        flexible 模式下：
        - residue / pocket 坐标随当前 protein atom 坐标刷新
        - pocket 连续特征随当前几何刷新
        - pocket 隐表示通过 refresh MLP 融合当前语义与几何 summary
        """

        residue_pos = self._compute_current_residue_positions(data=data, pos_dict=pos_dict)
        residue_batch = self._get_node_batch(
            data, "protein_residue", residue_pos.size(0), device
        )
        protein_atom_batch = self._get_node_batch(
            data, "protein_atom", data["protein_atom"].num_nodes, device
        )
        pocket_pos = scatter_mean(
            residue_pos.detach(),
            residue_batch,
            dim=0,
            dim_size=batch_size,
        )

        if not self.fix_protein:
            pocket_x_cont = build_pocket_features(
                residue_x_cont=data["protein_residue"].x_cont.to(device),
                residue_pos=residue_pos,
                protein_atom_pos=pos_dict["protein_atom"].detach(),
                residue_batch=residue_batch,
                protein_atom_batch=protein_atom_batch,
                residue_esm_missing_mask=getattr(data["protein_residue"], "esm_missing_mask", None),
                esm_feature_start=len(PROTEIN_RESIDUE_CONT_SCHEMA),
                center=pocket_pos.detach(),
            )
            refreshed_pocket = self.protein_pocket_embedder(pocket_x_cont)
            refresh_input = torch.cat(
                [x_dict["protein_pocket"], refreshed_pocket], dim=-1
            )
            x_dict["protein_pocket"] = x_dict["protein_pocket"] + self.pocket_refresh_mlp(refresh_input)

        full_pos_dict: dict[str, Tensor] = {
            "ligand_atom": pos_dict["ligand_atom"],
            "protein_atom": pos_dict["protein_atom"],
            "ligand_molecule": scatter_mean(
                pos_dict["ligand_atom"].detach(),
                data["ligand_atom"].batch,
                dim=0,
                dim_size=batch_size,
            ),
            "protein_residue": residue_pos,
            "protein_pocket": pocket_pos,
        }
        return residue_pos, pocket_pos, full_pos_dict


    def forward(self, data: HeteroData, t: Tensor) -> dict[str, Any]:
        """
        前向传播

        Args:
            data: 异构图数据
            t: 时间步 [B]

        Returns:
            包含编码后特征、坐标及初始配体位置的字典
        """
        x_dict, pos_dict, initial_lig_pos = self._embed_inputs(data)

        t_emb = self.time_embedder(t)

        # 添加时间嵌入
        for nt, x in x_dict.items():

            if (
                hasattr(data[nt], "batch")
                and data[nt].batch is not None
                and data[nt].batch.numel() > 0
            ):
                x_dict[nt] = x + t_emb[data[nt].batch]

            elif x.numel() > 0 and t_emb.shape[0] == 1:
                x_dict[nt] = x + t_emb[0]

            elif x.numel() > 0:
                logger.warning(
                    f"Failed to broadcast time embedding for node type {nt}: "
                    f"num_nodes={x.shape[0]}, num_time_steps={t_emb.shape[0]}."
                )

        edge_dict = dict(data.edge_index_dict)

        # 保存初始位置用于速度计算（在 GNN 更新之前）
        pos_input: dict[str, Tensor] = {
            "ligand_atom": pos_dict["ligand_atom"].clone(),
            "protein_atom": pos_dict["protein_atom"].clone(),
        }

        # 运行 GNN 块
        device    = t.device
        lig_batch = data["ligand_atom"].batch
        B         = int(lig_batch.max().item()) + 1

        for block in self.gnn_blocks:
            residue_pos, _, full_pos_dict = self._refresh_protein_context(
                data=data,
                x_dict=x_dict,
                pos_dict=pos_dict,
                device=device,
                batch_size=B,
            )

            edge_dict = self._build_dynamic_inter_atom_edges(
                data=data,
                pos_dict=pos_dict,
                edge_dict=edge_dict,
            )
            if self.interaction_profile == "full":
                edge_dict = self._build_dynamic_ligand_residue_edges(
                    data=data,
                    pos_dict=pos_dict,
                    edge_dict=edge_dict,
                    residue_pos=residue_pos,
                )
            x_dict, pos_dict = self._run_block(
                cast(nn.ModuleDict, block), x_dict, pos_dict, edge_dict, full_pos_dict
            )

        # 计算累计位移：delta_pos = pos_out - pos_in
        displacement_dict: dict[str, Tensor] = {}

        for nt in pos_dict:

            if nt in pos_input:
                displacement_dict[nt] = pos_dict[nt] - pos_input[nt]

            else:
                displacement_dict[nt] = torch.zeros_like(pos_dict[nt])

        # 如果冻结蛋白，将蛋白位移设为零，坐标恢复为初始值
        if self.fix_protein and "protein_atom" in displacement_dict:
            displacement_dict["protein_atom"] = torch.zeros_like(displacement_dict["protein_atom"])
            pos_dict["protein_atom"] = pos_input["protein_atom"]

        return {
            "x_dict": x_dict,
            "pos_dict": pos_dict,
            "displacement_dict": displacement_dict,
            "initial_ligand_pos": initial_lig_pos,
        }
