"""
统一预测头。

负责输出亲和力、位阻、pose 质量和几何更新相关预测，
是模型读出层的核心实现。
"""


import logging
import math

import torch
from torch import nn, Tensor
from torch_cluster import radius
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from ehfnet.models.layers import GaussianRBF

logger = logging.getLogger(__name__)


class PredictionConstants:
    """
    预测头常量集合。

    集中管理预测头内部使用的固定超参数和维度约定，
    避免相关数值散落在不同预测分支实现中。
    """

    NUM_RBF = 50
    RBF_START = 0.0
    RBF_STOP = 10.0
    BASE_MAX_NEIGHBORS = 256
    MIN_MAX_NEIGHBORS = 64
    MIN_DISTANCE = 1e-4
    EPSILON = 1e-4
    BASELINE_BINDING_ENERGY = -7.0
    FORCE_CUTOFF = 6.0
    FORCE_LIMIT = 20.0


class CosineCutoff(nn.Module):
    """
    平滑截断函数。

    在 r = r_cutoff 时平滑衰减到 0。
    公式：f(r) = 0.5 * [cos(π * r / r_cutoff) + 1]  当 r < r_cutoff
              = 0                                  当 r >= r_cutoff
    """

    def __init__(self, cutoff: float) -> None:
        """
        初始化对象。

        Args:
            cutoff: 截断阈值。
        """

        super().__init__()
        self.cutoff = float(cutoff)


    def forward(self, dist: Tensor) -> Tensor:
        """
        前向传播

        Args:
            dist: dist。

        Returns:
            截断权重 [...]，范围 [0, 1]
        """

        dist_safe = torch.clamp(dist, min=0.0, max=self.cutoff)
        cutoff_values = 0.5 * (torch.cos(math.pi * dist_safe / self.cutoff) + 1.0)
        cutoff_values = torch.where(
            dist < self.cutoff, cutoff_values, torch.zeros_like(cutoff_values)
        )

        return cutoff_values


class PredictionHead(nn.Module):
    """
    物理一致的统一预测头

    架构设计理念：
    1. 能量势场：E(r) → 结合亲和力 score_norm [B, 1]
    2. 力场预测：F ≈ -∇E，通过 MLP 近似能量梯度
    3. 速度转换：v = F/m_eff，学习有效质量倒数

    物理一致性保证：
    - 力的方向 = 相对位置方向（几何约束）
    - 力的幅度由能量特征决定（物理关联）
    - 速度通过力场转换获得（牛顿第二定律简化）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_rbf: int = PredictionConstants.NUM_RBF,
        r_cutoff: float = PredictionConstants.RBF_STOP,
        dropout_rate: float = 0.1,
        max_neighbors: int = PredictionConstants.BASE_MAX_NEIGHBORS,
        min_max_neighbors: int = PredictionConstants.MIN_MAX_NEIGHBORS,
        force_cutoff: float = PredictionConstants.FORCE_CUTOFF,
        force_limit: float = PredictionConstants.FORCE_LIMIT,
        knn_fallback_k: int = 8,
        clash_threshold: float = 2.0,
        clash_push_threshold: float = 2.2,
        clash_push_force: float = 6.0,
        score_clamp_min: float = -50.0,
        score_clamp_max: float = 50.0,
    ) -> None:
        """
        初始化对象。

        Args:
            hidden_dim: 隐藏层维度。
            num_rbf: RBF 基函数数量。
            r_cutoff: 几何邻域构建的距离截断半径。
            dropout_rate: Dropout 比例。
            max_neighbors: 预测头阶段每个节点保留的最大邻居数。
            min_max_neighbors: 预测头动态邻居数的下限。
            force_cutoff: 力相关分支使用的局部截断半径。
            force_limit: 力大小的软限制。
            knn_fallback_k: 回退到 kNN 时使用的邻居数。
            clash_threshold: 位阻判定阈值。
            clash_push_threshold: 位阻推开分支使用的距离阈值。
            clash_push_force: 位阻推开分支的力缩放系数。
            score_clamp_min: 分数裁剪下界。
            score_clamp_max: 分数裁剪上界。
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.r_cutoff = float(r_cutoff)
        self.scale = hidden_dim**-0.5
        self.base_max_neighbors = int(max_neighbors)
        self.min_max_neighbors = max(1, int(min_max_neighbors))
        self.force_cutoff = float(min(force_cutoff, self.r_cutoff))
        self.force_limit = float(force_limit)
        self.knn_fallback_k = max(1, int(knn_fallback_k))
        self.clash_threshold = float(clash_threshold)
        self.clash_push_threshold = float(clash_push_threshold)
        self.clash_push_force = float(clash_push_force)
        self.score_clamp_min = float(score_clamp_min)
        self.score_clamp_max = float(score_clamp_max)

        self.adaptive_max_neighbors = True
        self.cutoff_fn = CosineCutoff(cutoff=self.r_cutoff)
        self.distance_expansion = GaussianRBF(0.0, self.r_cutoff, num_gaussians=num_rbf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.q_atom = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_atom = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_atom = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.bias_proj = nn.Linear(hidden_dim, 1)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.pairwise_energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.force_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.global_correction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.pose_rank_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + 6, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.norm = nn.LayerNorm(hidden_dim)


    def forward(
        self,
        lig_atom_feat: Tensor,
        lig_atom_pos: Tensor,
        lig_batch: Tensor,
        pro_atom_feat: Tensor,
        pro_atom_pos: Tensor,
        pro_atom_batch: Tensor,
        lig_mol_feat: Tensor,
    ) -> dict[str, Tensor]:
        """
        前向传播

        Args:
            lig_atom_feat: 配体原子特征。
            lig_atom_pos: 配体原子坐标张量。
            lig_batch: 配体所属的 batch 索引。
            pro_atom_feat: 蛋白原子特征。
            pro_atom_pos: 蛋白原子坐标张量。
            pro_atom_batch: 蛋白原子所属的 batch 索引。
            lig_mol_feat: 配体分子特征。

        Returns:
            dict[str, Tensor]: 含 binding_affinity、steric_clash_batch、局部相互作用上下文与类力信号的字典。
        """

        device = lig_atom_feat.device
        B = lig_mol_feat.size(0)
        N_lig = lig_atom_feat.size(0)
        N_pro = pro_atom_feat.size(0)
        atom_counts = scatter_add(
            torch.ones(N_lig, device=device, dtype=torch.float32),
            lig_batch,
            dim=0,
            dim_size=B,
        ).clamp(min=1.0)

        if N_lig == 0 or N_pro == 0:

            return {
                "binding_affinity": torch.zeros((B, 1), device=device, dtype=lig_atom_feat.dtype),
                "pose_rank_score": torch.zeros((B, 1), device=device, dtype=lig_atom_feat.dtype),
                "steric_clash_batch": torch.zeros(B, device=device, dtype=torch.float32),
                "ligand_interaction_context": torch.zeros_like(lig_atom_feat),
                "ligand_force": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        if self.adaptive_max_neighbors:
            max_k = min(
                self.base_max_neighbors,
                max(self.min_max_neighbors, N_pro // 4),
            )

        else:
            max_k = self.base_max_neighbors

        edge_index = radius(
            x=pro_atom_pos,
            y=lig_atom_pos,
            r=self.r_cutoff,
            batch_x=pro_atom_batch,
            batch_y=lig_batch,
            max_num_neighbors=max_k,
        )

        if edge_index.size(1) == 0:
            edge_index = self._build_knn_edges(
                lig_pos=lig_atom_pos,
                lig_batch=lig_batch,
                pro_pos=pro_atom_pos,
                pro_batch=pro_atom_batch,
                k=self.knn_fallback_k,
            )

        if edge_index.size(1) > 0:
            covered_lig = torch.unique(edge_index[0])
            all_lig = torch.arange(N_lig, device=device)
            uncovered_mask = torch.ones(N_lig, dtype=torch.bool, device=device)
            uncovered_mask[covered_lig] = False
            uncovered_lig = all_lig[uncovered_mask]

            if uncovered_lig.numel() > 0:
                extra_edges = self._build_knn_edges(
                    lig_pos=lig_atom_pos,
                    lig_batch=lig_batch,
                    pro_pos=pro_atom_pos,
                    pro_batch=pro_atom_batch,
                    k=1,
                    lig_indices=uncovered_lig,
                )

                if extra_edges.numel() > 0:
                    edge_index = torch.cat([edge_index, extra_edges], dim=1)
                    edge_index = torch.unique(edge_index, dim=1)

        if edge_index.size(1) == 0:

            return {
                "binding_affinity": torch.zeros((B, 1), device=device, dtype=lig_atom_feat.dtype),
                "pose_rank_score": torch.zeros((B, 1), device=device, dtype=lig_atom_feat.dtype),
                "steric_clash_batch": torch.zeros(B, device=device, dtype=torch.float32),
                "ligand_interaction_context": torch.zeros_like(lig_atom_feat),
                "ligand_force": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        i_idx = edge_index[0]
        j_idx = edge_index[1]

        lig_pos_sel = lig_atom_pos[i_idx].float()
        pro_pos_sel = pro_atom_pos[j_idx].float()
        sq_dist = torch.sum((lig_pos_sel - pro_pos_sel) ** 2, dim=-1)
        dist = torch.sqrt(sq_dist + 1e-8)
        clash_edge = torch.nn.functional.relu(self.clash_threshold - dist).pow(2)
        edge_count_per_atom = scatter_add(
            torch.ones_like(dist, dtype=torch.float32),
            i_idx,
            dim=0,
            dim_size=N_lig,
        ).clamp(min=1.0)
        clash_per_atom = scatter_add(
            clash_edge.float(),
            i_idx,
            dim=0,
            dim_size=N_lig,
        ) / edge_count_per_atom
        steric_clash_batch = (
            scatter_add(clash_per_atom, lig_batch, dim=0, dim_size=B) / atom_counts
        ).float()
        clash_hotspot_batch = torch.zeros(B, device=device, dtype=torch.float32)
        clash_hotspot_batch.scatter_reduce_(
            0,
            lig_batch,
            clash_per_atom,
            reduce="amax",
            include_self=True,
        )
        cutoff_weights = self.cutoff_fn(dist)

        if torch.all(cutoff_weights <= PredictionConstants.EPSILON):
            cutoff_weights = torch.exp(-dist / max(self.r_cutoff, PredictionConstants.EPSILON))

        rbf = self.distance_expansion(dist)
        edge_feat = self.edge_mlp(rbf)

        lig_feat_sel = lig_atom_feat[i_idx]
        pro_feat_sel = pro_atom_feat[j_idx]
        pair_input = torch.cat([lig_feat_sel, pro_feat_sel, edge_feat], dim=-1)
        rel_vec = pro_pos_sel - lig_pos_sel
        rel_dir = rel_vec / dist.unsqueeze(-1).clamp(min=PredictionConstants.MIN_DISTANCE)

        E_ij_raw = self.pairwise_energy_mlp(pair_input).squeeze(-1)
        E_ij = E_ij_raw * cutoff_weights

        force_mask = dist < self.force_cutoff
        if force_mask.any():
            force_edge_raw = self.force_mlp(pair_input[force_mask]).squeeze(-1)
            learned_force_mag = torch.zeros(
                edge_index.size(1), device=device, dtype=pair_input.dtype
            )
            learned_force_mag[force_mask] = (
                torch.tanh(force_edge_raw) * self.force_limit
            )
        else:
            learned_force_mag = torch.zeros(
                edge_index.size(1), device=device, dtype=pair_input.dtype
            )
        clash_push = torch.nn.functional.relu(self.clash_push_threshold - dist) * self.clash_push_force
        force_edge = (learned_force_mag.unsqueeze(-1) * rel_dir) - (clash_push.unsqueeze(-1) * rel_dir)

        edge_mass_per_atom = scatter_add(cutoff_weights, i_idx, dim=0, dim_size=N_lig)
        edge_mass_per_atom = edge_mass_per_atom.float().clamp(min=PredictionConstants.EPSILON)
        E_lig_atom = scatter_add(E_ij.float(), i_idx, dim=0, dim_size=N_lig) / edge_mass_per_atom

        E_physical_sum = scatter_add(E_lig_atom, lig_batch, dim=0, dim_size=B)
        E_physical = E_physical_sum / atom_counts.clamp(min=1.0)

        lig_atoms_with_neighbors = torch.unique(i_idx, sorted=False)
        pro_atoms_with_neighbors = torch.unique(j_idx, sorted=False)

        Q = self.q_atom(lig_atom_feat[lig_atoms_with_neighbors])
        K = self.k_atom(pro_atom_feat[pro_atoms_with_neighbors])
        V = self.v_atom(pro_atom_feat[pro_atoms_with_neighbors])

        lig_atom_to_subset = torch.full((N_lig,), -1, dtype=torch.long, device=device)
        lig_atom_to_subset[lig_atoms_with_neighbors] = torch.arange(
            len(lig_atoms_with_neighbors), device=device
        )

        pro_atom_to_subset = torch.full((N_pro,), -1, dtype=torch.long, device=device)
        pro_atom_to_subset[pro_atoms_with_neighbors] = torch.arange(
            len(pro_atoms_with_neighbors), device=device
        )

        i_subset_idx = lig_atom_to_subset[i_idx]
        j_subset_idx = pro_atom_to_subset[j_idx]

        Q_sel = Q[i_subset_idx]
        K_sel = K[j_subset_idx]
        V_sel = V[j_subset_idx]

        logits_raw = (Q_sel * K_sel).sum(dim=-1)
        geo_bias = self.bias_proj(edge_feat).squeeze(-1)
        logits = (logits_raw + geo_bias) * self.scale

        attn_weights = softmax(logits, i_idx)

        gate = self.gate_net(edge_feat).squeeze(-1)
        gate = gate * cutoff_weights

        weighted_V = V_sel * gate.unsqueeze(-1) * attn_weights.unsqueeze(-1)
        context_atom = scatter_add(weighted_V, i_idx, dim=0, dim_size=N_lig)
        context_atom = self.norm(context_atom + lig_atom_feat)
        force_atom = scatter_add(force_edge, i_idx, dim=0, dim_size=N_lig)
        force_atom = force_atom / edge_mass_per_atom.unsqueeze(-1).clamp(min=PredictionConstants.EPSILON)

        context_sum = scatter_add(context_atom, lig_batch, dim=0, dim_size=B)
        context_global = context_sum / atom_counts.clamp(min=1).unsqueeze(-1)
        force_atom_norm = torch.sqrt(
            force_atom.pow(2).sum(dim=-1) + PredictionConstants.EPSILON
        )
        force_norm_sum = scatter_add(force_atom_norm, lig_batch, dim=0, dim_size=B)
        force_norm_global = force_norm_sum / atom_counts.clamp(min=1.0)

        global_input = torch.cat([lig_mol_feat, context_global], dim=-1)
        E_correction = self.global_correction_mlp(global_input).squeeze(-1)
        score_norm = E_physical + E_correction
        score_norm = torch.clamp(
            score_norm,
            min=min(self.score_clamp_min, self.score_clamp_max),
            max=max(self.score_clamp_min, self.score_clamp_max),
        )

        binding_affinity = score_norm.unsqueeze(-1)

        lig_centroid = scatter_add(
            lig_atom_pos, lig_batch, dim=0, dim_size=B
        ) / atom_counts.clamp(min=1).unsqueeze(-1)
        pro_centroid = scatter_add(
            pro_atom_pos, pro_atom_batch, dim=0, dim_size=B
        ) / scatter_add(
            torch.ones(N_pro, device=device, dtype=torch.float32),
            pro_atom_batch, dim=0, dim_size=B,
        ).clamp(min=1).unsqueeze(-1)
        centroid_delta = lig_centroid - pro_centroid
        centroid_dist = torch.sqrt(
            centroid_delta.pow(2).sum(dim=-1, keepdim=True) + PredictionConstants.EPSILON
        )

        min_dist_per_lig = torch.full((N_lig,), float("inf"), device=device)
        if edge_index.size(1) > 0:
            min_dist_per_lig.scatter_reduce_(0, i_idx, dist.float(), reduce="amin", include_self=True)
        min_dist_per_lig = torch.where(
            torch.isfinite(min_dist_per_lig),
            min_dist_per_lig,
            torch.full_like(min_dist_per_lig, self.r_cutoff),
        )
        min_dist_per_mol = scatter_add(
            min_dist_per_lig, lig_batch, dim=0, dim_size=B
        ) / atom_counts.clamp(min=1)

        dist_per_atom = scatter_add(dist.float(), i_idx, dim=0, dim_size=N_lig) / edge_count_per_atom
        dist_sq_per_atom = scatter_add(dist.float().pow(2), i_idx, dim=0, dim_size=N_lig) / edge_count_per_atom
        dist_var_per_atom = (dist_sq_per_atom - dist_per_atom.pow(2)).clamp(min=0.0)
        dist_std_per_atom = torch.sqrt(dist_var_per_atom + PredictionConstants.EPSILON)

        dist_mean_edge = scatter_add(dist_per_atom, lig_batch, dim=0, dim_size=B) / atom_counts
        dist_std_edge = scatter_add(dist_std_per_atom, lig_batch, dim=0, dim_size=B) / atom_counts

        geo_features = torch.cat([
            centroid_dist,
            min_dist_per_mol.unsqueeze(-1),
            dist_mean_edge.unsqueeze(-1),
            dist_std_edge.unsqueeze(-1),
            (atom_counts / 50.0).unsqueeze(-1),
            clash_hotspot_batch.unsqueeze(-1),
        ], dim=-1)

        pose_rank_input = torch.cat(
            [
                global_input,
                binding_affinity,
                steric_clash_batch.unsqueeze(-1),
                force_norm_global.unsqueeze(-1),
                geo_features,
            ],
            dim=-1,
        )
        pose_rank_score = self.pose_rank_mlp(pose_rank_input)

        return {
            "binding_affinity": binding_affinity,
            "pose_rank_score": pose_rank_score,
            "steric_clash_batch": steric_clash_batch,
            "ligand_interaction_context": context_atom,
            "ligand_force": force_atom,
        }


    @staticmethod
    def _build_knn_edges(
        *,
        lig_pos: Tensor,
        lig_batch: Tensor,
        pro_pos: Tensor,
        pro_batch: Tensor,
        k: int,
        lig_indices: Tensor | None = None,
    ) -> Tensor:
        """
        构建批内 ligand->protein 的 kNN 边，返回格式与 radius 一致：[lig_idx, pro_idx]

        Returns:
            Tensor: 返回计算得到的张量结果。
        """

        device = lig_pos.device

        if lig_indices is None:
            lig_indices = torch.arange(lig_pos.size(0), device=device)

        if lig_indices.numel() == 0 or pro_pos.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        edges: list[Tensor] = []
        unique_batches = torch.unique(lig_batch[lig_indices])

        for b in unique_batches:
            lig_mask = lig_batch[lig_indices] == b
            lig_ids = lig_indices[lig_mask]
            pro_ids = torch.where(pro_batch == b)[0]

            if lig_ids.numel() == 0 or pro_ids.numel() == 0:
                continue

            d = torch.cdist(lig_pos[lig_ids], pro_pos[pro_ids])
            k_eff = min(k, int(pro_ids.numel()))
            nn_local = torch.topk(d, k=k_eff, largest=False, dim=1).indices

            lig_rep = lig_ids.repeat_interleave(k_eff)
            pro_sel = pro_ids[nn_local.reshape(-1)]
            edges.append(torch.stack([lig_rep, pro_sel], dim=0))

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        return torch.cat(edges, dim=1)
