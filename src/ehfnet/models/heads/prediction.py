"""
预测头模块

物理一致的统一预测头，基于能量-力-速度链条。
"""

import logging
import math
import torch

from torch import nn, Tensor
from torch_cluster import radius
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

logger = logging.getLogger(__name__)


# 物理常量
class PredictionConstants:
    """
    预测头相关常量
    """

    # RBF 参数
    NUM_RBF = 50                    # 径向基函数数量
    RBF_START = 0.0                 # Å
    RBF_STOP = 10.0                 # Å, 非键相互作用截断距离

    # 邻居搜索
    BASE_MAX_NEIGHBORS = 256        # 基础最大邻居数（提升以减少高密度样本截断）
    MIN_MAX_NEIGHBORS = 64          # 最小最大邻居数

    # 数值稳定性
    MIN_DISTANCE = 1e-4             # Å, 最小距离阈值（提升 FP16 兼容性）
    EPSILON = 1e-4                  # 通用数值保护（提升 FP16 兼容性）

    # 物理参数
    MIN_MASS_INV = 0.01             # 保留兼容（当前不使用）
    BASELINE_BINDING_ENERGY = -7.0  # kcal/mol, 典型结合能
    FORCE_CUTOFF = 6.0              # Å, 力场局部相互作用半径
    FORCE_LIMIT = 20.0              # 力幅值软饱和上限


class GaussianSmearing(nn.Module):
    """
    RBF (径向基函数) 编码层

    将标量距离扩展为高维特征，采用高斯函数：exp(-0.5 * ((d - μ_i) / σ)^2)
    """

    def __init__(
        self,
        start: float = PredictionConstants.RBF_START,
        stop: float = PredictionConstants.RBF_STOP,
        num_gaussians: int = PredictionConstants.NUM_RBF,
    ) -> None:
        """
        Args:
            start: RBF 起始距离（单位：Å）
            stop: RBF 截断距离（单位：Å）
            num_gaussians: 高斯基函数数量
        """

        super().__init__()

        if num_gaussians < 10:
            raise ValueError(
                f"num_gaussians is too small; expected >= 10, got {num_gaussians}."
            )

        if stop <= start:
            raise ValueError(f"stop ({stop}) must be greater than start ({start}).")

        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)

        # 高斯宽度：相邻中心间距
        sigma = (stop - start) / (num_gaussians - 1)
        self.coeff = -0.5 / (sigma**2)
        self.offset: Tensor


    def forward(self, dist: Tensor) -> Tensor:
        """
        前向传播

        Args:
            dist: 距离标量 [N] 或 [..., 1]（单位：Å），应为非负值

        Returns:
            RBF 特征 [..., num_gaussians]，范围 [0, 1]
        """

        # 确保距离非负（数值稳定性）
        dist = torch.clamp(dist, min=0.0)

        # 扩展维度以广播
        if dist.ndim == 1:
            dist_exp = dist.unsqueeze(-1)

        else:
            dist_exp = dist

        diff = dist_exp - self.offset
        rbf = torch.exp(self.coeff * diff.pow(2))

        return rbf


class CosineCutoff(nn.Module):
    """
    平滑截断函数

    在 r = r_cutoff 时平滑衰减到 0。
    公式：f(r) = 0.5 * [cos(π * r / r_cutoff) + 1]  当 r < r_cutoff
              = 0                                  当 r >= r_cutoff
    """

    def __init__(self, cutoff: float) -> None:
        """
        Args:
            cutoff: 截断距离（单位：Å）
        """

        super().__init__()
        self.cutoff = float(cutoff)


    def forward(self, dist: Tensor) -> Tensor:
        """
        前向传播

        Args:
            dist: 距离张量 [...]

        Returns:
            截断权重 [...]，范围 [0, 1]
        """

        # 避免数值问题：dist 应在 [0, cutoff] 范围内
        dist_safe = torch.clamp(dist, min=0.0, max=self.cutoff)

        # 余弦截断
        cutoff_values = 0.5 * (torch.cos(math.pi * dist_safe / self.cutoff) + 1.0)

        # 超过 cutoff 的设为 0
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
        affinity_stats: dict | None = None,
        max_neighbors: int = PredictionConstants.BASE_MAX_NEIGHBORS,
        force_cutoff: float = PredictionConstants.FORCE_CUTOFF,
        force_limit: float = PredictionConstants.FORCE_LIMIT,
        knn_fallback_k: int = 8,
    ) -> None:
        """
        Args:
            hidden_dim: 隐藏层维度
            num_rbf: RBF 基函数数量
            r_cutoff: 截断距离（单位：Å）
            dropout_rate: Dropout 比例
            affinity_stats: 保留兼容参数（当前不在模型内做反归一化）
            max_neighbors: 跨图邻居上限，过小会导致高密度样本信息截断
            force_cutoff: 力分支局部交互半径（Å）
            force_limit: 力幅值软饱和上限
            knn_fallback_k: 半径边缺失时的最近邻回退数量
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.r_cutoff = float(r_cutoff)
        self.scale = hidden_dim**-0.5
        self.base_max_neighbors = int(max_neighbors)
        self.force_cutoff = float(min(force_cutoff, self.r_cutoff))
        self.force_limit = float(force_limit)
        self.knn_fallback_k = max(1, int(knn_fallback_k))
        
        # 兼容保留：亲和力统计由 trainer/数据集侧用于反归一化评估
        _ = affinity_stats

        # 动态调整邻居数：保证高密度样本不被过度截断
        self.adaptive_max_neighbors = True

        # 共享模块
        self.cutoff_fn = CosineCutoff(cutoff=self.r_cutoff)

        # 距离 RBF 编码 + 边特征 MLP
        self.distance_expansion = GaussianSmearing(0.0, self.r_cutoff, num_rbf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 能量预测分支
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
            lig_atom_feat: 配体原子特征 [N_lig, H]
            lig_atom_pos: 配体原子坐标 [N_lig, 3]
            lig_batch: 配体原子批次索引 [N_lig]
            pro_atom_feat: 蛋白原子特征 [N_pro, H]
            pro_atom_pos: 蛋白原子坐标 [N_pro, 3]
            pro_atom_batch: 蛋白原子批次索引 [N_pro]
            lig_mol_feat: 配体分子特征 [B, H]

        Returns:
            binding_affinity、steric_clash_batch、局部相互作用上下文与 force-like 信号的字典
        """

        device = lig_atom_feat.device
        B = lig_mol_feat.size(0)
        N_lig = lig_atom_feat.size(0)
        N_pro = pro_atom_feat.size(0)

        # 边界情况
        if N_lig == 0 or N_pro == 0:

            return {
                "binding_affinity": torch.zeros((B, 1), device=device, dtype=lig_atom_feat.dtype),
                "ligand_interaction_context": torch.zeros_like(lig_atom_feat),
                "ligand_force": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        # 1. 邻居搜索
        if self.adaptive_max_neighbors:
            # 对高密度口袋场景提高上限，减少半径图邻接被截断
            max_k = min(
                self.base_max_neighbors,
                max(PredictionConstants.MIN_MAX_NEIGHBORS, N_pro // 4),
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

        # 半径边为空时，使用批内 kNN 回退，保证配体原子至少有跨图连接
        if edge_index.size(1) == 0:
            edge_index = self._build_knn_edges(
                lig_pos=lig_atom_pos,
                lig_batch=lig_batch,
                pro_pos=pro_atom_pos,
                pro_batch=pro_atom_batch,
                k=self.knn_fallback_k,
            )

        # 若半径图遗漏了部分配体原子，也为遗漏原子补 1-NN 边
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
                "steric_clash_batch": torch.zeros(B, device=device, dtype=torch.float32),
                "ligand_interaction_context": torch.zeros_like(lig_atom_feat),
                "ligand_force": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        i_idx = edge_index[0]
        j_idx = edge_index[1]

        # 2. 计算几何特征（升为 FP32 确保 AMP 下数值稳定）
        lig_pos_sel = lig_atom_pos[i_idx].float()
        pro_pos_sel = pro_atom_pos[j_idx].float()
        # [修复] 用手动 sqrt(sum²+ε) 代替 torch.norm：
        # torch.norm 在距离=0 处梯度为 x/‖x‖，分母为零 → grad_norm=nan；
        # 加上 1e-8 偏移量使导数始终有界，彻底消除 nan 梯度。
        sq_dist = torch.sum((lig_pos_sel - pro_pos_sel) ** 2, dim=-1)
        dist = torch.sqrt(sq_dist + 1e-8)
        # 软位阻排斥惩罚 (Soft Steric Clash)
        # 阈值 2.0 Å ≈ 两个碳原子范德华半径之和的保守估计
        # ReLU 保证只有小于阈值的距离才产生惩罚；平方保证梯度平滑无跳跃
        _clash_threshold = 2.0
        clash_edge = torch.nn.functional.relu(_clash_threshold - dist).pow(2)   # [E]
        # 映射边 → 配体原子 → 分子，得到每分子的总体碰撞量 [B]
        steric_clash_batch = scatter_add(
            clash_edge, lig_batch[i_idx], dim=0, dim_size=B
        ).float()
        cutoff_weights = self.cutoff_fn(dist)

        # 当全部边都落在 cutoff 外时，启用长程衰减权重，避免“有边无信号”
        if torch.all(cutoff_weights <= PredictionConstants.EPSILON):
            cutoff_weights = torch.exp(-dist / max(self.r_cutoff, PredictionConstants.EPSILON))

        rbf = self.distance_expansion(dist)
        edge_feat = self.edge_mlp(rbf)

        lig_feat_sel = lig_atom_feat[i_idx]
        pro_feat_sel = pro_atom_feat[j_idx]
        pair_input = torch.cat([lig_feat_sel, pro_feat_sel, edge_feat], dim=-1)
        rel_vec = pro_pos_sel - lig_pos_sel
        rel_dir = rel_vec / dist.unsqueeze(-1).clamp(min=PredictionConstants.MIN_DISTANCE)

        # 3. 能量预测
        E_ij_raw = self.pairwise_energy_mlp(pair_input).squeeze(-1)
        E_ij = E_ij_raw * cutoff_weights

        learned_force_mag = torch.tanh(self.force_mlp(pair_input).squeeze(-1)) * self.force_limit
        clash_push = torch.nn.functional.relu(2.2 - dist) * 6.0
        force_edge = (learned_force_mag.unsqueeze(-1) * rel_dir) - (clash_push.unsqueeze(-1) * rel_dir)

        # 先在配体原子维度按有效边权做归一化，再在样本维度做均值归一化
        edge_mass_per_atom = scatter_add(cutoff_weights, i_idx, dim=0, dim_size=N_lig)
        edge_mass_per_atom = edge_mass_per_atom.float().clamp(min=PredictionConstants.EPSILON)
        E_lig_atom = scatter_add(E_ij.float(), i_idx, dim=0, dim_size=N_lig) / edge_mass_per_atom

        E_physical_sum = scatter_add(E_lig_atom, lig_batch, dim=0, dim_size=B)
        atom_counts = scatter_add(
            torch.ones(N_lig, device=device, dtype=torch.float32), lig_batch, dim=0, dim_size=B
        )
        E_physical = E_physical_sum / atom_counts.clamp(min=1.0)

        # 交叉注意力
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

        # 归一化聚合
        context_sum = scatter_add(context_atom, lig_batch, dim=0, dim_size=B)
        context_global = context_sum / atom_counts.clamp(min=1).unsqueeze(-1)

        # 全局修正
        global_input = torch.cat([lig_mol_feat, context_global], dim=-1)
        E_correction = self.global_correction_mlp(global_input).squeeze(-1)

        # 最终能量 (Score)
        # 神经网络输出归一化分数（与 y_energy 对齐）
        score_norm = E_physical + E_correction

        # 物理软截断：E_physical 和 E_correction 在初期权重随机时均可爆炸
        # 真实 pKd 范围 [2, 15]，将原始分数限制在 [-50, 50] 防止 Huber Loss 被击穿
        score_norm = torch.clamp(score_norm, min=-50.0, max=50.0)

        # 模型端不做反归一化：保持训练目标与输出标度一致
        binding_affinity = score_norm.unsqueeze(-1)

        return {
            "binding_affinity": binding_affinity,
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
