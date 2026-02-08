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
    BASE_MAX_NEIGHBORS = 128        # 基础最大邻居数
    MIN_MAX_NEIGHBORS = 32          # 最小最大邻居数

    # 数值稳定性
    MIN_DISTANCE = 1e-6             # Å, 最小距离阈值
    EPSILON = 1e-8                  # 通用数值保护

    # 物理参数
    MIN_MASS_INV = 0.01             # 最小质量倒数（防止梯度消失）
    BASELINE_BINDING_ENERGY = -7.0  # kcal/mol, 典型结合能


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
    1. 能量势场：E(r) → 结合亲和力 ΔG [B, 1]
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
    ) -> None:
        """
        Args:
            hidden_dim: 隐藏层维度
            num_rbf: RBF 基函数数量
            r_cutoff: 截断距离（单位：Å）
            dropout_rate: Dropout 比例
            affinity_stats: 结合能统计数据 (mean, std) 用于反归一化
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.r_cutoff = float(r_cutoff)
        self.scale = hidden_dim**-0.5
        
        # 注册结合能标准化参数
        if affinity_stats:
            self.register_buffer("aff_mean", affinity_stats["mean"])
            self.register_buffer("aff_std", affinity_stats["std"])
        else:
            # 默认 fallback
            self.register_buffer("aff_mean", torch.tensor(6.0))
            self.register_buffer("aff_std", torch.tensor(1.5))

        # 动态调整邻居数
        self.adaptive_max_neighbors = True
        self.base_max_neighbors = PredictionConstants.BASE_MAX_NEIGHBORS

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

        # 力场预测分支
        self.force_magnitude_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 速度投影模块（力 → 速度）
        # 物理约束：质量只依赖于原子特征（类型、局部环境），不依赖于当前受到的力
        self.velocity_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # 初始化最后一层偏置
        for module in self.velocity_projection.modules():

            if isinstance(module, nn.Linear) and module.out_features == 1:

                if module.bias is not None:
                    # Change bias from -2.0 to 0.0 to prevent "vanishing velocity" at early stage.
                    # Softplus(0.0) ≈ 0.69 (large enough initial inverse mass)
                    # Softplus(-2.0) ≈ 0.13 (too small, causes slow movement)
                    nn.init.constant_(module.bias, 0.0)

                break

        self.min_mass_inv = PredictionConstants.MIN_MASS_INV

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
            包含 v_atomic, binding_affinity, force_atomic 的字典
        """

        device = lig_atom_feat.device
        B = lig_mol_feat.size(0)
        N_lig = lig_atom_feat.size(0)
        N_pro = pro_atom_feat.size(0)

        # 边界情况
        if N_lig == 0 or N_pro == 0:

            return {
                "v_atomic": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
                "binding_affinity": self.aff_mean.expand(B).unsqueeze(-1),
                "force_atomic": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        # 1. 邻居搜索
        if self.adaptive_max_neighbors:
            max_k = min(
                self.base_max_neighbors,
                max(PredictionConstants.MIN_MAX_NEIGHBORS, N_pro // 10),
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

            return {
                "v_atomic": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
                "binding_affinity": self.aff_mean.expand(B).unsqueeze(-1),
                "force_atomic": torch.zeros((N_lig, 3), device=device, dtype=lig_atom_feat.dtype),
            }

        i_idx = edge_index[0]
        j_idx = edge_index[1]

        # 2. 计算几何特征
        lig_pos_sel = lig_atom_pos[i_idx]
        pro_pos_sel = pro_atom_pos[j_idx]
        dist = torch.norm(lig_pos_sel - pro_pos_sel, dim=-1, p=2)

        cutoff_weights = self.cutoff_fn(dist)
        rbf = self.distance_expansion(dist)
        edge_feat = self.edge_mlp(rbf)

        lig_feat_sel = lig_atom_feat[i_idx]
        pro_feat_sel = pro_atom_feat[j_idx]

        # 3. 力场预测
        pair_input = torch.cat([lig_feat_sel, pro_feat_sel, edge_feat], dim=-1)

        force_magnitude = self.force_magnitude_mlp(pair_input)

        rel_pos = lig_pos_sel - pro_pos_sel
        direction = torch.nn.functional.normalize(
            rel_pos, dim=-1, eps=PredictionConstants.EPSILON
        )
        zero_mask = dist < PredictionConstants.MIN_DISTANCE
        direction = torch.where(
            zero_mask.unsqueeze(-1), torch.zeros_like(direction), direction
        )

        force_pairwise = force_magnitude * direction * cutoff_weights.unsqueeze(-1)
        force_atomic = scatter_add(force_pairwise, i_idx, dim=0, dim_size=N_lig)

        # 力 → 速度
        # lig_atom_feat 包含 atomic_weight 信息，网络可以学习质量倒数
        mass_inv_raw = self.velocity_projection(lig_atom_feat)
        mass_inv = mass_inv_raw + self.min_mass_inv
        v_atomic = force_atomic * mass_inv

        # 4. 能量预测
        E_ij_raw = self.pairwise_energy_mlp(pair_input).squeeze(-1)
        E_ij = E_ij_raw * cutoff_weights

        E_lig_atom = scatter_add(E_ij, i_idx, dim=0, dim_size=N_lig)
        E_physical = scatter_add(E_lig_atom, lig_batch, dim=0, dim_size=B)

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

        # 归一化聚合
        atom_counts = scatter_add(
            torch.ones(N_lig, device=device), lig_batch, dim=0, dim_size=B
        )
        context_sum = scatter_add(context_atom, lig_batch, dim=0, dim_size=B)
        context_global = context_sum / atom_counts.clamp(min=1).unsqueeze(-1)

        # 全局修正
        global_input = torch.cat([lig_mol_feat, context_global], dim=-1)
        E_correction = self.global_correction_mlp(global_input).squeeze(-1)

        # 最终能量 (Score)
        # 神经网络输出的是归一化的分数 (期望在 0 附近)
        score_norm = E_physical + E_correction
        
        # 反归一化，输出真实物理值
        binding_affinity = (score_norm * self.aff_std + self.aff_mean).unsqueeze(-1)

        return {
            "v_atomic": v_atomic,
            "binding_affinity": binding_affinity,
            "force_atomic": force_atomic,
        }
