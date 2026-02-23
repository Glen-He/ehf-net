"""
帧感知异构消息传递

为分层 EGNN 编码器的 Aggregate / Broadcast / Intra-feat / Inter-feat 阶段提供
SE(3)-不变的消息传递，彻底取代原 SAGEConv。

根本问题：
    SAGEConv 的消息 m_{i→j} = W · [h_i ‖ mean_nbr(h)] 完全忽略几何位置，
    导致 Aggregate/Broadcast 传播的特征不含方向信息。后续 EGNN 的坐标更新：
        Δx_i = Σ_j (x_i - x_j) · φ_e(m_{ij})
    中，调制系数 φ_e 依赖 h，但 h 里混入了非等变高层广播信息，造成等变性软破坏：
    坐标更新方向正确（等变），但幅度调制不再是旋转不变量。

修复原理（FrameAwareConv）：
    在消息中引入三类旋转不变几何特征，构成严格 SE(3)-不变的消息：

      ① h_src, h_dst            — 标量节点特征（SE(3)-不变量）
      ② RBF(d_{ij})             — 距离径向基函数特征（旋转不变量）
      ③ R̂_j^T · r̂_{ij}         — 在 dst 局部帧中表达的单位方向向量（旋转不变量）

    局部帧 R̂_j 由 dst 节点的邻域均值方向经 Gram-Schmidt 正交化构造，
    全向量化（无 Python 循环），前向开销极小。

    不变性证明（③）：
        全局旋转 Q 作用时：
          r̂_{ij}                → Q r̂_{ij}
          R̂_j（由 dst 邻域构造）→ Q R̂_j
          R̂_j^T r̂_{ij}         → (Q R̂_j)^T (Q r̂_{ij})
                                = R̂_j^T Q^T Q r̂_{ij}
                                = R̂_j^T r̂_{ij}         ✓ 不变

消息公式：
    m_{i→j} = φ_msg( h_i, h_j, RBF(d_{ij}), R̂_j^T r̂_{ij} )
                · σ( φ_gate( RBF(d_{ij}) ) )

    φ_msg  — LayerNorm + Linear + SiLU + Linear
    φ_gate — Linear + Sigmoid（距离衰减门控，近邻权重大）
    聚合    — scatter_mean（不受节点度数比例影响）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter_mean


# ─────────────────────────────────────────────────────────────────────────────
# 高斯 RBF 编码
# ─────────────────────────────────────────────────────────────────────────────

class GaussianRBF(nn.Module):
    """
    高斯径向基函数距离编码。

    将标量距离 d 扩展为高维特征向量：
        rbf_k(d) = exp(-0.5 · ((d - μ_k) / σ)²)

    μ_k 均匀分布在 [start, stop]，σ = 相邻中心间距。

    输出范围 (0, 1]，在 d = μ_k 处取最大值 1。
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 20.0,
        num_gaussians: int = 32,
    ) -> None:
        super().__init__()
        assert stop > start, f"stop ({stop}) must be > start ({start})"
        assert num_gaussians >= 4, f"num_gaussians must be >= 4, got {num_gaussians}"

        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        sigma = (stop - start) / (num_gaussians - 1)
        self.coeff = -0.5 / (sigma ** 2)

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: 距离标量 [...] 或 [..., 1]

        Returns:
            RBF 特征 [..., num_gaussians]
        """
        dist = dist.clamp(min=0.0)
        diff = dist.unsqueeze(-1) - self.offset          # type: ignore[arg-type]
        return torch.exp(self.coeff * diff.pow(2))


# ─────────────────────────────────────────────────────────────────────────────
# 局部帧构造（Gram-Schmidt，全向量化）
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_direction_frame(
    src_pos: Tensor,
    dst_pos: Tensor,
    edge_index: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    为每个 dst 节点计算基于邻域均值方向的正交参考帧。全向量化，无 Python 循环。

    构造步骤：
        1. e1 = normalize( mean(src_pos[N(j)]) - dst_pos[j] )   ← 邻域主方向
        2. e2 = normalize( ref - (e1·ref)·e1 )                   ← Gram-Schmidt 正交
        3. e3 = e1 × e2                                           ← 完成右手系

    帧矩阵 R = [e1 | e2 | e3] ∈ ℝ^{3×3}，
        列向量 = 主轴方向（body → world 坐标变换）
        R^T 将 world 帧向量投影至 body 帧（R 是正交矩阵，R^T = R^{-1}）

    注意：
        - 全程 .detach()，帧只作几何参考，梯度不通过帧路径传播。
        - 无邻居 / 主方向接近零向量时退化为单位矩阵（保持 SE(3)-不变性）。

    Args:
        src_pos:    源节点坐标 [N_src, 3]
        dst_pos:    目标节点坐标 [N_dst, 3]
        edge_index: 边索引 [2, E]，row0=src_idx, row1=dst_idx
        eps:        向量归一化的数值保护项

    Returns:
        R: [N_dst, 3, 3]，列向量为体坐标轴（body→world），已 detach
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    N_dst   = dst_pos.shape[0]
    device  = dst_pos.device
    dtype   = dst_pos.dtype

    # 默认帧：单位矩阵（无邻居节点也有合法帧）
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N_dst, -1, -1).clone()

    if src_idx.numel() == 0:
        return R.detach()

    src_pos_d = src_pos.detach()
    dst_pos_d = dst_pos.detach()

    # ── 计算每个 dst 节点的邻域均值位置（向量化 scatter_add） ─────────────────
    count = torch.zeros(N_dst, device=device, dtype=dtype)
    count.scatter_add_(0, dst_idx, torch.ones(dst_idx.shape[0], device=device, dtype=dtype))

    mean_src = torch.zeros(N_dst, 3, device=device, dtype=dtype)
    mean_src.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, 3), src_pos_d[src_idx])

    has_nbrs = count > 0
    valid_count = count[has_nbrs].unsqueeze(-1).clamp(min=1.0)
    mean_src[has_nbrs] = mean_src[has_nbrs] / valid_count

    # ── e1：dst → 邻域均值方向 ────────────────────────────────────────────────
    e1_raw  = mean_src - dst_pos_d                            # [N_dst, 3]
    e1_norm = e1_raw.norm(dim=-1, keepdim=True).clamp(min=eps)
    e1      = e1_raw / e1_norm                                # [N_dst, 3]

    # ── Gram-Schmidt：用参考轴构造与 e1 正交的 e2 ─────────────────────────────
    # 默认参考轴：x 轴；若 e1 与 x 轴近似平行（|cos| > 0.9），改用 y 轴
    ref = torch.zeros_like(e1)
    ref[:, 0] = 1.0
    parallel_with_x = e1[:, 0].abs() > 0.9
    ref[parallel_with_x, 0] = 0.0
    ref[parallel_with_x, 1] = 1.0

    dot    = (e1 * ref).sum(dim=-1, keepdim=True)             # [N_dst, 1]
    e2_raw = ref - dot * e1
    e2_norm = e2_raw.norm(dim=-1, keepdim=True).clamp(min=eps)
    e2     = e2_raw / e2_norm                                 # [N_dst, 3]

    # ── e3 = e1 × e2（右手系） ────────────────────────────────────────────────
    e3 = torch.linalg.cross(e1, e2)                           # [N_dst, 3]

    # ── 组装：列向量 = 主轴 ───────────────────────────────────────────────────
    R_computed = torch.stack([e1, e2, e3], dim=-1)            # [N_dst, 3, 3]
    R[has_nbrs] = R_computed[has_nbrs]

    return R.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 单边类型帧感知卷积
# ─────────────────────────────────────────────────────────────────────────────

class FrameAwareConv(nn.Module):
    """
    SE(3)-不变单边类型消息传递（帧感知版 SAGEConv）。

    消息组合（全为 SE(3)-旋转不变量）：

        m_{i→j} = φ_msg( h_i, h_j, RBF(d_{ij}), R̂_j^T r̂_{ij} )
                    · σ( φ_gate( RBF(d_{ij}) ) )

    字段说明：
        h_i, h_j        — 源/目标标量特征（不变量）
        RBF(d_{ij})     — 距离编码（不变量）
        R̂_j^T r̂_{ij}   — 目标局部帧中的方向分量（不变量，见模块注释证明）
        φ_gate          — 距离门控（Sigmoid），远距邻居自动降权
        scatter_mean    — 聚合（对度数不敏感，防高度数节点主导训练）

    输出：每个 dst 节点的聚合消息 [N_dst, H]。
    由调用方（encoder._apply_residual_update）完成残差更新。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_rbf: int = 32,
        cutoff: float = 20.0,
    ) -> None:
        """
        Args:
            hidden_dim: 节点特征维度 H（src/dst 均为 H，输出也是 H）
            num_rbf:    高斯 RBF 基函数数量
            cutoff:     RBF 覆盖的最大距离 Å（超出范围的边几乎无信号）
        """
        super().__init__()
        self.rbf    = GaussianRBF(0.0, cutoff, num_rbf)

        # 消息 MLP：输入 = [h_src ‖ h_dst ‖ RBF(d) ‖ r_body]
        #   维度 = H + H + num_rbf + 3 = 2H + num_rbf + 3
        msg_in_dim = hidden_dim * 2 + num_rbf + 3
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(msg_in_dim),
            nn.Linear(msg_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 距离门控：仅依赖 RBF 特征，输出 H 维 Sigmoid 门控
        # 物理含义：耦合强度随距离单调衰减
        self.dist_gate = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_src:      Tensor,   # [N_src, H]
        pos_src:    Tensor,   # [N_src, 3]
        h_dst:      Tensor,   # [N_dst, H]
        pos_dst:    Tensor,   # [N_dst, 3]
        edge_index: Tensor,   # [2, E]，row0=src_idx, row1=dst_idx
    ) -> Tensor:
        """
        Returns:
            aggr: 聚合消息 [N_dst, H]
        """
        N_dst  = h_dst.shape[0]
        H      = h_src.shape[1]
        device = h_src.device
        dtype  = h_src.dtype

        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return torch.zeros(N_dst, H, device=device, dtype=dtype)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # ── 几何特征（全为旋转不变量） ─────────────────────────────────────────
        r_vec    = pos_src[src_idx] - pos_dst[dst_idx]           # [E, 3]
        dist     = r_vec.norm(dim=-1)                             # [E]
        rbf_feat = self.rbf(dist)                                 # [E, num_rbf]

        # dst 局部帧：[N_dst, 3, 3]，列向量 = 体轴（body→world），已 detach
        R_dst = compute_mean_direction_frame(pos_src, pos_dst, edge_index)

        # 将世界帧方向投影至 dst 局部帧（旋转不变量）
        # r_body[e,i] = Σ_j R_dst[dst_idx[e], j, i] · r_hat[e, j]
        #             = (R^T @ r_hat)_i  （列 = 体轴 → R 是 body→world，R^T 是 world→body）
        r_hat  = F.normalize(r_vec, dim=-1, eps=1e-8)             # [E, 3]
        R_edge = R_dst[dst_idx]                                   # [E, 3, 3]
        r_body = torch.einsum("eji,ej->ei", R_edge, r_hat)       # [E, 3]

        # ── 消息计算（Gated MLP） ────────────────────────────────────────────
        msg_in = torch.cat(
            [h_src[src_idx], h_dst[dst_idx], rbf_feat, r_body], dim=-1
        )                                                         # [E, 2H+num_rbf+3]
        gate = self.dist_gate(rbf_feat)                           # [E, H]
        msg  = self.msg_mlp(msg_in) * gate                        # [E, H]

        # ── scatter_mean 聚合 ─────────────────────────────────────────────────
        aggr = scatter_mean(msg, dst_idx, dim=0, dim_size=N_dst)  # [N_dst, H]
        return aggr


# ─────────────────────────────────────────────────────────────────────────────
# 多边类型异构帧感知卷积
# ─────────────────────────────────────────────────────────────────────────────

class FrameAwareHeteroConv(nn.Module):
    """
    多边类型帧感知异构消息传递。

    接口约定（与 PyG HeteroConv 兼容，但额外需要 full_pos_dict）：
        forward(x_dict, full_pos_dict, edge_index_dict) → out_dict

    out_dict[dst_type] 是该类型节点聚合的消息张量 [N_dst, H]，
    可能来自多个 src 类型（相加合并），由调用方通过
    encoder._apply_residual_update 完成残差更新。

    注意：
        - nn.ModuleDict 要求字符串键，内部用 "__" 将三元组 (src,rel,dst) 拼接
        - 缺失的边类型静默跳过（不报错），确保动态边图下的鲁棒性
    """

    def __init__(
        self,
        convs: dict[tuple[str, str, str], FrameAwareConv],
    ) -> None:
        super().__init__()

        # 存入 ModuleDict（字符串化键）
        self._convs = nn.ModuleDict({
            "__".join(k): v for k, v in convs.items()
        })
        # 反向映射：字符串键 → 原始三元组
        self._edge_keys: dict[str, tuple[str, str, str]] = {
            "__".join(k): k for k in convs.keys()
        }

    def forward(
        self,
        x_dict:          dict[str, Tensor],
        full_pos_dict:   dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """
        Args:
            x_dict:          节点特征字典 {node_type: [N, H]}
            full_pos_dict:   节点坐标字典 {node_type: [N, 3]}（含 molecule/residue/pocket）
            edge_index_dict: 边索引字典   {(src,rel,dst): [2, E]}

        Returns:
            out_dict: {dst_type: aggregated_msg [N_dst, H]}
        """
        out: dict[str, Tensor] = {}

        for key_str, (src_t, rel, dst_t) in self._edge_keys.items():
            conv = self._convs[key_str]

            edge = edge_index_dict.get((src_t, rel, dst_t))
            if edge is None or edge.numel() == 0:
                continue

            if src_t not in x_dict or dst_t not in x_dict:
                continue

            src_pos = full_pos_dict.get(src_t)
            dst_pos = full_pos_dict.get(dst_t)
            if src_pos is None or dst_pos is None:
                continue

            msg = conv(
                x_dict[src_t], src_pos,
                x_dict[dst_t], dst_pos,
                edge,
            )  # [N_dst, H]

            if dst_t in out:
                out[dst_t] = out[dst_t] + msg
            else:
                out[dst_t] = msg

        return out
