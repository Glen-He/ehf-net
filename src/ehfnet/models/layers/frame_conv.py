"""
帧感知卷积层。

提供结合局部参考帧的异构消息传递算子，
用于几何感知的特征更新与聚合。
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter_mean

from ehfnet.models.layers.rbf import GaussianRBF


def compute_mean_direction_frame(
    src_pos: Tensor,
    dst_pos: Tensor,
    edge_index: Tensor,
    *,
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
        src_pos: 源节点坐标，形状 [N_src, 3]。
        dst_pos: 目标节点坐标，形状 [N_dst, 3]。
        edge_index: 边索引张量，形状 [2, E]。
        eps: 数值稳定用的小常数，用于避免除零。

    Returns:
        R: [N_dst, 3, 3]，列向量为体坐标轴（body→world），已 detach
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    N_dst   = dst_pos.shape[0]
    device  = dst_pos.device
    dtype   = dst_pos.dtype

    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N_dst, -1, -1).clone()

    if src_idx.numel() == 0:
        return R.detach()

    src_pos_d = src_pos.detach()
    dst_pos_d = dst_pos.detach()

    count = torch.zeros(N_dst, device=device, dtype=dtype)
    count.scatter_add_(0, dst_idx, torch.ones(dst_idx.shape[0], device=device, dtype=dtype))

    mean_src = torch.zeros(N_dst, 3, device=device, dtype=dtype)
    mean_src.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, 3), src_pos_d[src_idx])

    has_nbrs = count > 0
    valid_count = count[has_nbrs].unsqueeze(-1).clamp(min=1.0)
    mean_src[has_nbrs] = mean_src[has_nbrs] / valid_count

    e1_raw = mean_src - dst_pos_d
    e1_norm = e1_raw.norm(dim=-1, keepdim=True).clamp(min=eps)
    e1 = e1_raw / e1_norm

    degenerate_frame = has_nbrs & (e1_norm.squeeze(-1) < eps * 10)
    if degenerate_frame.any():
        has_nbrs = has_nbrs & ~degenerate_frame

    ref = torch.zeros_like(e1)
    ref[:, 0] = 1.0
    parallel_with_x = e1[:, 0].abs() > 0.9
    ref[parallel_with_x, 0] = 0.0
    ref[parallel_with_x, 1] = 1.0

    dot = (e1 * ref).sum(dim=-1, keepdim=True)
    e2_raw = ref - dot * e1
    e2_norm = e2_raw.norm(dim=-1, keepdim=True).clamp(min=eps)
    e2 = e2_raw / e2_norm

    e3 = torch.linalg.cross(e1, e2)

    R_computed = torch.stack([e1, e2, e3], dim=-1)
    R[has_nbrs] = R_computed[has_nbrs]

    return R.detach()


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
        初始化对象。

        Args:
            hidden_dim: 隐藏层维度。
            num_rbf: RBF 基函数数量。
            cutoff: 截断阈值。
        """
        super().__init__()
        self.rbf    = GaussianRBF(0.0, cutoff, num_gaussians=num_rbf)

        msg_in_dim = hidden_dim * 2 + num_rbf + 3
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(msg_in_dim),
            nn.Linear(msg_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dist_gate = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_src: Tensor,
        pos_src: Tensor,
        h_dst: Tensor,
        pos_dst: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        执行帧感知消息传递。

        结合局部参考帧、距离编码与节点特征计算单类边的聚合消息，
        用于几何感知的节点更新。

        Args:
            h_src: 源节点特征张量。
            pos_src: 源节点坐标张量。
            h_dst: 目标节点特征张量。
            pos_dst: 目标节点坐标张量。
            edge_index: 边索引张量。

        Returns:
            aggr: 聚合消息 [N_dst, H]
        """
        N_dst = h_dst.shape[0]
        H = h_src.shape[1]
        device = h_src.device
        dtype = h_src.dtype

        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return torch.zeros(N_dst, H, device=device, dtype=dtype)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        r_vec = pos_src[src_idx] - pos_dst[dst_idx]
        dist = torch.sqrt((r_vec ** 2).sum(dim=-1) + 1e-8)
        rbf_feat = self.rbf(dist)

        R_dst = compute_mean_direction_frame(pos_src, pos_dst, edge_index)
        r_hat = F.normalize(r_vec, dim=-1, eps=1e-8)
        R_edge = R_dst[dst_idx]
        r_body = torch.einsum("eji,ej->ei", R_edge, r_hat)

        msg_in = torch.cat(
            [h_src[src_idx], h_dst[dst_idx], rbf_feat, r_body], dim=-1
        )
        gate = self.dist_gate(rbf_feat)
        msg = self.msg_mlp(msg_in) * gate

        aggr = scatter_mean(msg, dst_idx, dim=0, dim_size=N_dst)
        return aggr


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
        """
        初始化异构帧感知卷积层。

        按边类型组织多组帧感知卷积模块，
        为异构图上的多关系消息传递做准备。

        Args:
            convs: 按边类型 (src, rel, dst) 组织的帧感知卷积模块字典。
        """
        super().__init__()

        self._convs = nn.ModuleDict({
            "__".join(k): v for k, v in convs.items()
        })
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
        执行异构帧感知消息传递。

        遍历不同边类型的卷积模块并汇总结果，
        输出各节点类型的聚合消息字典。

        Args:
            x_dict: 各节点类型的标量特征字典。
            full_pos_dict: 各节点类型的坐标字典。
            edge_index_dict: 各边类型的边索引字典。

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
            )

            if dst_t in out:
                out[dst_t] = out[dst_t] + msg
            else:
                out[dst_t] = msg

        return out
