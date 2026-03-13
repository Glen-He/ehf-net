"""
蛋白 pocket 特征构建工具。

将 residue 级连续特征与局部几何统计汇总为 pocket 级连续特征，
供 protein_pocket 节点显式消费。
"""

from __future__ import annotations

import torch

from torch import Tensor
from torch_scatter import scatter_mean


POCKET_GEOM_SCALAR_DIM = 6


def pocket_feature_dim(residue_cont_dim: int) -> int:
    """
    pocket 连续特征维度：
    - residue 连续特征均值 [D]
    - residue 连续特征标准差 [D]
    - 6 个几何/规模统计量
    """

    return residue_cont_dim * 2 + POCKET_GEOM_SCALAR_DIM


def build_pocket_features(
    *,
    residue_x_cont: Tensor,
    residue_pos: Tensor,
    protein_atom_pos: Tensor | None = None,
    residue_batch: Tensor | None = None,
    protein_atom_batch: Tensor | None = None,
    center: Tensor | None = None,
) -> Tensor:
    """
    从 residue 级特征构建 pocket 级连续特征。

    Args:
        residue_x_cont: [N_res, D]
        residue_pos: [N_res, 3]
        protein_atom_pos: [N_atom, 3]，用于 atom 数统计；可为空
        residue_batch: [N_res]，多图 batch 索引；为空时视为单图
        protein_atom_batch: [N_atom]，多图 batch 索引；为空时视为单图
        center: [B, 3] 或 [3]，用于计算到 pocket 中心的统计；为空时使用 residue 几何中心

    Returns:
        pocket_x_cont: [B, 2D + 6]
    """

    device = residue_x_cont.device
    dtype = residue_x_cont.dtype
    d_cont = residue_x_cont.size(-1)

    if residue_x_cont.numel() == 0:
        if center is not None:
            batch_size = int(center.size(0)) if center.ndim == 2 else 1
        elif residue_batch is not None and residue_batch.numel() > 0:
            batch_size = int(residue_batch.max().item()) + 1
        else:
            batch_size = 1
        return torch.zeros(
            (batch_size, pocket_feature_dim(d_cont)),
            device=device,
            dtype=dtype,
        )

    if residue_batch is None:
        residue_batch = torch.zeros(
            residue_x_cont.size(0), dtype=torch.long, device=device
        )

    batch_size = int(residue_batch.max().item()) + 1 if residue_batch.numel() > 0 else 1

    mean_feat = scatter_mean(
        residue_x_cont, residue_batch, dim=0, dim_size=batch_size
    )

    mean_sq_feat = scatter_mean(
        residue_x_cont.pow(2), residue_batch, dim=0, dim_size=batch_size
    )
    std_feat = (mean_sq_feat - mean_feat.pow(2)).clamp(min=0.0).sqrt()

    residue_counts = torch.bincount(residue_batch, minlength=batch_size).to(
        device=device, dtype=dtype
    ).unsqueeze(-1)

    if protein_atom_pos is not None:
        if protein_atom_batch is None:
            protein_atom_batch = torch.zeros(
                protein_atom_pos.size(0), dtype=torch.long, device=device
            )
        atom_counts = torch.bincount(protein_atom_batch, minlength=batch_size).to(
            device=device, dtype=dtype
        ).unsqueeze(-1)
    else:
        atom_counts = torch.zeros((batch_size, 1), device=device, dtype=dtype)

    residue_centroid = scatter_mean(
        residue_pos, residue_batch, dim=0, dim_size=batch_size
    )

    if center is None:
        center_ref = residue_centroid
    else:
        center_ref = center.to(device=device, dtype=residue_pos.dtype)
        if center_ref.ndim == 1:
            center_ref = center_ref.unsqueeze(0)
        if center_ref.size(0) != batch_size:
            raise ValueError(
                f"center batch size mismatch: expected {batch_size}, got {int(center_ref.size(0))}."
            )

    dist_to_center = torch.norm(
        residue_pos - center_ref[residue_batch], dim=-1
    )

    mean_radius = scatter_mean(
        dist_to_center.unsqueeze(-1), residue_batch, dim=0, dim_size=batch_size
    )
    mean_sq_radius = scatter_mean(
        dist_to_center.pow(2).unsqueeze(-1), residue_batch, dim=0, dim_size=batch_size
    )
    std_radius = (mean_sq_radius - mean_radius.pow(2)).clamp(min=0.0).sqrt()

    max_radius = torch.zeros((batch_size, 1), device=device, dtype=dtype)
    min_radius = torch.full((batch_size, 1), float("inf"), device=device, dtype=dtype)
    for graph_idx in range(batch_size):
        mask = residue_batch == graph_idx
        if not bool(mask.any()):
            min_radius[graph_idx] = 0.0
            continue
        graph_dist = dist_to_center[mask]
        max_radius[graph_idx] = graph_dist.max()
        min_radius[graph_idx] = graph_dist.min()

    geom_scalars = torch.cat(
        [
            residue_counts / 64.0,
            atom_counts / 256.0,
            mean_radius / 10.0,
            std_radius / 5.0,
            max_radius / 10.0,
            min_radius / 10.0,
        ],
        dim=-1,
    )

    return torch.cat([mean_feat, std_feat, geom_scalars], dim=-1)
