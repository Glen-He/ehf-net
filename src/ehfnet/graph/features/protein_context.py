"""
蛋白上下文特征工具。

负责构建 local context 节点的几何与统计特征，
为盲对接阶段的局部环境建模提供输入。
"""


import torch

from torch import Tensor
from torch_scatter import scatter_add, scatter_mean

from ehfnet.data.featurizers import (
    PROTEIN_RESIDUE_TORSION_DIM,
    PROTEIN_RESIDUE_TORSION_VALID_DIM,
    PROTEIN_RESIDUE_TORSION_VALID_START,
)


CONTEXT_GEOM_SCALAR_DIM = 7


def context_feature_dim(residue_cont_dim: int) -> int:
    """
    context 连续特征维度。

    - residue 连续特征均值 [D]
    - residue 连续特征标准差 [D]
    - 7 个几何/规模统计量（含 ESM 缺失率）

    Args:
        residue_cont_dim: 残基连续特征维度。

    Returns:
        int: 返回单个 context 节点连续特征的总维度。
    """

    return residue_cont_dim * 2 + CONTEXT_GEOM_SCALAR_DIM


def build_context_features(
    *,
    residue_x_cont: Tensor,
    residue_pos: Tensor,
    protein_atom_pos: Tensor | None = None,
    residue_batch: Tensor | None = None,
    protein_atom_batch: Tensor | None = None,
    residue_esm_missing_mask: Tensor | None = None,
    esm_feature_start: int | None = None,
    center: Tensor | None = None,
) -> Tensor:
    """
    从 residue 级特征构建 context 级连续特征。

    Args:
        residue_x_cont: 残基连续特征张量。
        residue_pos: 残基坐标张量。
        protein_atom_pos: 蛋白原子坐标张量。
        residue_batch: 残基所属 batch 索引。
        protein_atom_batch: 蛋白原子所属的 batch 索引。
        residue_esm_missing_mask: 标记残基 ESM 是否缺失的布尔掩码。
        esm_feature_start: 残基连续特征中 ESM 特征起始下标。
        center: 局部裁剪或特征汇总所围绕的中心坐标。

    Returns:
        Tensor: 返回按 batch 聚合后的 context 连续特征张量。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
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
            (batch_size, context_feature_dim(d_cont)),
            device=device,
            dtype=dtype,
        )

    if residue_batch is None:
        residue_batch = torch.zeros(
            residue_x_cont.size(0), dtype=torch.long, device=device
        )

    batch_size = int(residue_batch.max().item()) + 1 if residue_batch.numel() > 0 else 1

    residue_counts = torch.bincount(residue_batch, minlength=batch_size).to(
        device=device, dtype=dtype
    ).unsqueeze(-1)

    if (
        residue_esm_missing_mask is not None
        and esm_feature_start is not None
        and 0 < int(esm_feature_start) < d_cont
    ):
        esm_missing_mask = residue_esm_missing_mask.to(device=device, dtype=torch.bool)
        torsion_feat = residue_x_cont[:, :PROTEIN_RESIDUE_TORSION_DIM]
        context_feat = residue_x_cont[:, PROTEIN_RESIDUE_TORSION_DIM:esm_feature_start]
        esm_feat = residue_x_cont[:, esm_feature_start:]

        torsion_valid = residue_x_cont[
            :,
            PROTEIN_RESIDUE_TORSION_VALID_START : PROTEIN_RESIDUE_TORSION_VALID_START + PROTEIN_RESIDUE_TORSION_VALID_DIM,
        ].clamp(0.0, 1.0)
        torsion_mask = torsion_valid.repeat_interleave(2, dim=1)
        torsion_sum = scatter_add(
            torsion_feat * torsion_mask,
            residue_batch,
            dim=0,
            dim_size=batch_size,
        )
        torsion_sum_sq = scatter_add(
            torsion_feat.pow(2) * torsion_mask,
            residue_batch,
            dim=0,
            dim_size=batch_size,
        )
        torsion_count = scatter_add(
            torsion_mask,
            residue_batch,
            dim=0,
            dim_size=batch_size,
        ).clamp_min(1.0)
        torsion_mean = torsion_sum / torsion_count
        torsion_mean_sq = torsion_sum_sq / torsion_count
        torsion_std = (torsion_mean_sq - torsion_mean.pow(2)).clamp(min=0.0).sqrt()

        if context_feat.numel() > 0:
            context_mean = scatter_mean(
                context_feat, residue_batch, dim=0, dim_size=batch_size
            )
            context_mean_sq = scatter_mean(
                context_feat.pow(2), residue_batch, dim=0, dim_size=batch_size
            )
            context_std = (context_mean_sq - context_mean.pow(2)).clamp(min=0.0).sqrt()
        else:
            context_mean = torch.zeros((batch_size, 0), device=device, dtype=dtype)
            context_std = torch.zeros((batch_size, 0), device=device, dtype=dtype)

        esm_valid_mask = ~esm_missing_mask
        valid_counts = torch.bincount(
            residue_batch[esm_valid_mask], minlength=batch_size
        ).to(device=device, dtype=dtype).unsqueeze(-1)

        esm_sum = torch.zeros(
            (batch_size, d_cont - esm_feature_start), device=device, dtype=dtype
        )
        esm_sum_sq = torch.zeros_like(esm_sum)

        if bool(esm_valid_mask.any()):
            esm_sum = scatter_add(
                esm_feat[esm_valid_mask],
                residue_batch[esm_valid_mask],
                dim=0,
                dim_size=batch_size,
            )
            esm_sum_sq = scatter_add(
                esm_feat[esm_valid_mask].pow(2),
                residue_batch[esm_valid_mask],
                dim=0,
                dim_size=batch_size,
            )

        valid_counts_safe = valid_counts.clamp_min(1.0)
        esm_mean = esm_sum / valid_counts_safe
        esm_mean_sq = esm_sum_sq / valid_counts_safe
        esm_std = (esm_mean_sq - esm_mean.pow(2)).clamp(min=0.0).sqrt()

        no_valid_esm = valid_counts.squeeze(-1) == 0
        if bool(no_valid_esm.any()):
            esm_mean[no_valid_esm] = 0.0
            esm_std[no_valid_esm] = 0.0

        mean_feat = torch.cat([torsion_mean, context_mean, esm_mean], dim=-1)
        std_feat = torch.cat([torsion_std, context_std, esm_std], dim=-1)
        esm_missing_rate = 1.0 - (valid_counts / residue_counts.clamp_min(1.0))
    else:
        mean_feat = scatter_mean(
            residue_x_cont, residue_batch, dim=0, dim_size=batch_size
        )

        mean_sq_feat = scatter_mean(
            residue_x_cont.pow(2), residue_batch, dim=0, dim_size=batch_size
        )
        std_feat = (mean_sq_feat - mean_feat.pow(2)).clamp(min=0.0).sqrt()
        esm_missing_rate = torch.zeros((batch_size, 1), device=device, dtype=dtype)

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
            esm_missing_rate,
        ],
        dim=-1,
    )

    return torch.cat([mean_feat, std_feat, geom_scalars], dim=-1)
