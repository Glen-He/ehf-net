"""
候选构象标签工具。

用于离线计算 candidate RMSD 和生成质量标签，避免将 GT 派生信息混入在线模型输入。
"""

from __future__ import annotations

import torch

from torch import Tensor


def compute_candidate_rmsd(candidate_pos: Tensor, gt_pos: Tensor) -> Tensor:
    """
    计算候选构象相对 GT 的逐 candidate RMSD。

    Args:
        candidate_pos: [K, N, 3]
        gt_pos: [N, 3]

    Returns:
        [K] RMSD
    """

    if candidate_pos.ndim != 3:
        raise ValueError(f"candidate_pos must have shape [K, N, 3], got {tuple(candidate_pos.shape)}")

    if gt_pos.ndim != 2:
        raise ValueError(f"gt_pos must have shape [N, 3], got {tuple(gt_pos.shape)}")

    if candidate_pos.size(1) != gt_pos.size(0):
        raise ValueError(
            f"candidate atom count ({candidate_pos.size(1)}) does not match gt atom count ({gt_pos.size(0)})"
        )

    sq_diff = (candidate_pos - gt_pos.unsqueeze(0)).pow(2).sum(dim=-1)
    return torch.sqrt(sq_diff.mean(dim=-1))


def make_quality_labels(rmsd: Tensor, *, near_native_cutoff: float = 2.0, medium_cutoff: float = 5.0) -> Tensor:
    """
    生成三分类质量标签。

    标签定义：
    - 0: bad (> medium_cutoff)
    - 1: medium (near_native_cutoff, medium_cutoff]
    - 2: near-native (<= near_native_cutoff)
    """

    labels = torch.zeros_like(rmsd, dtype=torch.long)
    labels = torch.where(rmsd <= medium_cutoff, torch.ones_like(labels), labels)
    labels = torch.where(rmsd <= near_native_cutoff, torch.full_like(labels, 2), labels)
    return labels


def make_near_native_labels(rmsd: Tensor, *, cutoff: float = 2.0) -> Tensor:
    """
    生成二分类 near-native 标签。
    """

    return (rmsd <= cutoff).to(dtype=torch.float32)