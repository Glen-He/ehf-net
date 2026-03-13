"""
Reranker 训练损失函数。

提供 BCE、pairwise margin ranking 和 listwise (ListMLE) 损失，
用于从候选池重放数据训练 pose_quality / pose_rank 头。
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def rmsd_to_soft_target(rmsd: Tensor, center: float = 4.0, scale: float = 0.75) -> Tensor:
    """Smooth sigmoid mapping: low RMSD → high target."""
    return torch.sigmoid((center - rmsd) / scale)


def rerank_bce_loss(
    logits: Tensor,
    rmsd: Tensor,
    *,
    center: float = 4.0,
    scale: float = 0.75,
) -> Tensor:
    """Weighted BCE loss against RMSD-derived soft targets.

    Args:
        logits: model output logits [N]
        rmsd: ground-truth RMSD values [N]

    Returns:
        scalar loss
    """
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    targets = rmsd_to_soft_target(rmsd, center=center, scale=scale)
    weight = 1.0 + 2.0 * targets
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def rerank_pairwise_loss(
    logits: Tensor,
    rmsd: Tensor,
    *,
    margin: float = 0.5,
    min_delta: float = 0.05,
    pair_indices: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Pairwise margin ranking loss.

    If ``pair_indices`` is provided (shape [P, 2]), uses those pairs.
    Otherwise constructs all-vs-all pairs within the batch.

    Returns:
        (loss, num_valid_pairs)
    """
    if logits.numel() < 2:
        return logits.new_tensor(0.0), 0

    targets = rmsd_to_soft_target(rmsd)

    if pair_indices is not None:
        idx_a = pair_indices[:, 0]
        idx_b = pair_indices[:, 1]
    else:
        N = logits.size(0)
        idx_a = torch.arange(N, device=logits.device).repeat_interleave(N)
        idx_b = torch.arange(N, device=logits.device).repeat(N)
        mask_diag = idx_a != idx_b
        idx_a = idx_a[mask_diag]
        idx_b = idx_b[mask_diag]

    qa = targets[idx_a]
    qb = targets[idx_b]
    sa = logits[idx_a]
    sb = logits[idx_b]

    delta = qa - qb
    valid = delta.abs() >= min_delta
    if not valid.any():
        return logits.new_tensor(0.0), 0

    direction = torch.sign(delta[valid])
    loss = F.relu(margin - direction * (sa[valid] - sb[valid])).mean()
    return loss, int(valid.sum().item())


def rerank_listwise_loss(
    logits: Tensor,
    rmsd: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """ListMLE loss for listwise ranking.

    Sorts candidates by RMSD quality (best first) and computes the
    negative log-likelihood of the sorted permutation under a Plackett-Luce model.

    Args:
        logits: model output logits [N]
        rmsd: ground-truth RMSD [N]
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    if logits.numel() < 2:
        return logits.new_tensor(0.0)

    targets = rmsd_to_soft_target(rmsd)
    sorted_indices = torch.argsort(targets, descending=True)
    sorted_logits = logits[sorted_indices] / temperature

    N = sorted_logits.size(0)
    max_logits = sorted_logits.detach().max()
    shifted = sorted_logits - max_logits

    cumsums = torch.zeros(N, device=logits.device, dtype=logits.dtype)
    running = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    for i in range(N - 1, -1, -1):
        running = running + torch.exp(shifted[i])
        cumsums[i] = running

    log_cumsums = torch.log(cumsums + 1e-10) + max_logits
    loss = (log_cumsums - sorted_logits).mean()
    return loss


def compute_rerank_losses(
    logits: Tensor,
    rmsd: Tensor,
    *,
    margin: float = 0.5,
    pair_indices: Tensor | None = None,
    listwise_temperature: float = 1.0,
    lambda_bce: float = 1.0,
    lambda_pair: float = 1.0,
    lambda_list: float = 0.5,
) -> dict[str, Tensor]:
    """Compute combined BCE + pairwise + listwise losses.

    Returns dict with individual losses and combined total.
    """
    loss_bce = rerank_bce_loss(logits, rmsd)
    loss_pair, n_pairs = rerank_pairwise_loss(logits, rmsd, margin=margin, pair_indices=pair_indices)
    loss_list = rerank_listwise_loss(logits, rmsd, temperature=listwise_temperature)

    total = lambda_bce * loss_bce + lambda_pair * loss_pair + lambda_list * loss_list

    return {
        "rerank_bce": loss_bce,
        "rerank_pairwise": loss_pair,
        "rerank_listwise": loss_list,
        "rerank_total": total,
        "rerank_n_pairs": logits.new_tensor(float(n_pairs)),
    }


def compute_center_value_loss(
    center_logits: Tensor,
    center_value_targets: Tensor,
) -> Tensor:
    """BCE loss for center-value supervision from blind pool experience.

    Args:
        center_logits: predicted center value logits [N]
        center_value_targets: experience-based targets [N] in [0, 1]
            - 1.0 = center has pose < 2A (strong positive)
            - 0.5 = center best pose in 2-5A (weak positive)
            - 0.0 = all poses bad (negative)
    """
    if center_logits.numel() == 0:
        return center_logits.new_tensor(0.0)
    targets = center_value_targets.clamp(0.0, 1.0)
    weight = 1.0 + 2.0 * targets
    return F.binary_cross_entropy_with_logits(center_logits, targets, weight=weight)
