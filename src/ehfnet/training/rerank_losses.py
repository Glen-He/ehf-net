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


def _pairwise_margin_ranking_core(
    logit_a: Tensor,
    logit_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    *,
    margin: float,
    min_delta: float = 0.05,
    extra_mask: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Pairwise margin ranking 核心逻辑。

    期望 logit_a[i] vs logit_b[i] 构成一对，target 越大表示质量越好。
    """
    qa = target_a.view(-1)
    qb = target_b.view(-1)
    sa = logit_a.view(-1)
    sb = logit_b.view(-1)
    delta = qa - qb
    valid = delta.abs() >= min_delta
    if extra_mask is not None:
        valid = valid & extra_mask.view(-1).to(device=valid.device, dtype=torch.bool)
    if not valid.any():
        return sa.new_zeros(()), 0
    direction = torch.sign(delta[valid])
    loss = F.relu(margin - direction * (sa[valid] - sb[valid])).mean()
    return loss, int(valid.sum().item())


def pairwise_ranking_loss_from_pairs(
    pose_logit_a: Tensor,
    pose_target_a: Tensor,
    pose_logit_b: Tensor,
    pose_target_b: Tensor,
    *,
    margin: float,
    min_delta: float = 0.05,
    extra_mask: Tensor | None = None,
) -> tuple[Tensor, int]:
    """从已配对 (logit_a, target_a) vs (logit_b, target_b) 计算 pairwise ranking loss。

    供 trainer 等调用。
    """
    return _pairwise_margin_ranking_core(
        pose_logit_a,
        pose_logit_b,
        pose_target_a,
        pose_target_b,
        margin=margin,
        min_delta=min_delta,
        extra_mask=extra_mask,
    )


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
        logit_a = logits[idx_a]
        logit_b = logits[idx_b]
        target_a = targets[idx_a]
        target_b = targets[idx_b]
    else:
        N = logits.size(0)
        idx_a = torch.arange(N, device=logits.device).repeat_interleave(N)
        idx_b = torch.arange(N, device=logits.device).repeat(N)
        mask_diag = idx_a != idx_b
        idx_a = idx_a[mask_diag]
        idx_b = idx_b[mask_diag]
        logit_a = logits[idx_a]
        logit_b = logits[idx_b]
        target_a = targets[idx_a]
        target_b = targets[idx_b]

    return _pairwise_margin_ranking_core(
        logit_a, logit_b, target_a, target_b,
        margin=margin,
        min_delta=min_delta,
    )


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
