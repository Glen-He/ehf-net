"""
重排序损失工具。

负责计算 BCE、pairwise 和 listwise 损失，
服务 blind pool 回放阶段的排序学习。
"""


import torch
import torch.nn.functional as F
from torch import Tensor

SOFT_TARGET_CENTER = 4.0
SOFT_TARGET_SCALE = 0.75


def rmsd_to_soft_target(
    rmsd: Tensor,
    *,
    center: float = SOFT_TARGET_CENTER,
    scale: float = SOFT_TARGET_SCALE,
) -> Tensor:
    """
    将 RMSD 映射为软目标。

    用平滑函数把低 RMSD 转换为更高监督值，
    为重排序阶段的 BCE 与排序损失提供统一目标。

    Args:
        rmsd: 真实 RMSD 张量（Å）。
        center: 软目标 sigmoid 中心，RMSD 低于此值时监督值偏高。
        scale: 软目标 sigmoid 尺度，控制过渡陡峭程度。

    Returns:
        Tensor: 与 rmsd 同形状的软目标监督值，范围 (0, 1)。
    """
    return torch.sigmoid((center - rmsd) / scale)


def rerank_bce_loss(
    logits: Tensor,
    rmsd: Tensor,
    *,
    center: float = SOFT_TARGET_CENTER,
    scale: float = SOFT_TARGET_SCALE,
) -> Tensor:
    """
    基于 RMSD 软目标的加权 BCE 损失。

    Args:
        logits: 模型输出的打分 logits。
        rmsd: 真实 RMSD 张量（Å）。
        center: 软目标 sigmoid 中心参数。
        scale: 软目标 sigmoid 尺度参数。

    Returns:
        Tensor: 标量 BCE 损失。
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
    """
    Pairwise margin ranking 核心逻辑。

    期望 logit_a[i] vs logit_b[i] 构成一对，target 越大表示质量越好。

    Args:
        logit_a: 配对中第一侧预测 logit。
        logit_b: 配对中第二侧预测 logit。
        target_a: 配对中第一侧目标质量。
        target_b: 配对中第二侧目标质量。
        margin: 排序边界间隔。
        min_delta: 目标差异最小阈值，低于此不参与损失。
        extra_mask: 可选的有效样本掩码。

    Returns:
        tuple[Tensor, int]: 返回 pairwise ranking 损失与有效配对数量。
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
    """
    计算成对排序损失。

    基于已配对的 logit 与目标质量差异计算 pairwise ranking loss，
    供 blind pool 回放和其他排序训练场景直接复用。

    Args:
        pose_logit_a: 排序对中第一个构象的预测 logit。
        pose_target_a: 排序对中第一个构象的目标质量值。
        pose_logit_b: 排序对中第二个构象的预测 logit。
        pose_target_b: 排序对中第二个构象的目标质量值。
        margin: 排序损失中的最小边界间隔。
        min_delta: 仅在目标差异超过该阈值时才构造成对样本。
        extra_mask: 额外的有效样本掩码。

    Returns:
        tuple[Tensor, int]: 返回成对排序损失与参与计算的有效样本对数量。
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
    """
    Pairwise margin ranking 损失。

    若提供 `pair_indices`（形状 [P, 2]），则使用指定配对；
    否则在 batch 内构建全对全配对。

    Args:
        logits: 模型输出的打分 logits。
        rmsd: 真实 RMSD 张量。
        margin: 排序损失中的最小边界间隔。
        min_delta: 仅在目标差异超过该阈值时才构造成对样本。
        pair_indices: 显式指定的排序样本配对索引。

    Returns:
        tuple[Tensor, int]: 成对排序损失标量与有效配对数。
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
    """
    ListMLE listwise ranking 损失。

    按 RMSD 质量对候选 pose 排序（最优在前），
    在 Plackett-Luce 模型下计算排序排列的负对数似然。

    Args:
        logits: 模型输出的打分 logits。
        rmsd: 真实 RMSD 张量。
        temperature: 温度系数。

    Returns:
        Tensor: ListMLE 标量损失。
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
    """
    汇总重排序损失。

    统一计算 BCE、pairwise 和 listwise 三类损失并完成加权汇总，
    为排序训练阶段提供结构化损失输出。

    Args:
        logits: 模型输出的打分 logits。
        rmsd: 真实 RMSD 张量。
        margin: 排序损失中的最小边界间隔。
        pair_indices: 显式指定的排序样本配对索引。
        listwise_temperature: listwise 损失中的 softmax 温度。
        lambda_bce: BCE 损失权重。
        lambda_pair: pairwise 损失权重。
        lambda_list: listwise 损失权重。

    Returns:
        dict[str, Tensor]: 返回各项 rerank 损失分量及其加权汇总结果。
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
    """
    基于 blind pool 经验的 center-value 监督 BCE 损失。

    Args:
        center_logits: 中心对应的预测 logits。
        center_value_targets: 中心质量监督目标，范围 [0, 1]。

    Returns:
        Tensor: 标量 BCE 损失。
    """
    if center_logits.numel() == 0:
        return center_logits.new_tensor(0.0)
    targets = center_value_targets.clamp(0.0, 1.0)
    weight = 1.0 + 2.0 * targets
    return F.binary_cross_entropy_with_logits(center_logits, targets, weight=weight)
