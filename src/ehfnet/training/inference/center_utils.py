"""
中心推理工具。

负责中心提议、候选筛选和多分数融合，
为局部对接阶段提供候选中心。
"""


from typing import Any

import torch

from ehfnet.models import EHFNet


DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "pose_weight": 1.0,
    "center_weight": 0.35,
    "aff_weight": 0.08,
    "clash_weight": 0.12,
    "bias": 0.0,
}


def compute_center_guidance_scores(
    proposal_logits: torch.Tensor,
    residue_prior_feat: torch.Tensor | None,
    *,
    learned_score_fraction: float,
) -> torch.Tensor:
    """
    计算中心引导分数。

    将模型输出的提议 logit 与残基先验按进度平滑融合，
    用于候选中心选择阶段的打分。

    Args:
        proposal_logits: 中心提议分支输出的 logits。
        residue_prior_feat: 残基级几何先验特征。
        learned_score_fraction: 学习分数的融合占比，0 表示纯先验，1 表示纯学习分数。

    Returns:
        Tensor: 用于中心排序的最终引导分数张量。
    """
    learned_scores = proposal_logits.view(-1)
    alpha = float(max(0.0, min(1.0, learned_score_fraction)))
    if residue_prior_feat is None or residue_prior_feat.numel() == 0:
        return learned_scores

    if residue_prior_feat.ndim != 2 or residue_prior_feat.size(-1) < 4:
        return learned_scores

    density = residue_prior_feat[:, 0]
    cavity = residue_prior_feat[:, 3]
    heuristic_prob = (0.30 * density + 0.70 * cavity).clamp(1e-4, 1.0 - 1e-4)
    heuristic_scores = torch.logit(heuristic_prob)
    if alpha <= 0.0:
        return heuristic_scores
    if alpha >= 1.0:
        return learned_scores
    return torch.lerp(heuristic_scores, learned_scores, alpha)


def compute_center_guidance_fraction(
    *,
    progress: float,
    learned_start: float,
    learned_end: float,
) -> float:
    """
    计算中心引导中学习分数的融合占比。

    通过平滑的课程调度控制 heuristic prior 与 learned score 的交接，
    避免训练过程中因过早全量切换导致局部裁剪中心突然失稳。

    Args:
        progress: 当前训练进度，范围 [0, 1]。
        learned_start: 学习分数开始进入融合的训练进度。
        learned_end: 学习分数完全接管的训练进度。

    Returns:
        float: 学习分数融合占比，范围 [0, 1]。
    """
    clamped_progress = float(max(0.0, min(1.0, progress)))
    start = float(max(0.0, min(1.0, learned_start)))
    end = float(max(start, min(1.0, learned_end)))
    if end <= start:
        return 1.0 if clamped_progress >= end else 0.0
    alpha = (clamped_progress - start) / max(end - start, 1e-8)
    alpha = float(max(0.0, min(1.0, alpha)))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def compute_residue_proposal_priors(
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    knn: int = 16,
) -> torch.Tensor:
    """
    计算残基提议先验。

    基于残基局部密度、暴露度和深度估计启发式先验特征，
    为中心提议模块提供额外几何线索。

    Args:
        residue_pos: 残基坐标张量。
        residue_batch: 残基所属 batch 索引。
        knn: 计算局部先验时使用的邻居数。

    Returns:
        Tensor: 每个残基对应的启发式先验特征张量。
    """
    priors = residue_pos.new_zeros((residue_pos.size(0), 4))
    if residue_pos.numel() == 0:
        return priors

    num_graphs = int(residue_batch.max().item()) + 1 if residue_batch.numel() > 0 else 0
    for graph_idx in range(num_graphs):
        mask = residue_batch == graph_idx
        pos = residue_pos[mask]
        if pos.size(0) == 0:
            continue
        if pos.size(0) == 1:
            priors[mask] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0],
                device=residue_pos.device,
                dtype=residue_pos.dtype,
            )
            continue

        dist = torch.cdist(pos, pos)
        dist.fill_diagonal_(float("inf"))
        k = min(knn, max(1, pos.size(0) - 1))
        knn_dist = torch.topk(dist, k=k, largest=False, dim=-1).values
        mean_knn = knn_dist.mean(dim=-1)

        protein_center = pos.mean(dim=0, keepdim=True)
        radial = torch.norm(pos - protein_center, dim=-1)
        radial_norm = radial / radial.max().clamp_min(1e-6)
        depth = 1.0 - radial_norm

        density = torch.exp(-mean_knn / 4.0)
        exposure = torch.sigmoid(
            (mean_knn - mean_knn.mean())
            / mean_knn.std(unbiased=False).clamp_min(1e-6)
        )
        cavity = density * depth * (1.0 - exposure)
        priors[mask] = torch.stack([density, exposure, depth, cavity], dim=-1)

    return priors.clamp(0.0, 1.0)


def resolve_ehfnet_model(model: torch.nn.Module) -> EHFNet:
    """
    解析底层 EHFNet 模型。

    兼容 DataParallel 等包装场景，提取真正的 `EHFNet` 实例，
    供推理辅助函数统一调用模型特定接口。

    Args:
        model: 当前使用的模型实例。

    Returns:
        EHFNet: 去除包装器后的底层 EHFNet 模型实例。

    Raises:
        TypeError: 当传入模型不是 `EHFNet` 或其兼容包装器时抛出。
    """
    base_model = getattr(model, "module", model)
    if not isinstance(base_model, EHFNet):
        raise TypeError(f"Expected EHFNet-compatible model, got {type(base_model)!r}")
    return base_model


def predict_center_proposal_logits(
    model: torch.nn.Module,
    batch_obj: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预测中心提议 logits。

    调用模型的中心提议分支并整理残基位置、batch 和先验特征，
    为候选中心筛选阶段提供完整输入。

    Args:
        model: 当前使用的模型实例。
        batch_obj: 当前批次图对象。
        device: 运行所用设备，如 CPU 或 CUDA 设备。

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: 中心 logits、残基坐标、batch 索引和残基先验特征。
    """
    base_model = resolve_ehfnet_model(model)
    residue_store = batch_obj["protein_residue"]
    lig_store = batch_obj["ligand_molecule"]
    residue_batch = getattr(
        residue_store,
        "batch",
        torch.zeros(residue_store.pos.size(0), dtype=torch.long),
    )
    esm_missing_mask = getattr(residue_store, "esm_missing_mask", None)
    residue_prior_feat = compute_residue_proposal_priors(
        residue_store.pos.to(device),
        residue_batch.to(device),
    )
    logits = base_model.predict_center_logits(
        residue_x_cat=residue_store.x_cat.to(device),
        residue_x_cont=residue_store.x_cont.to(device),
        residue_pos=residue_store.pos.to(device),
        residue_batch=residue_batch.to(device),
        lig_mol_x_cont=lig_store.x_cont.to(device),
        residue_esm_missing_mask=(
            esm_missing_mask.to(device) if esm_missing_mask is not None else None
        ),
        residue_prior_feat=residue_prior_feat,
    )
    return (
        logits,
        residue_store.pos.to(device),
        residue_batch.to(device),
        residue_prior_feat,
    )


def select_diverse_center_indices(
    logits: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int,
    min_distance: float,
) -> torch.Tensor:
    """
    选择多样化中心索引。

    按打分排序后结合最小距离约束挑选中心，
    避免局部对接阶段的候选中心过度聚集。

    Args:
        logits: 模型输出的打分 logits。
        positions: 候选位置坐标集合。
        topk: 保留的前 K 个结果数量。
        min_distance: 中心去重时要求的最小距离。

    Returns:
        Tensor: 满足距离约束的候选中心索引。
    """
    order = torch.argsort(logits.view(-1), descending=True)
    selected: list[int] = []

    for idx in order.tolist():
        if len(selected) >= topk:
            break
        if not selected:
            selected.append(idx)
            continue
        pos = positions[idx]
        if all(torch.norm(pos - positions[j]).item() >= min_distance for j in selected):
            selected.append(idx)

    if len(selected) < min(topk, positions.size(0)):
        for idx in order.tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min(topk, positions.size(0)):
                break

    return torch.tensor(selected, dtype=torch.long, device=positions.device)


def combine_center_pose_score(
    center_logit: torch.Tensor,
    pose_logit: torch.Tensor,
    *,
    aff_logit: torch.Tensor | None = None,
    clash_value: torch.Tensor | None = None,
    fusion_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    融合中心与 pose 分数。

    将中心分支、排序分支以及可选亲和力和位阻信号合成为最终排序分数，
    用于候选 pose 的重排序。

    Args:
        center_logit: 中心分支输出的 logit。
        pose_logit: 构象排序分支输出的 logit。
        aff_logit: 亲和力分支输出的 logit。
        clash_value: 位阻分支给出的冲突值。
        fusion_weights: 融合不同分支分数时使用的权重字典。

    Returns:
        Tensor: 融合中心、构象及可选辅助信号后的最终排序分数。
    """
    fusion = dict(DEFAULT_FUSION_WEIGHTS)
    if fusion_weights is not None:
        fusion.update(fusion_weights)
    center_score = torch.sigmoid(center_logit.view(-1))
    pose_score = torch.sigmoid(pose_logit.view(-1))
    result = (
        fusion["pose_weight"] * pose_score
        + fusion["center_weight"] * center_score
        + fusion["bias"]
    )
    if aff_logit is not None and fusion.get("aff_weight", 0.0) != 0.0:
        aff_score = torch.sigmoid(aff_logit.view(-1))
        result = result + fusion["aff_weight"] * aff_score
    if clash_value is not None and fusion.get("clash_weight", 0.0) != 0.0:
        clash_penalty = torch.exp(-clash_value.view(-1) / 10.0)
        result = result + fusion["clash_weight"] * clash_penalty
    return result
