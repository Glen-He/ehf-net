"""
推理评估工具。

负责汇总盲对接候选记录、计算排序指标并校准融合权重，
服务候选排序与推理分析流程。
"""


from typing import Any

import torch

from ehfnet.training.inference.center_utils import (
    DEFAULT_FUSION_WEIGHTS,
    combine_center_pose_score,
)


def summarize_blind_candidate_records(
    candidate_records: list[dict[str, Any]],
    *,
    topk_values: tuple[int, ...],
    fusion_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    汇总 blind 候选指标。

    根据候选中心和 pose 记录计算 Top-N、proposal gap 和 rerank 指标，
    用于评估完整盲对接流水线的效果。

    Args:
        candidate_records: 候选中心与构象的记录列表。
        topk_values: 需要统计的 Top-K 指标列表。
        fusion_weights: 融合不同分支分数时使用的权重字典。

    Returns:
        dict[str, float]: 汇总后的 blind pipeline 指标字典。
    """
    if not candidate_records:
        return {"topn_total_graphs": 0.0}

    topk_unique = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    metrics: dict[str, float] = {
        "topn_total_graphs": float(len(candidate_records)),
    }
    center_recall_hits = {1: 0.0, 3: 0.0, 8: 0.0, 16: 0.0}
    oracle_top1_rmsd: list[float] = []
    oracle_top5_rmsd: list[float] = []
    reranked_top1_rmsd: list[float] = []
    reranked_top5_rmsd: list[float] = []
    reranked_topk_rmsd: dict[int, list[float]] = {k: [] for k in topk_unique}
    proposal_failures = 0.0
    local_failures = 0.0
    ranking_failures = 0.0
    successes = 0.0

    for record in candidate_records:
        centers = list(record.get("centers", []))
        poses = list(record.get("poses", []))
        if not poses:
            proposal_failures += 1.0
            continue

        center_hits = [bool(center.get("is_center_hit_4A", False)) for center in centers]
        for k in center_recall_hits:
            if any(center_hits[: min(k, len(center_hits))]):
                center_recall_hits[k] += 1.0

        oracle_all = min(float(item["rmsd"]) for item in poses)
        oracle_first5_pool = [
            float(item["rmsd"])
            for item in poses
            if int(item.get("center_id", 999)) <= 5
        ]
        if not oracle_first5_pool:
            oracle_first5_pool = [float(item["rmsd"]) for item in poses]
        oracle_top1_rmsd.append(oracle_all)
        oracle_top5_rmsd.append(min(oracle_first5_pool))

        reranked = sorted(
            poses,
            key=lambda item: float(
                combine_center_pose_score(
                    torch.tensor([item["center_logit"]], dtype=torch.float32),
                    torch.tensor([item["ranking_logit"]], dtype=torch.float32),
                    aff_logit=torch.tensor(
                        [item.get("binding_affinity_teacher", 0.0)],
                        dtype=torch.float32,
                    ),
                    clash_value=torch.tensor(
                        [item.get("steric_clash_teacher", 0.0)],
                        dtype=torch.float32,
                    ),
                    fusion_weights=fusion_weights,
                )[0].item()
            ),
            reverse=True,
        )
        reranked_top1 = float(reranked[0]["rmsd"])
        reranked_top5 = min(
            float(item["rmsd"]) for item in reranked[: min(5, len(reranked))]
        )
        reranked_top1_rmsd.append(reranked_top1)
        reranked_top5_rmsd.append(reranked_top5)
        for k in topk_unique:
            reranked_topk_rmsd[k].append(
                min(float(item["rmsd"]) for item in reranked[: min(k, len(reranked))])
            )

        has_center_hit = any(center_hits)
        if not has_center_hit:
            proposal_failures += 1.0
        elif oracle_all >= 2.0:
            local_failures += 1.0
        elif reranked_top1 >= 2.0:
            ranking_failures += 1.0
        else:
            successes += 1.0

    total = max(1.0, float(len(candidate_records)))
    for k, hit_count in center_recall_hits.items():
        metrics[f"center_recall@{k}"] = (hit_count / total) * 100.0

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    def _success(values: list[float], threshold: float) -> float:
        return 100.0 * float(sum(v < threshold for v in values)) / max(1, len(values))

    metrics["oracle_top1_success_2a"] = _success(oracle_top1_rmsd, 2.0)
    metrics["oracle_top1_success_5a"] = _success(oracle_top1_rmsd, 5.0)
    metrics["oracle_top5_success_2a"] = _success(oracle_top5_rmsd, 2.0)
    metrics["oracle_top5_success_5a"] = _success(oracle_top5_rmsd, 5.0)
    metrics["oracle_top1_mean_best_rmsd"] = _mean(oracle_top1_rmsd)
    metrics["oracle_top5_mean_best_rmsd"] = _mean(oracle_top5_rmsd)
    metrics["reranked_top1_success_2a"] = _success(reranked_top1_rmsd, 2.0)
    metrics["reranked_top1_success_5a"] = _success(reranked_top1_rmsd, 5.0)
    metrics["reranked_top5_success_2a"] = _success(reranked_top5_rmsd, 2.0)
    metrics["reranked_top5_success_5a"] = _success(reranked_top5_rmsd, 5.0)
    metrics["reranked_top1_mean_best_rmsd"] = _mean(reranked_top1_rmsd)
    metrics["reranked_top5_mean_best_rmsd"] = _mean(reranked_top5_rmsd)
    metrics["proposal_gap"] = (proposal_failures / total) * 100.0
    metrics["local_gap"] = (local_failures / total) * 100.0
    metrics["ranking_gap"] = (ranking_failures / total) * 100.0
    metrics["pipeline_success"] = (successes / total) * 100.0

    for k in topk_unique:
        source = reranked_topk_rmsd[k]
        metrics[f"top{k}_success_2a"] = _success(source, 2.0)
        metrics[f"top{k}_success_5a"] = _success(source, 5.0)
        metrics[f"top{k}_mean_best_rmsd"] = _mean(source)

    return metrics


def calibrate_linear_fusion_weights(
    candidate_records: list[dict[str, Any]],
    *,
    topk_values: tuple[int, ...],
    search_center_weights: tuple[float, ...],
    search_aff_weights: tuple[float, ...],
    search_clash_weights: tuple[float, ...],
) -> dict[str, float]:
    """
    搜索融合权重。

    在候选记录上遍历中心、亲和力和位阻权重组合，
    找到当前评估指标下更优的线性融合参数。

    Args:
        candidate_records: 候选中心与构象的记录列表。
        topk_values: 需要统计的 Top-K 指标列表。
        search_center_weights: 待搜索的中心分支融合权重集合。
        search_aff_weights: 待搜索的亲和力融合权重集合。
        search_clash_weights: 待搜索的位阻融合权重集合。

    Returns:
        dict[str, float]: 当前候选记录下表现最优的线性融合权重。
    """
    best_weights = dict(DEFAULT_FUSION_WEIGHTS)
    best_metrics = summarize_blind_candidate_records(
        candidate_records,
        topk_values=topk_values,
        fusion_weights=best_weights,
    )

    def _is_better(trial: dict[str, float], ref: dict[str, float]) -> bool:
        t1 = trial.get("reranked_top1_success_2a", 0.0)
        r1 = ref.get("reranked_top1_success_2a", 0.0)
        if t1 > r1 + 1e-6:
            return True
        if t1 < r1 - 1e-6:
            return False
        t5 = trial.get("reranked_top5_success_2a", 0.0)
        r5 = ref.get("reranked_top5_success_2a", 0.0)
        if t5 > r5 + 1e-6:
            return True
        if t5 < r5 - 1e-6:
            return False
        return trial.get("reranked_top1_mean_best_rmsd", float("inf")) < ref.get(
            "reranked_top1_mean_best_rmsd",
            float("inf"),
        ) - 1e-6

    for cw in search_center_weights:
        for aw in search_aff_weights:
            for clw in search_clash_weights:
                trial_weights = {
                    "pose_weight": 1.0,
                    "center_weight": float(cw),
                    "aff_weight": float(aw),
                    "clash_weight": float(clw),
                    "bias": 0.0,
                }
                trial_metrics = summarize_blind_candidate_records(
                    candidate_records,
                    topk_values=topk_values,
                    fusion_weights=trial_weights,
                )
                if _is_better(trial_metrics, best_metrics):
                    best_weights = trial_weights
                    best_metrics = trial_metrics

    return best_weights
