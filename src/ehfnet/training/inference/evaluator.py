"""
推理评估入口。

负责组织 blind Top-N 候选生成与指标汇总，
将最终测试评估与训练期轻量验证职责拆分开来。
"""


from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from ehfnet.graph import GraphCollator
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.inference.metrics import summarize_blind_candidate_records


def evaluate_topn_success(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    loader: DataLoader,
    device: torch.device,
    graph_builder: Any,
    collator: GraphCollator,
    topk_values: tuple[int, ...],
    center_topk: int,
    refine_topk: int,
    center_nms_radius: float,
    stage1_pose_samples: int,
    stage2_pose_samples: int,
    crop_radius: float,
    ode_steps: int,
    ode_method: str,
    warmup_epochs: int,
    crop_min_residues: int,
    crop_atom_margin: float,
    cost_guard_limit: int | None = None,
    num_gnn_blocks: int = 1,
    dynamic_inter_max_neighbors: int = 1,
    dynamic_residue_max_neighbors: int = 1,
    dynamic_residue_candidate_topk: int = 1,
    phase_multiplier: float = 1.0,
    max_oom_retry_splits: int = 0,
    center_hit_radius: float,
    fusion_weights: dict[str, float] | None = None,
    return_candidate_records: bool = False,
    progress_desc: str = "TopN Eval",
    dataset_raw_dir: str | None = None,
) -> dict[str, Any]:
    """
    评估 Top-N 对接成功率。

    基于统一候选生成流程统计 Top-N 成功率与相关指标，
    用于衡量完整两阶段盲对接流水线的效果。

    Args:
        model: 当前使用的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        loader: 提供批次数据的 DataLoader。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        topk_values: 需要统计的 Top-K 指标列表。
        center_topk: 中心提议阶段保留的候选中心数量。
        refine_topk: 局部重排序阶段保留的候选构象数量。
        center_nms_radius: 中心去重时使用的最小间距半径。
        stage1_pose_samples: 第一阶段局部对接生成的候选构象数。
        stage2_pose_samples: 第二阶段精排生成的候选构象数。
        crop_radius: 局部裁剪半径。
        ode_steps: ODE 推理积分步数。
        ode_method: blind Top-N 评估使用的 ODE 积分方法。
        warmup_epochs: 课程学习预热轮数。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        cost_guard_limit: 候选生成阶段的成本保护上限。
        num_gnn_blocks: 主干 GNN 块数量，用于估计运行时成本。
        dynamic_inter_max_neighbors: 动态原子跨图边的单源邻居上限。
        dynamic_residue_max_neighbors: 动态配体-残基边的单源邻居上限。
        dynamic_residue_candidate_topk: 动态配体-残基边每个复合物保留的候选残基数。
        phase_multiplier: 当前候选生成阶段的成本倍率。
        max_oom_retry_splits: 单个候选生成 batch 允许递归拆分重试的最大深度。
        center_hit_radius: 判断中心命中的距离阈值。
        fusion_weights: 融合不同分支分数时使用的权重字典。
        return_candidate_records: 是否同时返回候选记录明细。
        progress_desc: 终端中显示的 Top-N 评估阶段名称。
        dataset_raw_dir: 数据集原始样本目录，用于计算对称感知 RMSD。

    Returns:
        dict[str, Any]: Top-N 指标字典，必要时附带候选记录明细。

    Raises:
        RuntimeError: 当候选生成流程发生不可恢复错误时抛出。
    """
    topk_unique = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    if not topk_unique:
        raise ValueError("topk_values must contain at least one positive integer")

    generation_result = generate_candidates_from_loader(
        model=model,
        matcher=matcher,
        loader=loader,
        device=device,
        graph_builder=graph_builder,
        collator=collator,
        center_topk=center_topk,
        refine_topk=refine_topk,
        center_nms_radius=center_nms_radius,
        stage1_pose_samples=stage1_pose_samples,
        stage2_pose_samples=stage2_pose_samples,
        crop_radius=crop_radius,
        ode_steps=ode_steps,
        ode_method=ode_method,
        warmup_epochs=warmup_epochs,
        center_hit_radius=center_hit_radius,
        crop_min_residues=crop_min_residues,
        crop_atom_margin=crop_atom_margin,
        fusion_weights=fusion_weights,
        cost_guard_limit=cost_guard_limit,
        num_gnn_blocks=num_gnn_blocks,
        dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
        dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
        dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
        phase_multiplier=phase_multiplier,
        max_oom_retry_splits=max_oom_retry_splits,
        progress_desc=progress_desc,
        dataset_raw_dir=dataset_raw_dir,
    )
    candidate_records = cast(
        list[dict[str, Any]],
        generation_result.get("candidate_records", []),
    )

    total_graphs = len(candidate_records)
    if total_graphs == 0:
        result: dict[str, Any] = {"topn_total_graphs": 0.0}
        if return_candidate_records:
            result["candidate_records"] = []
        return result

    metrics: dict[str, Any] = {
        "topn_total_graphs": float(total_graphs),
        "topn_cost_guard_skips": float(generation_result.get("cost_guard_skips", 0.0)),
        "topn_oom_batches": float(generation_result.get("oom_batches", 0.0)),
        "topn_failed_batches": float(generation_result.get("failed_batches", 0.0)),
    }
    metrics.update(
        summarize_blind_candidate_records(
            candidate_records,
            topk_values=topk_unique,
            fusion_weights=fusion_weights,
        )
    )
    if return_candidate_records:
        metrics["candidate_records"] = candidate_records
    return metrics
