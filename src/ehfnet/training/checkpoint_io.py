"""
Checkpoint 工具。

负责组装、筛选和序列化 checkpoint 内容，
并维护训练期间的模型保存逻辑。
"""


import math
from typing import Any

from ehfnet.contracts import build_feature_signature, build_model_config


def safe_metric(value: Any, default: float, *, higher_is_better: bool = True) -> float:
    """
    安全读取指标值。

    在指标缺失、解析失败或数值非法时返回兜底值，
    避免 checkpoint 选择逻辑被异常指标打断。

    Args:
        value: 待处理或校验的输入值。
        default: 异常情况下返回的默认值。
        higher_is_better: 指标是否满足值越大越好的比较方向。

    Returns:
        float: 返回可安全参与 checkpoint 选择比较的指标值。
    """
    try:
        metric = float(value)
    except Exception:
        return default

    if math.isnan(metric) or math.isinf(metric):
        return default

    return metric


def build_selection_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """
    构建 checkpoint 选择指标。

    从原始验证指标中整理出用于模型选择的关键分数，
    供保存最佳 checkpoint 时统一比较。

    Args:
        metrics: 原始或汇总后的指标字典。

    Returns:
        dict[str, float]: 用于最佳 checkpoint 选择的指标字典。
    """
    success_2a = safe_metric(metrics.get("reranked_top1_success_2a", metrics.get("success_2a")), 0.0)
    success_5a = safe_metric(metrics.get("reranked_top5_success_2a", metrics.get("success_5a")), 0.0)
    oracle_top5_success_2a = safe_metric(metrics.get("oracle_top5_success_2a"), 0.0)
    mean_rmsd = safe_metric(
        metrics.get("reranked_top1_mean_best_rmsd", metrics.get("mean_rmsd_final")),
        1e9,
        higher_is_better=False,
    )
    val_loss = safe_metric(
        metrics.get("val_loss", metrics.get("reranked_top1_mean_best_rmsd")),
        1e9,
        higher_is_better=False,
    )
    center_recall = safe_metric(metrics.get("center_recall@8"), 0.0)
    proposal_gap = safe_metric(metrics.get("proposal_gap"), 100.0, higher_is_better=False)
    ranking_gap = safe_metric(metrics.get("ranking_gap"), 100.0, higher_is_better=False)
    composite_score = (
        1.75 * success_2a
        + 0.35 * success_5a
        + 0.20 * oracle_top5_success_2a
        + 0.10 * center_recall
        - 0.75 * mean_rmsd
        - 0.15 * proposal_gap
        - 0.20 * ranking_gap
    )
    blind_combo_score = 1.0 * success_2a + 0.35 * oracle_top5_success_2a - 0.10 * ranking_gap

    return {
        "composite_score": composite_score,
        "blind_combo_score": blind_combo_score,
        "success_2a": success_2a,
        "success_5a": success_5a,
        "oracle_top5_success_2a": oracle_top5_success_2a,
        "mean_rmsd": mean_rmsd,
        "val_loss": val_loss,
        "center_recall@8": center_recall,
        "proposal_gap": proposal_gap,
        "ranking_gap": ranking_gap,
    }


CHECKPOINT_SELECTION_MODES: dict[str, tuple[str, bool, str]] = {
    "composite": ("composite_score", True, "Composite"),
    "reranked_top1_success_2a": ("success_2a", True, "Rerank@1<2A"),
    "reranked_top5_success_2a": ("success_5a", True, "Rerank@5<2A"),
    "reranked_top1_plus_oracle_top5": ("blind_combo_score", True, "Rerank@1 + Oracle@5"),
}


def resolve_selection_rule(
    checkpoint_selection_mode: str,
) -> tuple[str, bool, str]:
    """
    解析 checkpoint 选择规则。

    根据配置返回主比较指标、比较方向和展示标签，
    统一训练期间的最佳模型判定逻辑。

    Args:
        checkpoint_selection_mode: checkpoint 选择规则名称。

    Returns:
        tuple[str, bool, str]: 主指标名、比较方向和展示标签。

    Raises:
        ValueError: 当选择规则名称不受支持时抛出。
    """
    if checkpoint_selection_mode not in CHECKPOINT_SELECTION_MODES:
        raise ValueError(
            "checkpoint_selection_mode must be one of "
            f"{tuple(CHECKPOINT_SELECTION_MODES.keys())}, got {checkpoint_selection_mode!r}"
        )
    return CHECKPOINT_SELECTION_MODES[checkpoint_selection_mode]


def is_better_checkpoint(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    *,
    primary_key: str,
    primary_higher_is_better: bool,
    tol: float = 1e-6,
) -> bool:
    """
    比较两个 checkpoint 候选。

    根据选择规则判断新候选是否优于当前最优结果，
    用于驱动最佳模型的更新。

    Args:
        candidate: 待比较的新 checkpoint 指标。
        incumbent: 当前最优 checkpoint 指标。
        primary_key: 主比较指标名称。
        primary_higher_is_better: 主指标是否满足越大越好。
        tol: 比较两个指标时允许的数值容差。

    Returns:
        bool: 返回布尔判断结果。
    """
    if incumbent is None:
        return True

    candidate_primary = candidate[primary_key]
    incumbent_primary = incumbent[primary_key]

    if primary_higher_is_better:
        if candidate_primary > incumbent_primary + tol:
            return True
        if candidate_primary < incumbent_primary - tol:
            return False
    else:
        if candidate_primary < incumbent_primary - tol:
            return True
        if candidate_primary > incumbent_primary + tol:
            return False

    if candidate["success_2a"] > incumbent["success_2a"] + tol:
        return True
    if candidate["success_2a"] < incumbent["success_2a"] - tol:
        return False

    if candidate["success_5a"] > incumbent["success_5a"] + tol:
        return True
    if candidate["success_5a"] < incumbent["success_5a"] - tol:
        return False

    if candidate["mean_rmsd"] < incumbent["mean_rmsd"] - tol:
        return True
    if candidate["mean_rmsd"] > incumbent["mean_rmsd"] + tol:
        return False

    if candidate["val_loss"] < incumbent["val_loss"] - tol:
        return True
    if candidate["val_loss"] > incumbent["val_loss"] + tol:
        return False

    return False


def compose_checkpoint(
    *,
    epoch_idx: int,
    avg_train_loss_value: float,
    val_metrics_obj: dict[str, Any],
    selection_metrics: dict[str, float],
    model: Any,
    ema_model: Any | None,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    best_val_loss: float,
    best_rmsd: float,
    current_fusion_weights: dict[str, float],
    normalization_stats: dict,
    run_name: str | None,
    run_log_file: str | None,
    esm_dim: int,
    interaction_profile: str,
    hidden_dim: int,
    num_gnn_blocks: int,
    m_dim_scalar: int,
    dropout_rate: float,
    lig_atom_cont_count: int,
    lig_mol_cont_count: int,
    pro_atom_cont_count: int,
    pro_res_cont_count: int,
    num_rbf: int,
    r_cutoff: float,
    force_cutoff: float,
    frame_refine_threshold: float,
    frame_refine_temperature: float,
    energy_guide_threshold: float,
    energy_guide_temperature: float,
    clash_threshold: float,
    clash_push_threshold: float,
    clash_push_force: float,
    score_clamp_min: float,
    score_clamp_max: float,
    force_limit: float,
    prediction_max_neighbors: int,
    prediction_min_max_neighbors: int,
    prediction_knn_fallback_k: int,
    r_cutoff_intra: float,
    max_neighbors_intra: int,
    atom_neighbor_cap: int,
    residue_neighbor_cap: int,
    residue_radius_scale: float,
    residue_radius_bias: float,
    ligand_atom_fallback_k: int,
    protein_atom_fallback_k: int,
    protein_residue_fallback_k: int,
    dynamic_inter_cutoff: float,
    dynamic_inter_knn_k: int,
    dynamic_residue_cutoff: float,
    dynamic_residue_knn_k: int,
) -> dict[str, Any]:
    """
    组装 checkpoint 内容。

    将模型参数、优化器状态、训练配置和关键指标整理为保存字典，
    供训练中断恢复与评估复用。

    Args:
        epoch_idx: 待写入 checkpoint 的当前轮次索引。
        avg_train_loss_value: 当前轮次的平均训练损失。
        val_metrics_obj: 当前轮次的验证指标对象。
        selection_metrics: 用于模型选择的关键指标字典。
        model: 当前使用的模型实例。
        ema_model: 指数滑动平均得到的模型副本。
        criterion: 训练或验证阶段使用的损失函数对象。
        optimizer: 优化器实例。
        scheduler: 学习率调度器实例。
        best_val_loss: 历史最佳验证损失。
        best_rmsd: 历史最佳 RMSD 指标。
        current_fusion_weights: 当前使用的分数融合权重。
        normalization_stats: 输入特征归一化统计量。
        run_name: 本次运行的名称标识。
        run_log_file: 本次运行对应的日志文件路径。
        esm_dim: ESM 残基嵌入维度。
        interaction_profile: 跨图交互拓扑配置。
        hidden_dim: 隐藏层维度。
        num_gnn_blocks: 主干 GNN 块数量。
        m_dim_scalar: 消息传递分支的标量维度。
        dropout_rate: Dropout 比例。
        lig_atom_cont_count: 配体原子连续特征维度。
        lig_mol_cont_count: 配体分子连续特征维度。
        pro_atom_cont_count: 蛋白原子连续特征维度。
        pro_res_cont_count: 蛋白残基连续特征维度。
        num_rbf: RBF 基函数数量。
        r_cutoff: 几何邻域构建的距离截断半径。
        force_cutoff: 力相关分支使用的局部截断半径。
        frame_refine_threshold: 主惯量帧细化门控阈值。
        frame_refine_temperature: 主惯量帧细化门控温度。
        energy_guide_threshold: 能量引导门控阈值。
        energy_guide_temperature: 能量引导门控温度。
        clash_threshold: 位阻判定阈值。
        clash_push_threshold: 位阻推开分支使用的距离阈值。
        clash_push_force: 位阻推开分支的力缩放系数。
        score_clamp_min: 分数裁剪下界。
        score_clamp_max: 分数裁剪上界。
        force_limit: 力大小的软限制。
        prediction_max_neighbors: 预测头阶段允许保留的最大邻居数。
        prediction_min_max_neighbors: 预测头动态邻居上限的最小值。
        prediction_knn_fallback_k: 预测头回退到 kNN 时使用的邻居数。
        r_cutoff_intra: 图内边构建的距离截断半径。
        max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
        atom_neighbor_cap: 原子层图内边的邻居上限。
        residue_neighbor_cap: 残基层图内边的邻居上限。
        residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
        residue_radius_bias: 残基层邻域半径的额外偏置。
        ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
        protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
        protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
        dynamic_inter_cutoff: 动态跨图原子边的半径阈值。
        dynamic_inter_knn_k: 动态跨图原子边回退到 kNN 时的邻居数。
        dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
        dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。

    Returns:
        dict[str, Any]: 可直接保存到磁盘的完整 checkpoint 字典。
    """
    model_config = build_model_config(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        esm_dim=esm_dim,
        interaction_profile=interaction_profile,
        m_dim_scalar=m_dim_scalar,
        dropout_rate=dropout_rate,
        num_rbf=num_rbf,
        r_cutoff=r_cutoff,
        force_cutoff=force_cutoff,
        frame_refine_threshold=frame_refine_threshold,
        frame_refine_temperature=frame_refine_temperature,
        energy_guide_threshold=energy_guide_threshold,
        energy_guide_temperature=energy_guide_temperature,
        clash_threshold=clash_threshold,
        clash_push_threshold=clash_push_threshold,
        clash_push_force=clash_push_force,
        score_clamp_min=score_clamp_min,
        score_clamp_max=score_clamp_max,
        force_limit=force_limit,
        max_neighbors=prediction_max_neighbors,
        min_max_neighbors=prediction_min_max_neighbors,
        knn_fallback_k=prediction_knn_fallback_k,
        r_cutoff_intra=r_cutoff_intra,
        max_neighbors_intra=max_neighbors_intra,
        atom_neighbor_cap=atom_neighbor_cap,
        residue_neighbor_cap=residue_neighbor_cap,
        residue_radius_scale=residue_radius_scale,
        residue_radius_bias=residue_radius_bias,
        ligand_atom_fallback_k=ligand_atom_fallback_k,
        protein_atom_fallback_k=protein_atom_fallback_k,
        protein_residue_fallback_k=protein_residue_fallback_k,
        dynamic_inter_cutoff=dynamic_inter_cutoff,
        dynamic_inter_knn_k=dynamic_inter_knn_k,
        dynamic_residue_cutoff=dynamic_residue_cutoff,
        dynamic_residue_knn_k=dynamic_residue_knn_k,
    )
    return {
        "epoch": epoch_idx,
        "run_name": run_name,
        "run_log_file": run_log_file,
        "model_config": model_config,
        "feature_signature": build_feature_signature(esm_dim=esm_dim),
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.module.state_dict() if ema_model is not None else model.state_dict(),
        "loss_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_rmsd": best_rmsd,
        "avg_train_loss": avg_train_loss_value,
        "val_metrics": dict(val_metrics_obj),
        "selection_metrics": dict(selection_metrics),
        "fusion_weights": dict(current_fusion_weights),
        "normalization_stats": normalization_stats,
    }
