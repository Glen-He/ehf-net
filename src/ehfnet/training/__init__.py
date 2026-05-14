"""
训练模块入口。

导出训练主流程和公开训练工具，
统一训练侧能力的访问路径。
"""


from importlib import import_module

__all__ = [
    "BlindCandidateReplayDataset",
    "ConditionalFlowMatcher",
    "FlowMatchingLoss",
    "apply_loss_context",
    "build_center_value_targets",
    "build_local_batch_from_centers",
    "build_selection_metrics",
    "compose_checkpoint",
    "compute_bootstrap_pose_rank_loss",
    "compute_center_value_loss",
    "compute_pose_rank_target",
    "compute_rerank_losses",
    "compute_train_split_normalization_stats",
    "compute_validation_loss",
    "evaluate_topn_success",
    "generate_blind_candidates",
    "generate_candidates_from_loader",
    "get_pool_stats",
    "is_better_checkpoint",
    "load_blind_pool",
    "pairwise_ranking_loss_from_pairs",
    "refresh_blind_candidate_pool",
    "replay_and_compute_losses",
    "resolve_selection_rule",
    "save_blind_pool",
    "select_bootstrap_blind_centers",
    "select_pose_rank_logit",
    "select_training_crop_centers",
    "select_wrong_center_candidates",
    "should_refresh_pool",
    "should_run_bootstrap",
    "train",
]

_EXPORT_MAP = {
    "BlindCandidateReplayDataset": (
        "ehfnet.training.blind_pool",
        "BlindCandidateReplayDataset",
    ),
    "ConditionalFlowMatcher": ("ehfnet.training.flow_matcher", "ConditionalFlowMatcher"),
    "FlowMatchingLoss": ("ehfnet.training.losses", "FlowMatchingLoss"),
    "apply_loss_context": ("ehfnet.training.batch_helpers", "apply_loss_context"),
    "build_center_value_targets": (
        "ehfnet.training.rerank_losses",
        "build_center_value_targets",
    ),
    "build_local_batch_from_centers": (
        "ehfnet.training.batch_helpers",
        "build_local_batch_from_centers",
    ),
    "build_selection_metrics": (
        "ehfnet.training.checkpoint_io",
        "build_selection_metrics",
    ),
    "compose_checkpoint": ("ehfnet.training.checkpoint_io", "compose_checkpoint"),
    "compute_bootstrap_pose_rank_loss": (
        "ehfnet.training.center_sampling",
        "compute_bootstrap_pose_rank_loss",
    ),
    "compute_center_value_loss": (
        "ehfnet.training.rerank_losses",
        "compute_center_value_loss",
    ),
    "compute_pose_rank_target": (
        "ehfnet.training.batch_helpers",
        "compute_pose_rank_target",
    ),
    "compute_rerank_losses": ("ehfnet.training.rerank_losses", "compute_rerank_losses"),
    "compute_train_split_normalization_stats": (
        "ehfnet.training.normalization",
        "compute_train_split_normalization_stats",
    ),
    "compute_validation_loss": (
        "ehfnet.training.validation",
        "compute_validation_loss",
    ),
    "evaluate_topn_success": (
        "ehfnet.training.inference.evaluator",
        "evaluate_topn_success",
    ),
    "generate_blind_candidates": (
        "ehfnet.training.candidate_generation",
        "generate_blind_candidates",
    ),
    "generate_candidates_from_loader": (
        "ehfnet.training.candidate_generation",
        "generate_candidates_from_loader",
    ),
    "get_pool_stats": ("ehfnet.training.blind_pool", "get_pool_stats"),
    "is_better_checkpoint": ("ehfnet.training.checkpoint_io", "is_better_checkpoint"),
    "load_blind_pool": ("ehfnet.training.blind_pool", "load_blind_pool"),
    "pairwise_ranking_loss_from_pairs": (
        "ehfnet.training.rerank_losses",
        "pairwise_ranking_loss_from_pairs",
    ),
    "refresh_blind_candidate_pool": (
        "ehfnet.training.blind_pool",
        "refresh_blind_candidate_pool",
    ),
    "replay_and_compute_losses": (
        "ehfnet.training.blind_pool",
        "replay_and_compute_losses",
    ),
    "resolve_selection_rule": (
        "ehfnet.training.checkpoint_io",
        "resolve_selection_rule",
    ),
    "save_blind_pool": ("ehfnet.training.blind_pool", "save_blind_pool"),
    "select_bootstrap_blind_centers": (
        "ehfnet.training.center_sampling",
        "select_bootstrap_blind_centers",
    ),
    "select_pose_rank_logit": (
        "ehfnet.training.batch_helpers",
        "select_pose_rank_logit",
    ),
    "select_training_crop_centers": (
        "ehfnet.training.center_sampling",
        "select_training_crop_centers",
    ),
    "select_wrong_center_candidates": (
        "ehfnet.training.center_sampling",
        "select_wrong_center_candidates",
    ),
    "should_refresh_pool": ("ehfnet.training.blind_pool", "should_refresh_pool"),
    "should_run_bootstrap": ("ehfnet.training.center_sampling", "should_run_bootstrap"),
    "train": ("ehfnet.training.trainer", "train"),
}


def __getattr__(name: str):
    """
    按名称返回公开对象。

    仅在首次访问时执行真实导入，
    用于避免包初始化阶段触发重模块加载或循环依赖。

    Args:
        name: 请求访问或解析的公开对象名称。

    Returns:
        object: 返回与名称对应的惰性导出对象。

    Raises:
        AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
    """

    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
