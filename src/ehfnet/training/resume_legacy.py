"""
Legacy resume 兼容工具。

负责承接旧版 checkpoint 的兼容恢复与新旧训练状态桥接，
便于后续在不改动主恢复流程的前提下整体删除。
"""


import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.optim.swa_utils import AveragedModel

from ehfnet.contracts.checkpoint import validate_checkpoint_contract
from ehfnet.training.adaptive_batching import WindowAimdBudgetController
from ehfnet.training.checkpoint_io import restore_rng_state
from ehfnet.training.resume import (
    _restore_metric_snapshot,
    ResumeLoadResult,
    TrainerRuntimeState,
    validate_resume_training_config,
)

logger = logging.getLogger(__name__)

LEGACY_TRAINING_CONFIG_KEY_ALIASES: dict[str, str] = {
    "loss_weight_trans": "loss_weight_translation",
    "loss_weight_rot": "loss_weight_rotation",
    "loss_coarse_trans": "loss_coarse_translation",
    "loss_coarse_rot": "loss_coarse_rotation",
    "loss_transition_trans": "loss_transition_translation",
    "loss_transition_rot": "loss_transition_rotation",
    "loss_refine_trans": "loss_refine_translation",
    "loss_refine_rot": "loss_refine_rotation",
}


@dataclass(frozen=True)
class ResumeCompatibilityAssessment:
    """
    Legacy resume 兼容性评估结果。

    Attributes:
        checkpoint_schema_version: checkpoint schema 版本。
        checkpoint_role: checkpoint 角色标记。
        missing_components: 当前 checkpoint 中缺失、因此无法严格恢复的状态组件。
    """

    checkpoint_schema_version: int
    checkpoint_role: str
    missing_components: tuple[str, ...]


def is_legacy_resume_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    """
    判断 checkpoint 是否需要走 legacy 兼容恢复。

    Args:
        checkpoint: 已加载的 checkpoint 字典。

    Returns:
        bool: 若缺少严格恢复所需的关键状态则返回 `True`。
    """
    return not (
        isinstance(checkpoint.get("training_config"), Mapping)
        and isinstance(checkpoint.get("trainer_state"), Mapping)
        and isinstance(checkpoint.get("rng_state"), Mapping)
    )


def uses_legacy_training_config_keys(
    training_config: Mapping[str, Any],
) -> bool:
    """
    判断训练配置快照是否仍使用旧版缩写键名。

    Args:
        training_config: checkpoint 中保存的训练配置快照。

    Returns:
        bool: 若存在旧版缩写键名则返回 `True`。
    """
    return any(
        legacy_key in training_config
        for legacy_key in LEGACY_TRAINING_CONFIG_KEY_ALIASES
    )


def normalize_legacy_training_config_keys(
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将旧版缩写训练配置键名规范化为当前全称键名。

    Args:
        training_config: checkpoint 中保存的训练配置快照。

    Returns:
        dict[str, Any]: 键名已统一后的训练配置字典。
    """
    return {
        LEGACY_TRAINING_CONFIG_KEY_ALIASES.get(str(key), str(key)): value
        for key, value in training_config.items()
    }


def _assess_legacy_resume_compatibility(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_schema_version: int,
    checkpoint_role: str,
    training_config_snapshot: Mapping[str, Any],
) -> ResumeCompatibilityAssessment:
    """
    评估 legacy checkpoint 的缺失状态与兼容范围。

    Args:
        checkpoint: 原始 checkpoint 字典。
        checkpoint_schema_version: checkpoint schema 版本。
        checkpoint_role: checkpoint 角色标记。
        training_config_snapshot: 当前运行的训练配置快照。

    Returns:
        ResumeCompatibilityAssessment: 缺失状态说明。

    Raises:
        ValueError: 当 legacy checkpoint 中仍保存了训练配置且与当前运行不兼容时抛出。
    """
    saved_training_config = checkpoint.get("training_config")
    if isinstance(saved_training_config, Mapping):
        validate_resume_training_config(
            saved_config=normalize_legacy_training_config_keys(saved_training_config),
            current_config=training_config_snapshot,
        )

    missing_components: list[str] = []
    if not isinstance(saved_training_config, Mapping):
        missing_components.append("training_config")
    if not isinstance(checkpoint.get("trainer_state"), Mapping):
        missing_components.append("trainer_state")
    if not isinstance(checkpoint.get("rng_state"), Mapping):
        missing_components.append("rng_state")

    return ResumeCompatibilityAssessment(
        checkpoint_schema_version=checkpoint_schema_version,
        checkpoint_role=checkpoint_role,
        missing_components=tuple(missing_components),
    )


def _log_legacy_resume_compatibility(
    assessment: ResumeCompatibilityAssessment,
) -> None:
    """
    输出 legacy checkpoint 的兼容恢复诊断信息。

    Args:
        assessment: legacy 兼容性评估结果。
    """
    logger.warning(
        "Resume compatibility | mode=best_effort | schema=%d | role=%s | missing=%s. "
        "Optimizer/model states will still be restored, but missing components will fall back to runtime defaults.",
        assessment.checkpoint_schema_version,
        assessment.checkpoint_role,
        ", ".join(assessment.missing_components),
    )


def _restore_runtime_state_from_trainer_state(
    *,
    trainer_state: Mapping[str, Any],
    runtime_state: TrainerRuntimeState,
    current_fusion_weights: dict[str, float],
) -> TrainerRuntimeState:
    """
    从完整 trainer_state 恢复训练期运行状态。

    Args:
        trainer_state: checkpoint 中保存的 trainer_state。
        runtime_state: 当前运行的默认训练状态。
        current_fusion_weights: 当前已恢复的融合权重。

    Returns:
        TrainerRuntimeState: 恢复后的训练期状态。
    """
    return TrainerRuntimeState(
        best_val_loss=runtime_state.best_val_loss,
        best_rmsd=runtime_state.best_rmsd,
        best_composite_metrics=_restore_metric_snapshot(
            trainer_state.get("best_composite_metrics")
        ),
        best_single_shot_success2a_metrics=_restore_metric_snapshot(
            trainer_state.get("best_single_shot_success2a_metrics")
        ),
        best_rmsd_metrics=_restore_metric_snapshot(
            trainer_state.get("best_rmsd_metrics")
        ),
        best_selected_metrics=_restore_metric_snapshot(
            trainer_state.get("best_selected_metrics")
        ),
        current_fusion_weights=current_fusion_weights,
        total_oom_batches=int(
            trainer_state.get(
                "total_oom_batches",
                runtime_state.total_oom_batches,
            )
        ),
        runtime_benchmark_history=list(
            trainer_state.get(
                "runtime_benchmark_history",
                runtime_state.runtime_benchmark_history,
            )
        ),
        clone_safety_checked=bool(
            trainer_state.get(
                "clone_safety_checked",
                runtime_state.clone_safety_checked,
            )
        ),
    )


def _restore_runtime_state_from_selection_metrics(
    *,
    selection_metrics: Mapping[str, Any] | None,
    runtime_state: TrainerRuntimeState,
    current_fusion_weights: dict[str, float],
) -> TrainerRuntimeState:
    """
    从旧版 selection_metrics 做兼容恢复。

    Args:
        selection_metrics: checkpoint 中保存的选优指标。
        runtime_state: 当前运行的默认训练状态。
        current_fusion_weights: 当前已恢复的融合权重。

    Returns:
        TrainerRuntimeState: best-effort 恢复后的训练期状态。
    """
    if not isinstance(selection_metrics, Mapping):
        logger.warning(
            "Legacy resume checkpoint does not include selection_metrics; "
            "best-metric snapshots will be reset to runtime defaults."
        )
        return TrainerRuntimeState(
            best_val_loss=runtime_state.best_val_loss,
            best_rmsd=runtime_state.best_rmsd,
            best_composite_metrics=runtime_state.best_composite_metrics,
            best_single_shot_success2a_metrics=runtime_state.best_single_shot_success2a_metrics,
            best_rmsd_metrics=runtime_state.best_rmsd_metrics,
            best_selected_metrics=runtime_state.best_selected_metrics,
            current_fusion_weights=current_fusion_weights,
            total_oom_batches=runtime_state.total_oom_batches,
            runtime_benchmark_history=list(runtime_state.runtime_benchmark_history),
            clone_safety_checked=runtime_state.clone_safety_checked,
        )

    normalized_selection_metrics = {
        key: float(value)
        for key, value in selection_metrics.items()
    }
    return TrainerRuntimeState(
        best_val_loss=runtime_state.best_val_loss,
        best_rmsd=runtime_state.best_rmsd,
        best_composite_metrics=dict(normalized_selection_metrics),
        best_single_shot_success2a_metrics=dict(normalized_selection_metrics),
        best_rmsd_metrics=dict(normalized_selection_metrics),
        best_selected_metrics=dict(normalized_selection_metrics),
        current_fusion_weights=current_fusion_weights,
        total_oom_batches=runtime_state.total_oom_batches,
        runtime_benchmark_history=list(runtime_state.runtime_benchmark_history),
        clone_safety_checked=runtime_state.clone_safety_checked,
    )


def _resolve_resume_blind_pool_source_dir(
    *,
    resume_path: Path,
    resume_blind_pool_dir: str | None,
) -> str | None:
    """
    解析 resume 使用的 blind pool 源目录。

    Args:
        resume_path: 当前 resume checkpoint 路径。
        resume_blind_pool_dir: 显式指定的 blind pool 目录。

    Returns:
        str | None: 可用的 blind pool 源目录。
    """
    if resume_blind_pool_dir is not None:
        return str(Path(resume_blind_pool_dir).expanduser())
    inferred_blind_pool_dir = resume_path.resolve().parent / "blind_pool_cache"
    if inferred_blind_pool_dir.is_dir():
        return str(inferred_blind_pool_dir)
    return None


def load_legacy_resume_checkpoint(
    *,
    resume_ckpt: str,
    checkpoint: Mapping[str, Any],
    device: torch.device,
    use_configured_cuda: bool,
    training_config_snapshot: Mapping[str, Any],
    resume_blind_pool_dir: str | None,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    ema_decay: float,
    runtime_state: TrainerRuntimeState,
    train_budget_controller: WindowAimdBudgetController,
    val_partial_budget_controller: WindowAimdBudgetController,
    val_full_budget_controller: WindowAimdBudgetController,
    same_center_budget_controller: WindowAimdBudgetController,
    ranking_budget_controller: WindowAimdBudgetController,
    replay_budget_controller: WindowAimdBudgetController,
) -> ResumeLoadResult:
    """
    从 legacy checkpoint 做兼容恢复。

    Args:
        resume_ckpt: resume checkpoint 路径。
        checkpoint: 已加载的 checkpoint 字典。
        device: 当前训练设备。
        use_configured_cuda: 当前运行是否启用 CUDA。
        training_config_snapshot: 当前运行的训练配置快照。
        resume_blind_pool_dir: 显式指定的 blind pool 恢复目录。
        model: 当前模型实例。
        criterion: 当前损失函数对象。
        optimizer: 当前优化器实例。
        scheduler: 当前学习率调度器实例。
        ema_decay: EMA 衰减系数。
        runtime_state: 当前初始化后的训练状态基线。
        train_budget_controller: 训练预算控制器。
        val_partial_budget_controller: partial 验证预算控制器。
        val_full_budget_controller: full 验证预算控制器。
        same_center_budget_controller: same-center 排序预算控制器。
        ranking_budget_controller: wrong-center 排序预算控制器。
        replay_budget_controller: replay 预算控制器。

    Returns:
        ResumeLoadResult: 恢复后的 resume 结果对象。
    """
    resume_path = Path(resume_ckpt).expanduser()
    checkpoint_schema_version = int(checkpoint.get("checkpoint_schema_version", 1))
    checkpoint_model_config = validate_checkpoint_contract(checkpoint)
    checkpoint_role = str(checkpoint.get("checkpoint_role", "legacy"))
    compatibility = _assess_legacy_resume_compatibility(
        checkpoint=checkpoint,
        checkpoint_schema_version=checkpoint_schema_version,
        checkpoint_role=checkpoint_role,
        training_config_snapshot=training_config_snapshot,
    )
    _log_legacy_resume_compatibility(compatibility)

    model.load_state_dict(checkpoint["model_state_dict"])

    loss_state_dict = checkpoint.get("loss_state_dict")
    if isinstance(loss_state_dict, dict):
        criterion.load_state_dict(loss_state_dict)

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state_dict, dict):
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if isinstance(scheduler_state_dict, dict):
        scheduler.load_state_dict(scheduler_state_dict)

    ema_model: AveragedModel | None = None
    ema_state_dict = checkpoint.get("ema_model_state_dict")
    if isinstance(ema_state_dict, dict):
        ema_model = AveragedModel(
            model,
            avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
        )
        ema_model.module.load_state_dict(ema_state_dict)

    best_val_loss = float(checkpoint.get("best_val_loss", runtime_state.best_val_loss))
    best_rmsd = float(checkpoint.get("best_rmsd", runtime_state.best_rmsd))
    current_fusion_weights = dict(runtime_state.current_fusion_weights)
    fusion_weights = checkpoint.get("fusion_weights")
    if isinstance(fusion_weights, dict):
        current_fusion_weights = {
            key: float(value)
            for key, value in fusion_weights.items()
        }

    strict_runtime_state = TrainerRuntimeState(
        best_val_loss=best_val_loss,
        best_rmsd=best_rmsd,
        best_composite_metrics=runtime_state.best_composite_metrics,
        best_single_shot_success2a_metrics=runtime_state.best_single_shot_success2a_metrics,
        best_rmsd_metrics=runtime_state.best_rmsd_metrics,
        best_selected_metrics=runtime_state.best_selected_metrics,
        current_fusion_weights=current_fusion_weights,
        total_oom_batches=runtime_state.total_oom_batches,
        runtime_benchmark_history=list(runtime_state.runtime_benchmark_history),
        clone_safety_checked=runtime_state.clone_safety_checked,
    )

    trainer_state = checkpoint.get("trainer_state")
    if isinstance(trainer_state, Mapping):
        restored_runtime_state = _restore_runtime_state_from_trainer_state(
            trainer_state=trainer_state,
            runtime_state=strict_runtime_state,
            current_fusion_weights=current_fusion_weights,
        )
        controller_states = trainer_state.get("budget_controllers")
        if isinstance(controller_states, Mapping):
            controller_map = {
                "train": train_budget_controller,
                "val_partial": val_partial_budget_controller,
                "val_full": val_full_budget_controller,
                "same_center": same_center_budget_controller,
                "ranking": ranking_budget_controller,
                "replay": replay_budget_controller,
            }
            for controller_name, controller in controller_map.items():
                controller_state = controller_states.get(controller_name)
                if controller_state is not None:
                    controller.load_state_dict(
                        cast(Mapping[str, Any], controller_state)
                    )
    else:
        restored_runtime_state = _restore_runtime_state_from_selection_metrics(
            selection_metrics=cast(
                Mapping[str, Any] | None,
                checkpoint.get("selection_metrics"),
            ),
            runtime_state=strict_runtime_state,
            current_fusion_weights=current_fusion_weights,
        )

    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, Mapping):
        restore_rng_state(
            cast(Mapping[str, Any] | None, rng_state),
            restore_cuda=use_configured_cuda,
        )
    else:
        logger.warning(
            "Resume checkpoint does not include rng_state; random generators will continue from the current process state."
        )

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = max(0, resumed_epoch + 1)
    resume_blind_pool_source_dir = _resolve_resume_blind_pool_source_dir(
        resume_path=resume_path,
        resume_blind_pool_dir=resume_blind_pool_dir,
    )

    logger.info(
        "Resuming training from %s | next_epoch=%d | checkpoint_epoch=%d | role=%s | hidden_dim=%d | blocks=%d.",
        resume_path,
        start_epoch + 1,
        resumed_epoch + 1,
        checkpoint_role,
        int(checkpoint_model_config["hidden_dim"]),
        int(checkpoint_model_config["num_gnn_blocks"]),
    )

    return ResumeLoadResult(
        start_epoch=start_epoch,
        checkpoint_role=checkpoint_role,
        resume_blind_pool_source_dir=resume_blind_pool_source_dir,
        ema_model=ema_model,
        runtime_state=restored_runtime_state,
    )
