"""
Resume 工具。

负责严格校验 resume 配置、恢复训练期状态，
并将 checkpoint 恢复逻辑从训练主循环中拆分出来。
"""


import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.optim.swa_utils import AveragedModel

from ehfnet.contracts.checkpoint import validate_checkpoint_contract
from ehfnet.training.adaptive_batching import WindowAimdBudgetController
from ehfnet.training.checkpoint_io import restore_rng_state

logger = logging.getLogger(__name__)

RESUME_TRAINING_CONFIG_GROUPS: dict[str, tuple[str, ...]] = {
    "data": (
        "data_root",
        "index_file",
        "esm",
        "split_train_frac",
        "split_val_frac",
        "split_test_frac",
        "split_seed",
        "split_cache_file",
        "force_resplit",
        "ablation_mode",
    ),
    "training": (
        "epochs",
        "lr",
        "weight_decay",
        "clip_grad",
        "crop_radius",
        "warmup_epochs",
        "val_subset_ratio",
        "val_full_every",
        "val_full_last_epochs",
        "ode_method",
        "accumulation_steps",
        "ema_decay",
        "run_test_after_training",
        "test_topk_values",
        "checkpoint_selection_mode",
    ),
    "batching": (
        "train_cost_budget",
        "val_cost_budget",
        "blind_pool_cost_budget",
        "final_topn_cost_budget",
        "eval_cost_guard_headroom",
        "max_oom_retry_splits",
        "enable_train_budget_callback",
        "oom_reduce_threshold",
        "oom_reduce_factor",
        "min_train_cost_budget",
        "enable_val_budget_callback",
        "val_oom_reduce_threshold",
        "val_oom_reduce_factor",
        "min_val_cost_budget",
        "train_budget_window_size",
        "train_budget_recover_window_count",
        "train_budget_recover_step",
        "train_offender_cooldown",
        "val_budget_window_size",
        "val_budget_recover_window_count",
        "val_budget_recover_step",
        "val_offender_cooldown",
    ),
    "proposal": (
        "center_proposal_weight",
        "center_positive_radius",
        "center_guidance_learned_start",
        "center_proposal_topk",
        "center_refine_topk",
        "center_nms_radius",
        "stage1_pose_samples",
        "stage2_pose_samples",
        "crop_candidate_topk",
        "crop_proposal_start",
        "crop_near_miss_start",
        "crop_hard_negative_start",
        "crop_min_residues",
        "crop_atom_margin",
        "disable_jitter_crop",
        "disable_hard_negative_crop",
    ),
    "ranking": (
        "pose_ranking_pair_weight",
        "pose_ranking_margin",
        "ranking_same_center_start",
        "ranking_wrong_center_start",
        "same_center_micro_batch_size",
        "same_center_budget_window_size",
        "same_center_budget_recover_window_count",
        "same_center_budget_recover_step",
        "same_center_offender_cooldown",
        "ranking_budget_window_size",
        "ranking_budget_recover_window_count",
        "ranking_offender_cooldown",
        "ranking_wrong_center_cap",
    ),
    "bootstrap": (
        "pose_bootstrap_weight",
        "pose_bootstrap_start",
        "pose_bootstrap_frequency",
        "pose_bootstrap_ode_steps",
    ),
    "replay": (
        "blind_pool_refresh_every",
        "blind_pool_start_epoch",
        "blind_pool_refresh_on_best_update",
        "blind_pool_max_complexes",
        "blind_pool_cache_bce_weight",
        "blind_pool_cache_rank_weight",
        "blind_pool_pairs_per_complex",
        "replay_start_ratio",
        "replay_micro_batch_size",
        "replay_budget_window_size",
        "replay_budget_recover_window_count",
        "replay_candidate_cooldown",
        "replay_max_candidates_per_complex",
    ),
    "blind_inference": ("val_ode_steps",),
    "flow": (
        "flow_sigma_min",
        "flow_spatial_sigma_min",
        "flow_spatial_sigma_max",
        "flow_fd_dt",
        "flow_rotation_angle_min",
        "flow_rotation_angle_max",
        "flow_torsion_scale_min",
        "flow_torsion_scale_max",
    ),
    "loss": (
        "loss_characteristic_scale",
        "loss_weight_translation",
        "loss_weight_rotation",
        "loss_weight_torsion",
        "loss_weight_energy",
        "loss_weight_clash",
        "loss_weight_pose_rank",
        "loss_coarse_translation",
        "loss_coarse_rotation",
        "loss_coarse_torsion",
        "loss_coarse_energy",
        "loss_coarse_clash",
        "loss_coarse_pose_rank",
        "loss_transition_translation",
        "loss_transition_rotation",
        "loss_transition_torsion",
        "loss_transition_energy",
        "loss_transition_clash",
        "loss_transition_pose_rank",
        "loss_refine_translation",
        "loss_refine_rotation",
        "loss_refine_torsion",
        "loss_refine_energy",
        "loss_refine_clash",
        "loss_refine_pose_rank",
        "loss_refine_start",
        "loss_pose_gate_epoch_start",
        "loss_pose_gate_epoch_end",
        "loss_pose_gate_tau_start",
        "loss_pose_gate_tau_end",
        "loss_pose_gate_temperature",
    ),
}

RESUME_ALLOWED_OVERRIDE_KEYS = {
    "val_subset_ratio",
    "val_full_every",
    "val_full_last_epochs",
    "run_test_after_training",
    "blind_pool_refresh_every",
    "blind_pool_refresh_on_best_update",
    "blind_pool_max_complexes",
    "blind_pool_cost_budget",
    "final_topn_cost_budget",
    "eval_cost_guard_headroom",
    "test_topk_values",
}


def _iter_resume_training_keys() -> tuple[str, ...]:
    """
    按分组顺序展开 resume 配置键。

    Returns:
        tuple[str, ...]: 供配置快照构建与比对使用的扁平键序列。
    """
    resume_training_keys = tuple(
        key
        for grouped_keys in RESUME_TRAINING_CONFIG_GROUPS.values()
        for key in grouped_keys
    )
    if len(resume_training_keys) != len(set(resume_training_keys)):
        raise ValueError(
            "RESUME_TRAINING_CONFIG_GROUPS contains duplicated config keys."
        )
    return resume_training_keys


@dataclass(frozen=True)
class TrainerRuntimeState:
    """
    训练期可恢复运行状态。

    Attributes:
        best_val_loss: 历史最佳验证损失，越小越好。
        best_rmsd: 历史最佳 mean RMSD，越小越好。
        best_composite_metrics: 历史最佳 composite 指标快照。
        best_single_shot_success2a_metrics: 历史最佳 Success@2A 指标快照。
        best_rmsd_metrics: 历史最佳 RMSD 指标快照。
        best_selected_metrics: 主 checkpoint 规则对应的历史最佳指标快照。
        current_fusion_weights: 当前 top-n 重排使用的融合权重。
        total_oom_batches: 训练至今累计的根 OOM 事件数。
        runtime_benchmark_history: 每轮训练记录的运行时 benchmark 历史。
        clone_safety_checked: 是否已经完成 clone 安全检查。
    """

    best_val_loss: float
    best_rmsd: float
    best_composite_metrics: dict[str, float] | None
    best_single_shot_success2a_metrics: dict[str, float] | None
    best_rmsd_metrics: dict[str, float] | None
    best_selected_metrics: dict[str, float] | None
    current_fusion_weights: dict[str, float]
    total_oom_batches: int
    runtime_benchmark_history: list[dict[str, Any]]
    clone_safety_checked: bool


@dataclass(frozen=True)
class ResumeLoadResult:
    """
    Resume 恢复结果。

    Attributes:
        start_epoch: 恢复后下一轮训练的 0-based epoch 索引。
        checkpoint_role: 当前 checkpoint 的角色标记。
        resume_blind_pool_source_dir: 可用于恢复 blind pool 的源目录。
        ema_model: 恢复后的 EMA 模型；若 checkpoint 中不存在则为 `None`。
        runtime_state: 恢复后的训练期运行状态。
    """

    start_epoch: int
    checkpoint_role: str
    resume_blind_pool_source_dir: str | None
    ema_model: AveragedModel | None
    runtime_state: TrainerRuntimeState


def _normalize_resume_value(value: Any) -> Any:
    """
    将 resume 配置值规范化为可比较对象。

    Args:
        value: 待规范化的原始配置值。

    Returns:
        Any: 适合做 resume 配置比对的规范化结果。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_normalize_resume_value(item) for item in value)
    if isinstance(value, list):
        return [_normalize_resume_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_resume_value(item)
            for key, item in value.items()
        }
    return value


def _resume_values_equal(left: Any, right: Any) -> bool:
    """
    比较两个 resume 配置值是否等价。

    Args:
        left: 左侧配置值。
        right: 右侧配置值。

    Returns:
        bool: 若两者在 resume 语义上等价则返回 `True`。
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _resume_values_equal(a, b)
            for a, b in zip(left, right)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _resume_values_equal(a, b)
            for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(
            _resume_values_equal(left[key], right[key])
            for key in left
        )
    return left == right


def _restore_metric_snapshot(value: Any) -> dict[str, float] | None:
    """
    从 checkpoint 恢复最佳指标快照。

    Args:
        value: checkpoint 中保存的指标对象。

    Returns:
        dict[str, float] | None: 恢复后的指标字典。
    """
    if not isinstance(value, Mapping):
        return None
    return {str(key): float(metric) for key, metric in value.items()}


def _serialize_metric_snapshot(
    metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    """
    规范化保存最佳指标快照。

    Args:
        metrics: 待写入 checkpoint 的指标字典。

    Returns:
        dict[str, float] | None: 规范化后的指标字典。
    """
    if metrics is None:
        return None
    return {str(key): float(value) for key, value in metrics.items()}


def build_training_config_snapshot(source_values: Mapping[str, Any]) -> dict[str, Any]:
    """
    从训练入口参数构建 resume 配置快照。

    Args:
        source_values: 包含训练输入参数的映射对象。

    Returns:
        dict[str, Any]: 可写入 checkpoint 的训练配置快照。
    """
    resume_training_keys = _iter_resume_training_keys()
    return {
        key: _normalize_resume_value(source_values[key])
        for key in resume_training_keys
    }


def validate_resume_training_config(
    *,
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> None:
    """
    校验 resume 时当前配置是否与 checkpoint 兼容。

    Args:
        saved_config: checkpoint 中保存的训练配置快照。
        current_config: 当前运行解析得到的训练配置快照。

    Raises:
        ValueError: 当存在不允许的 resume 配置漂移时抛出。
    """
    mismatch_lines: list[str] = []
    allowed_override_lines: list[str] = []
    for key, saved_value in saved_config.items():
        if key not in current_config:
            mismatch_lines.append(
                f"{key}: checkpoint={saved_value!r}, current=<missing>"
            )
            continue
        current_value = current_config[key]
        if _resume_values_equal(saved_value, current_value):
            continue
        if key in RESUME_ALLOWED_OVERRIDE_KEYS:
            allowed_override_lines.append(
                f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
            )
            continue
        mismatch_lines.append(
            f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
        )
    if allowed_override_lines:
        logger.info(
            "Resume allowed overrides detected:\n%s",
            "\n".join(f"  - {line}" for line in allowed_override_lines),
        )
    if mismatch_lines:
        raise ValueError(
            "Resume checkpoint is incompatible with the current training configuration.\n"
            + "\n".join(f"  - {line}" for line in mismatch_lines)
        )


def build_trainer_state_snapshot(
    *,
    runtime_state: TrainerRuntimeState,
    train_budget_controller: WindowAimdBudgetController,
    val_partial_budget_controller: WindowAimdBudgetController,
    val_full_budget_controller: WindowAimdBudgetController,
    same_center_budget_controller: WindowAimdBudgetController,
    ranking_budget_controller: WindowAimdBudgetController,
    replay_budget_controller: WindowAimdBudgetController,
) -> dict[str, Any]:
    """
    组装训练主循环的可恢复状态。

    Args:
        runtime_state: 当前训练期运行状态。
        train_budget_controller: 训练预算控制器。
        val_partial_budget_controller: partial 验证预算控制器。
        val_full_budget_controller: full 验证预算控制器。
        same_center_budget_controller: same-center 排序预算控制器。
        ranking_budget_controller: wrong-center 排序预算控制器。
        replay_budget_controller: replay 预算控制器。

    Returns:
        dict[str, Any]: 可直接写入 checkpoint 的 trainer 状态字典。
    """
    return {
        "total_oom_batches": int(runtime_state.total_oom_batches),
        "runtime_benchmark_history": list(runtime_state.runtime_benchmark_history),
        "clone_safety_checked": bool(runtime_state.clone_safety_checked),
        "best_composite_metrics": _serialize_metric_snapshot(
            runtime_state.best_composite_metrics
        ),
        "best_single_shot_success2a_metrics": _serialize_metric_snapshot(
            runtime_state.best_single_shot_success2a_metrics
        ),
        "best_rmsd_metrics": _serialize_metric_snapshot(runtime_state.best_rmsd_metrics),
        "best_selected_metrics": _serialize_metric_snapshot(
            runtime_state.best_selected_metrics
        ),
        "budget_controllers": {
            "train": train_budget_controller.state_dict(),
            "val_partial": val_partial_budget_controller.state_dict(),
            "val_full": val_full_budget_controller.state_dict(),
            "same_center": same_center_budget_controller.state_dict(),
            "ranking": ranking_budget_controller.state_dict(),
            "replay": replay_budget_controller.state_dict(),
        },
    }
def load_resume_checkpoint(
    *,
    resume_ckpt: str,
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
    加载并恢复 resume checkpoint。

    Args:
        resume_ckpt: resume checkpoint 路径。
        device: 当前训练设备。
        use_configured_cuda: 当前运行是否启用 CUDA。
        training_config_snapshot: 当前运行解析出的训练配置快照。
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

    Raises:
        FileNotFoundError: 当 resume checkpoint 不存在时抛出。
        ValueError: 当 checkpoint 与当前训练配置不兼容时抛出。
    """
    resume_path = Path(resume_ckpt).expanduser()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(
        str(resume_path),
        map_location=device,
        weights_only=False,
    )
    checkpoint_schema_version = int(checkpoint.get("checkpoint_schema_version", 1))
    checkpoint_model_config = validate_checkpoint_contract(checkpoint)
    checkpoint_role = str(checkpoint.get("checkpoint_role", "legacy"))
    resume_capable = bool(checkpoint.get("resume_capable", False))
    epoch_boundary_complete = bool(checkpoint.get("epoch_boundary_complete", False))

    saved_training_config = checkpoint.get("training_config")
    has_strict_resume_components = (
        isinstance(saved_training_config, Mapping)
        and isinstance(checkpoint.get("trainer_state"), Mapping)
        and isinstance(checkpoint.get("rng_state"), Mapping)
    )
    needs_legacy_training_config_bridge = False
    if isinstance(saved_training_config, Mapping):
        from ehfnet.training.resume_legacy import uses_legacy_training_config_keys

        needs_legacy_training_config_bridge = uses_legacy_training_config_keys(
            saved_training_config
        )

    if not has_strict_resume_components or needs_legacy_training_config_bridge:
        from ehfnet.training.resume_legacy import load_legacy_resume_checkpoint

        return load_legacy_resume_checkpoint(
            resume_ckpt=resume_ckpt,
            checkpoint=checkpoint,
            device=device,
            use_configured_cuda=use_configured_cuda,
            training_config_snapshot=training_config_snapshot,
            resume_blind_pool_dir=resume_blind_pool_dir,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_decay=ema_decay,
            runtime_state=runtime_state,
            train_budget_controller=train_budget_controller,
            val_partial_budget_controller=val_partial_budget_controller,
            val_full_budget_controller=val_full_budget_controller,
            same_center_budget_controller=same_center_budget_controller,
            ranking_budget_controller=ranking_budget_controller,
            replay_budget_controller=replay_budget_controller,
        )

    if checkpoint_schema_version >= 3 and not resume_capable:
        raise ValueError(
            "The requested checkpoint is not marked as resume-capable. "
            "Use latest_model.pt or model_epoch_*.pt instead of best_* selection checkpoints."
        )
    if checkpoint_schema_version >= 3 and not epoch_boundary_complete:
        raise ValueError(
            "The requested checkpoint does not contain a full epoch-boundary state. "
            "Use latest_model.pt or model_epoch_*.pt for resume."
        )

    validate_resume_training_config(
        saved_config=cast(Mapping[str, Any], saved_training_config),
        current_config=training_config_snapshot,
    )
    logger.info(
        "Resume compatibility | mode=strict | schema=%d | role=%s.",
        checkpoint_schema_version,
        checkpoint_role,
    )

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

    trainer_state = cast(Mapping[str, Any], checkpoint["trainer_state"])
    restored_runtime_state = TrainerRuntimeState(
        best_val_loss=best_val_loss,
        best_rmsd=best_rmsd,
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

    restore_rng_state(
        cast(Mapping[str, Any], checkpoint["rng_state"]),
        restore_cuda=use_configured_cuda,
    )

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = max(0, resumed_epoch + 1)
    if resume_blind_pool_dir is not None:
        resume_blind_pool_source_dir = str(Path(resume_blind_pool_dir).expanduser())
    else:
        inferred_blind_pool_dir = resume_path.resolve().parent / "blind_pool_cache"
        resume_blind_pool_source_dir = (
            str(inferred_blind_pool_dir)
            if inferred_blind_pool_dir.is_dir()
            else None
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
