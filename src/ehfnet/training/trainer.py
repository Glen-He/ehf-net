"""
训练主循环。

负责组织数据加载、优化更新、验证评估、候选池训练与 checkpoint 管理，
是训练阶段调度各模块的核心入口。
"""


import gc
import json
import logging
import math
import os
import time
import traceback
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Subset
from torch_scatter import scatter_mean
from tqdm import tqdm

from ehfnet.data import ProteinLigandDataset
from ehfnet.data.datasets import ScaffoldSplitter
from ehfnet.data.preprocess import LigandGeometryPreFilter
from ehfnet.data.featurizers import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.contracts import build_blind_pool_signature
from ehfnet.graph import GraphCollator, estimate_graph_cost_units
from ehfnet.runtime import build_dataset, build_model, resolve_interaction_profile
from ehfnet.training.batch_helpers import (
    apply_loss_context,
    build_local_batch_from_centers,
    compute_pose_rank_target,
    select_pose_rank_logit,
)
from ehfnet.training.adaptive_batching import (
    AdaptiveCostBatchSampler,
    estimate_runtime_batch_cost,
    extract_root_dataset_indices,
    resolve_subset_root_indices,
    split_collated_batch,
    WindowAimdBudgetController,
)
from ehfnet.training.blind_pool import (
    BlindCandidateReplayDataset,
    get_pool_stats,
    load_blind_pool,
    refresh_blind_candidate_pool,
    replay_and_compute_losses,
    save_blind_pool,
    should_refresh_pool,
)
from ehfnet.training.candidate_generation import generate_blind_candidates
from ehfnet.training.center_sampling import (
    compute_bootstrap_pose_rank_loss,
    select_bootstrap_blind_centers,
    select_training_crop_centers,
    select_wrong_center_candidates,
    should_run_bootstrap,
)
from ehfnet.training.checkpoint_io import (
    build_selection_metrics,
    build_checkpoint_model_config_kwargs,
    capture_rng_state,
    CheckpointEpochState,
    CheckpointModelState,
    compose_resume_checkpoint,
    compose_selection_checkpoint,
    is_better_checkpoint,
    resolve_selection_rule,
)
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.inference import (
    DEFAULT_FUSION_WEIGHTS,
    compute_center_guidance_fraction,
    compute_center_guidance_scores,
    evaluate_topn_success,
    predict_center_proposal_logits,
    select_diverse_center_indices,
    summarize_blind_candidate_records,
)
from ehfnet.training.losses import FlowMatchingLoss
from ehfnet.training.normalization import compute_train_split_normalization_stats
from ehfnet.training.resume import (
    build_trainer_state_snapshot,
    build_training_config_snapshot,
    load_resume_checkpoint,
    TrainerRuntimeState,
)
from ehfnet.training.rerank_losses import (
    build_center_value_targets,
    compute_center_value_loss,
    pairwise_ranking_loss_from_pairs,
)
from ehfnet.training.validation import compute_validation_loss

logger = logging.getLogger(__name__)


def train(
    *,
    data_root: str,
    index_file: str,
    save_dir: str,
    esm: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad: float,
    hidden_dim: int,
    num_gnn_blocks: int,
    m_dim_scalar: int,
    dropout_rate: float,
    lig_atom_cont_count: int,
    lig_mol_cont_count: int,
    pro_atom_cont_count: int,
    pro_res_cont_count: int,
    esm_dim: int,
    esm_model_name: str,
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
    dynamic_inter_max_neighbors: int,
    dynamic_residue_cutoff: float,
    dynamic_residue_knn_k: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
    flow_sigma_min: float,
    flow_spatial_sigma_min: float,
    flow_spatial_sigma_max: float,
    flow_fd_dt: float,
    flow_rotation_angle_min: float,
    flow_rotation_angle_max: float,
    flow_torsion_scale_min: float,
    flow_torsion_scale_max: float,
    loss_characteristic_scale: float,
    loss_weight_translation: float,
    loss_weight_rotation: float,
    loss_weight_torsion: float,
    loss_weight_energy: float,
    loss_weight_clash: float,
    loss_weight_pose_rank: float,
    loss_coarse_translation: float,
    loss_coarse_rotation: float,
    loss_coarse_torsion: float,
    loss_coarse_energy: float,
    loss_coarse_clash: float,
    loss_coarse_pose_rank: float,
    loss_transition_translation: float,
    loss_transition_rotation: float,
    loss_transition_torsion: float,
    loss_transition_energy: float,
    loss_transition_clash: float,
    loss_transition_pose_rank: float,
    loss_refine_translation: float,
    loss_refine_rotation: float,
    loss_refine_torsion: float,
    loss_refine_energy: float,
    loss_refine_clash: float,
    loss_refine_pose_rank: float,
    loss_refine_start: float,
    loss_pose_gate_epoch_start: float,
    loss_pose_gate_epoch_end: float,
    loss_pose_gate_tau_start: float,
    loss_pose_gate_tau_end: float,
    loss_pose_gate_temperature: float,
    device: str | torch.device,
    crop_radius: float,
    warmup_epochs: int,
    val_subset_ratio: float,
    val_full_every: int,
    val_full_last_epochs: int,
    min_checkpoint_selection_coverage: float,
    max_val_non_oom_failures: int,
    max_val_oom_failures: int,
    final_topn_min_coverage: float,
    ode_method: str,
    accumulation_steps: int,
    train_cost_budget: int,
    val_cost_budget: int,
    blind_pool_cost_budget: int,
    final_topn_cost_budget: int,
    eval_cost_guard_headroom: float,
    ema_decay: float,
    dataloader_num_workers: int,
    dataloader_pin_memory: bool,
    dataloader_persistent_workers: bool,
    max_oom_retry_splits: int,
    split_train_frac: float,
    split_val_frac: float,
    split_test_frac: float,
    split_seed: int,
    split_cache_file: str,
    force_resplit: bool,
    ablation_mode: str,
    run_test_after_training: bool,
    test_topk_values: tuple[int, ...],
    enable_train_budget_callback: bool,
    oom_reduce_threshold: int,
    oom_reduce_factor: float,
    min_train_cost_budget: int,
    enable_val_budget_callback: bool,
    val_oom_reduce_threshold: int,
    val_oom_reduce_factor: float,
    min_val_cost_budget: int,
    train_budget_window_size: int,
    train_budget_recover_window_count: int,
    train_budget_recover_step: int,
    train_offender_cooldown: int,
    val_budget_window_size: int,
    val_budget_recover_window_count: int,
    val_budget_recover_step: int,
    val_offender_cooldown: int,
    center_proposal_weight: float,
    center_positive_radius: float,
    center_guidance_learned_start: float,
    center_proposal_topk: int,
    center_refine_topk: int,
    center_nms_radius: float,
    stage1_pose_samples: int,
    stage2_pose_samples: int,
    crop_candidate_topk: int,
    crop_proposal_start: float,
    crop_near_miss_start: float,
    crop_hard_negative_start: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    disable_jitter_crop: bool,
    disable_hard_negative_crop: bool,
    pose_ranking_pair_weight: float,
    pose_ranking_margin: float,
    ranking_same_center_start: float,
    ranking_wrong_center_start: float,
    pose_bootstrap_weight: float,
    pose_bootstrap_start: float,
    pose_bootstrap_frequency: int,
    pose_bootstrap_ode_steps: int,
    val_ode_steps: int,
    checkpoint_selection_mode: str,
    blind_pool_refresh_every: int,
    blind_pool_start_epoch: int,
    blind_pool_refresh_on_best_update: bool,
    blind_pool_max_complexes: int,
    blind_pool_cache_bce_weight: float,
    blind_pool_cache_rank_weight: float,
    blind_pool_pairs_per_complex: int,
    replay_start_ratio: float,
    same_center_micro_batch_size: int,
    same_center_budget_window_size: int,
    same_center_budget_recover_window_count: int,
    same_center_budget_recover_step: int,
    same_center_offender_cooldown: int,
    ranking_budget_window_size: int,
    ranking_budget_recover_window_count: int,
    ranking_offender_cooldown: int,
    ranking_wrong_center_cap: int,
    replay_micro_batch_size: int,
    replay_budget_window_size: int,
    replay_budget_recover_window_count: int,
    replay_candidate_cooldown: int,
    replay_max_candidates_per_complex: int,
    geometry_min_atom_distance: float,
    resume_ckpt: str | None = None,
    resume_blind_pool_dir: str | None = None,
    stop_after_epoch: int | None = None,
    esm_path: str | None = None,
    run_name: str | None = None,
    run_log_file: str | None = None,
):
    """
    训练 EHFNet 模型

    Args:
        data_root: 数据集根目录。
        index_file: 数据索引文件路径。
        save_dir: 训练产物保存目录。
        esm: ESM 处理模式或缓存策略。
        epochs: 训练总轮数。
        lr: 优化器学习率。
        weight_decay: 权重衰减系数。
        clip_grad: 梯度裁剪阈值。
        hidden_dim: 隐藏层维度。
        num_gnn_blocks: 主干 GNN 块数量。
        m_dim_scalar: 消息传递分支的标量维度。
        dropout_rate: Dropout 比例。
        lig_atom_cont_count: 配体原子连续特征维度。
        lig_mol_cont_count: 配体分子连续特征维度。
        pro_atom_cont_count: 蛋白原子连续特征维度。
        pro_res_cont_count: 蛋白残基连续特征维度。
        esm_dim: ESM 残基嵌入维度。
        esm_model_name: ESM 主干模型名称。
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
        dynamic_inter_max_neighbors: 动态跨图原子边的单源邻居上限。
        dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
        dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。
        dynamic_residue_max_neighbors: 动态配体-残基边的单源邻居上限。
        dynamic_residue_candidate_topk: 动态配体-残基边每个复合物保留的候选残基数。
        flow_sigma_min: 流匹配时间噪声下界。
        flow_spatial_sigma_min: 平移扰动课程的最小尺度。
        flow_spatial_sigma_max: 平移扰动课程的最大尺度。
        flow_fd_dt: 流匹配目标构造时使用的有限差分步长。
        flow_rotation_angle_min: 课程初期允许的最大旋转角。
        flow_rotation_angle_max: 课程后期允许的最大旋转角。
        flow_torsion_scale_min: 课程初期的扭转扰动缩放系数。
        flow_torsion_scale_max: 课程后期的扭转扰动缩放系数。
        loss_characteristic_scale: 平衡平移与旋转量纲的特征长度尺度。
        loss_weight_translation: 平移损失的全局权重。
        loss_weight_rotation: 旋转损失的全局权重。
        loss_weight_torsion: 扭转损失的全局权重。
        loss_weight_energy: 亲和力损失的全局权重。
        loss_weight_clash: 位阻损失的全局权重。
        loss_weight_pose_rank: 构象排序损失的全局权重。
        loss_coarse_translation: 粗阶段平移损失权重。
        loss_coarse_rotation: 粗阶段旋转损失权重。
        loss_coarse_torsion: 粗阶段扭转损失权重。
        loss_coarse_energy: 粗阶段亲和力损失权重。
        loss_coarse_clash: 粗阶段位阻损失权重。
        loss_coarse_pose_rank: 粗阶段构象排序损失权重。
        loss_transition_translation: 过渡阶段平移损失权重。
        loss_transition_rotation: 过渡阶段旋转损失权重。
        loss_transition_torsion: 过渡阶段扭转损失权重。
        loss_transition_energy: 过渡阶段亲和力损失权重。
        loss_transition_clash: 过渡阶段位阻损失权重。
        loss_transition_pose_rank: 过渡阶段构象排序损失权重。
        loss_refine_translation: 细化阶段平移损失权重。
        loss_refine_rotation: 细化阶段旋转损失权重。
        loss_refine_torsion: 细化阶段扭转损失权重。
        loss_refine_energy: 细化阶段亲和力损失权重。
        loss_refine_clash: 细化阶段位阻损失权重。
        loss_refine_pose_rank: 细化阶段构象排序损失权重。
        loss_refine_start: 进入细化阶段时对应的训练进度阈值。
        loss_pose_gate_epoch_start: 构象相关损失开始打开门控的训练进度。
        loss_pose_gate_epoch_end: 构象相关损失完全打开门控的训练进度。
        loss_pose_gate_tau_start: 构象门控在初期使用的时间阈值。
        loss_pose_gate_tau_end: 构象门控在后期使用的时间阈值。
        loss_pose_gate_temperature: 构象时间门控的温度系数。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        crop_radius: 局部裁剪半径。
        warmup_epochs: 课程学习预热轮数。
        val_subset_ratio: 日常 partial 验证覆盖的验证集比例。
        val_full_every: 每隔多少个 epoch 执行一次 100% 轻量验证；0 表示关闭周期全量验证。
        val_full_last_epochs: 训练末尾连续执行 100% 轻量验证的 epoch 数。
        min_checkpoint_selection_coverage: Minimum validation coverage required for best checkpoint updates.
        max_val_non_oom_failures: Maximum non-OOM validation failures allowed before validation fails.
        max_val_oom_failures: Maximum irreducible validation OOM failures allowed before validation fails.
        final_topn_min_coverage: Minimum final Top-N coverage required for official test metrics.
        ode_method: 训练、bootstrap 与 blind 推理统一使用的 ODE 积分方法。
        accumulation_steps: 梯度累积步数。
        train_cost_budget: 训练阶段的基础成本预算。
        val_cost_budget: 验证阶段的基础成本预算。
        blind_pool_cost_budget: blind pool 刷新阶段的基础成本预算。
        final_topn_cost_budget: 最终 blind Top-N 评估阶段的基础成本预算。
        eval_cost_guard_headroom: 评估阶段为成本保护预留的额外裕量。
        ema_decay: EMA 模型更新的衰减系数。
        dataloader_num_workers: DataLoader 使用的 worker 数。
        dataloader_pin_memory: 是否为 DataLoader 启用 pin_memory。
        dataloader_persistent_workers: 是否为 DataLoader 启用持久 worker。
        max_oom_retry_splits: 单个 OOM batch 允许递归拆分重试的最大深度。
        split_train_frac: 训练集划分比例。
        split_val_frac: 验证集划分比例。
        split_test_frac: 测试集划分比例。
        split_seed: 数据划分使用的随机种子。
        split_cache_file: 数据划分缓存文件路径。
        force_resplit: 是否忽略已有划分缓存并重新划分数据集。
        ablation_mode: 当前训练使用的消融模式名称。
        run_test_after_training: 训练结束后是否自动执行测试评估。
        test_topk_values: 测试阶段统计 Top-N 成功率时使用的 N 列表。
        enable_train_budget_callback: 是否启用训练阶段的窗口式预算回调。
        oom_reduce_threshold: 单个训练窗口中触发自动降批所需的 OOM 根事件阈值。
        oom_reduce_factor: 触发 OOM 后缩小训练成本预算的比例系数。
        min_train_cost_budget: 训练阶段自动降批后的最小成本预算。
        enable_val_budget_callback: 是否启用验证阶段独立的窗口式预算回调。
        val_oom_reduce_threshold: 验证窗口中触发降批所需的 OOM 事件阈值。
        val_oom_reduce_factor: 验证阶段缩小成本预算的比例系数。
        min_val_cost_budget: 验证阶段自动降批后的最小成本预算。
        train_budget_window_size: 训练预算回调使用的根 batch 窗口大小。
        train_budget_recover_window_count: 训练预算回升所需的连续干净窗口数。
        train_budget_recover_step: 训练预算每次回升的加性步长。
        train_offender_cooldown: 训练坏样本冷却时长，以根 batch 事件计。
        val_budget_window_size: 验证预算回调使用的窗口大小。
        val_budget_recover_window_count: 验证预算回升所需的连续干净窗口数。
        val_budget_recover_step: 验证预算每次回升的加性步长。
        val_offender_cooldown: 验证坏样本冷却时长。
        center_proposal_weight: 中心提议分支在线监督与 replay 监督的共享权重。
        center_positive_radius: 中心判定为正样本时使用的距离半径。
        center_guidance_learned_start: 中心打分开始从几何先验平滑过渡到学习分数的进度阈值。
        center_proposal_topk: 中心提议阶段保留的 Top-K 数量。
        center_refine_topk: 中心细化阶段阶段保留的 Top-K 数量。
        center_nms_radius: 中心去重时使用的最小间距半径。
        stage1_pose_samples: 第一阶段局部对接生成的候选构象数。
        stage2_pose_samples: 第二阶段精排生成的候选构象数。
        crop_candidate_topk: crop候选阶段保留的 Top-K 数量。
        crop_proposal_start: `proposal_pos` 进入中心课程的进度阈值。
        crop_near_miss_start: `near_miss` 进入中心课程的进度阈值。
        crop_hard_negative_start: `hard_neg` 进入中心课程的进度阈值。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        disable_jitter_crop: 是否关闭jittercrop。
        disable_hard_negative_crop: 是否关闭hard负例crop。
        pose_ranking_pair_weight: 构象rankingpair相关的权重。
        pose_ranking_margin: 构象rankingmargin。
        ranking_same_center_start: same-center ranking 启用的进度阈值。
        ranking_wrong_center_start: wrong-center ranking 启用的进度阈值。
        pose_bootstrap_weight: 构象bootstrap相关的权重。
        pose_bootstrap_start: bootstrap 监督启用的进度阈值。
        pose_bootstrap_frequency: 构象bootstrapfrequency。
        pose_bootstrap_ode_steps: 构象bootstrapode的步数。
        val_ode_steps: valode的步数。
        checkpoint_selection_mode: checkpoint 选择规则名称。
        blind_pool_refresh_every: blind pool 的刷新间隔。
        blind_pool_start_epoch: 允许开始刷新 blind pool 的最小训练轮次。
        blind_pool_refresh_on_best_update: 新最佳 checkpoint 出现时是否立刻刷新 blind pool。
        blind_pool_max_complexes: 单次刷新 blind pool 时最多处理的复合物数量。
        blind_pool_cache_bce_weight: blind pool 回放中 BCE 损失的权重。
        blind_pool_cache_rank_weight: blind pool 回放中排序损失的权重。
        blind_pool_pairs_per_complex: 每个复合物在 blind pool 中采样的配对数量。
        replay_start_ratio: replay 回放监督启用的进度阈值。
        same_center_micro_batch_size: same-center ranking 分支的初始微批大小。
        same_center_budget_window_size: same-center ranking 回调窗口大小。
        same_center_budget_recover_window_count: same-center ranking 回升所需的连续干净窗口数。
        same_center_budget_recover_step: same-center ranking 每次回升的加性步长。
        same_center_offender_cooldown: same-center ranking 坏样本冷却时长。
        ranking_budget_window_size: wrong-center ranking 回调窗口大小。
        ranking_budget_recover_window_count: wrong-center ranking 回升所需的连续干净窗口数。
        ranking_offender_cooldown: ranking 坏样本冷却时长。
        ranking_wrong_center_cap: wrong-center ranking 的最大启用级别。
        replay_micro_batch_size: replay 候选打分的初始微批大小。
        replay_budget_window_size: replay 微批回调窗口大小。
        replay_budget_recover_window_count: replay 微批回升所需的连续干净窗口数。
        replay_candidate_cooldown: replay 复杂样本冷却时长。
        replay_max_candidates_per_complex: replay 每个复合物保留的最大候选数。
        geometry_min_atom_distance: Minimum ligand atom distance used by the preprocessing filter.
        resume_ckpt: 继续训练时要加载的 checkpoint 路径。
        resume_blind_pool_dir: 继续训练时优先读取 blind pool 缓存的目录。
        stop_after_epoch: 提前停止训练的绝对 epoch 编号，按 1-based 计数且包含该轮。
        esm_path: esm路径。
        run_name: 本次运行的名称标识。
        run_log_file: 本次运行对应的日志文件路径。

    Returns:
        None: 训练完成后不返回值，相关结果会写入日志、checkpoint 和报告文件。

    Raises:
        ValueError: 当关键训练配置缺失、设备非法或数据划分参数不合法时抛出。
        RuntimeError: 当训练、验证或候选池回放过程中出现不可恢复错误时抛出。
    """

    if device == "auto":
        raise ValueError(
            "Device must be provided explicitly. Please configure `device` in "
            "configs/train.toml or pass it from the caller."
        )
    device = torch.device(device)
    use_configured_cuda = device.type == "cuda"

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Using device: {device}")
    if run_name is not None:
        logger.info(f"Run name: {run_name}")
    if run_log_file is not None:
        logger.info(f"Run log file: {run_log_file}")

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)

    except Exception:
        pass

    logger.info("Initializing Dataset...")
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])

    interaction_profile = resolve_interaction_profile(ablation_mode=ablation_mode)

    dataset = build_dataset(
        root=data_root,
        index_file=index_file,
        esm_root=esm_path,
        esm=esm,
        esm_model_name=esm_model_name,
        esm_device=str(device),
        esm_dim=esm_dim,
        r_cutoff_intra=r_cutoff_intra,
        max_neighbors_intra=max_neighbors_intra,
        atom_neighbor_cap=atom_neighbor_cap,
        residue_neighbor_cap=residue_neighbor_cap,
        residue_radius_scale=residue_radius_scale,
        residue_radius_bias=residue_radius_bias,
        ligand_atom_fallback_k=ligand_atom_fallback_k,
        protein_atom_fallback_k=protein_atom_fallback_k,
        protein_residue_fallback_k=protein_residue_fallback_k,
        interaction_profile=interaction_profile,
        pre_filter=LigandGeometryPreFilter(
            min_atom_distance=float(geometry_min_atom_distance),
        ),
    )
    graph_builder = dataset.graph_builder

    if not math.isclose(split_train_frac + split_val_frac + split_test_frac, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Split fractions must sum to 1.0, got "
            f"{split_train_frac + split_val_frac + split_test_frac:.6f}."
        )

    if split_cache_file is None:
        raise ValueError(
            "split_cache_file must be configured explicitly. Please set it in "
            "configs/train.toml or pass it from the caller."
        )

    logger.info("Splitting dataset by Scaffold with persisted indices...")
    splitter = ScaffoldSplitter(include_chirality=False, seed=split_seed)

    split_indices: dict[str, list[int]]
    split_metadata: dict[str, Any] = {}
    if os.path.exists(split_cache_file) and not force_resplit:
        split_indices, split_metadata = ScaffoldSplitter.load_split(split_cache_file)
        max_idx = max(
            split_indices.get("train", [0])
            + split_indices.get("val", [0])
            + split_indices.get("test", [0])
        )
        cached_dataset_size = split_metadata.get("dataset_size")
        cached_index_file = split_metadata.get("index_file")
        cached_fractions = split_metadata.get("fractions", {})
        current_index_file = os.path.abspath(index_file)
        cached_index_file_abs = os.path.abspath(str(cached_index_file)) if cached_index_file is not None else None

        split_cache_mismatch = any([
            max_idx >= len(dataset),
            cached_dataset_size != len(dataset),
            cached_index_file_abs != current_index_file,
            split_metadata.get("seed") != split_seed,
            bool(split_metadata) and cached_fractions.get("train") != split_train_frac,
            bool(split_metadata) and cached_fractions.get("val") != split_val_frac,
            bool(split_metadata) and cached_fractions.get("test") != split_test_frac,
        ])

        if split_cache_mismatch:
            logger.warning(
                f"Cached split file {split_cache_file} is incompatible with current dataset configuration; regenerating. "
                f"(cached_dataset_size={cached_dataset_size}, current_dataset_size={len(dataset)}, "
                f"cached_index_file={cached_index_file}, current_index_file={current_index_file})"
            )
            split_indices = splitter.split_indices(
                dataset,
                frac_train=split_train_frac,
                frac_val=split_val_frac,
                frac_test=split_test_frac,
            )
            split_metadata = {
                "seed": split_seed,
                "include_chirality": False,
                "fractions": {
                    "train": split_train_frac,
                    "val": split_val_frac,
                    "test": split_test_frac,
                },
                "dataset_size": len(dataset),
                "index_file": current_index_file,
            }
            ScaffoldSplitter.save_split(split_cache_file, split_indices, metadata=split_metadata)
        else:
            logger.info(f"Loaded split indices from {split_cache_file}")
            logger.info(f"Split metadata: {split_metadata}")
    else:
        split_indices = splitter.split_indices(
            dataset,
            frac_train=split_train_frac,
            frac_val=split_val_frac,
            frac_test=split_test_frac,
        )
        split_metadata = {
            "seed": split_seed,
            "include_chirality": False,
            "fractions": {
                "train": split_train_frac,
                "val": split_val_frac,
                "test": split_test_frac,
            },
            "dataset_size": len(dataset),
            "index_file": os.path.abspath(index_file),
        }
        ScaffoldSplitter.save_split(split_cache_file, split_indices, metadata=split_metadata)
        logger.info(f"Saved split indices to {split_cache_file}")

    train_set, val_set, test_set = ScaffoldSplitter.subsets_from_indices(dataset, split_indices)
    train_indices = [int(i) for i in split_indices.get("train", [])]
    val_indices = [int(i) for i in split_indices.get("val", [])]
    test_indices = [int(i) for i in split_indices.get("test", [])]
    if not train_indices:
        raise ValueError("Train split is empty; cannot compute train-only normalization stats.")

    normalization_stats, train_affinity_stats = compute_train_split_normalization_stats(
        dataset,
        train_indices,
        split_cache_file=split_cache_file,
    )
    normalization_stats["affinity"] = {
        "mean": torch.tensor(train_affinity_stats["mean"], dtype=torch.float32),
        "std": torch.tensor(train_affinity_stats["std"], dtype=torch.float32),
    }
    dataset.set_affinity_stats(train_affinity_stats)
    logger.info(
        "Using train-only affinity stats: mean=%.4f std=%.4f",
        train_affinity_stats["mean"],
        train_affinity_stats["std"],
    )

    logger.info(
        f"Final Dataset Sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}"
    )

    if accumulation_steps < 1:
        raise ValueError(f"Invalid accumulation_steps={accumulation_steps}.")
    if not (0.0 < val_subset_ratio <= 1.0):
        raise ValueError(f"Invalid val_subset_ratio={val_subset_ratio}.")
    if val_full_every < 0:
        raise ValueError(f"Invalid val_full_every={val_full_every}.")
    if val_full_last_epochs < 0:
        raise ValueError(f"Invalid val_full_last_epochs={val_full_last_epochs}.")
    valid_val_ode_methods = {"euler", "rk4"}
    if ode_method not in valid_val_ode_methods:
        raise ValueError(
            f"Unsupported ode_method={ode_method!r}. "
            f"Choose from {tuple(sorted(valid_val_ode_methods))}."
        )
    progress_thresholds = {
        "center_guidance_learned_start": center_guidance_learned_start,
        "crop_proposal_start": crop_proposal_start,
        "crop_near_miss_start": crop_near_miss_start,
        "crop_hard_negative_start": crop_hard_negative_start,
        "ranking_same_center_start": ranking_same_center_start,
        "ranking_wrong_center_start": ranking_wrong_center_start,
        "pose_bootstrap_start": pose_bootstrap_start,
        "replay_start_ratio": replay_start_ratio,
    }
    for name, value in progress_thresholds.items():
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"Invalid {name}={value}.")
    if crop_proposal_start > crop_near_miss_start:
        raise ValueError(
            "crop_proposal_start must be <= crop_near_miss_start."
        )
    if crop_near_miss_start > crop_hard_negative_start:
        raise ValueError(
            "crop_near_miss_start must be <= crop_hard_negative_start."
        )
    if ranking_same_center_start > ranking_wrong_center_start:
        raise ValueError(
            "ranking_same_center_start must be <= ranking_wrong_center_start."
        )
    center_guidance_learned_end = max(
        float(center_guidance_learned_start),
        float(replay_start_ratio),
    )

    def _resolve_center_guidance_fraction(progress: float) -> float:
        return compute_center_guidance_fraction(
            progress=progress,
            learned_start=center_guidance_learned_start,
            learned_end=center_guidance_learned_end,
        )

    if not (0.0 < oom_reduce_factor < 1.0):
        raise ValueError(f"Invalid oom_reduce_factor={oom_reduce_factor}.")
    if train_budget_window_size < 1:
        raise ValueError(f"Invalid train_budget_window_size={train_budget_window_size}.")
    if train_budget_recover_window_count < 1:
        raise ValueError(
            f"Invalid train_budget_recover_window_count={train_budget_recover_window_count}."
        )
    if train_budget_recover_step < 1:
        raise ValueError(f"Invalid train_budget_recover_step={train_budget_recover_step}.")
    if train_offender_cooldown < 0:
        raise ValueError(f"Invalid train_offender_cooldown={train_offender_cooldown}.")
    if val_budget_window_size < 1:
        raise ValueError(f"Invalid val_budget_window_size={val_budget_window_size}.")
    if val_budget_recover_window_count < 1:
        raise ValueError(
            f"Invalid val_budget_recover_window_count={val_budget_recover_window_count}."
        )
    if val_budget_recover_step < 1:
        raise ValueError(f"Invalid val_budget_recover_step={val_budget_recover_step}.")
    if val_offender_cooldown < 0:
        raise ValueError(f"Invalid val_offender_cooldown={val_offender_cooldown}.")
    if ranking_budget_window_size < 1:
        raise ValueError(f"Invalid ranking_budget_window_size={ranking_budget_window_size}.")
    if ranking_budget_recover_window_count < 1:
        raise ValueError(
            f"Invalid ranking_budget_recover_window_count={ranking_budget_recover_window_count}."
        )
    if ranking_offender_cooldown < 0:
        raise ValueError(f"Invalid ranking_offender_cooldown={ranking_offender_cooldown}.")
    if same_center_micro_batch_size < 1:
        raise ValueError(f"Invalid same_center_micro_batch_size={same_center_micro_batch_size}.")
    if same_center_budget_window_size < 1:
        raise ValueError(
            f"Invalid same_center_budget_window_size={same_center_budget_window_size}."
        )
    if same_center_budget_recover_window_count < 1:
        raise ValueError(
            "Invalid same_center_budget_recover_window_count="
            f"{same_center_budget_recover_window_count}."
        )
    if same_center_budget_recover_step < 1:
        raise ValueError(
            f"Invalid same_center_budget_recover_step={same_center_budget_recover_step}."
        )
    if same_center_offender_cooldown < 0:
        raise ValueError(
            f"Invalid same_center_offender_cooldown={same_center_offender_cooldown}."
        )
    if ranking_wrong_center_cap < 0:
        raise ValueError(f"Invalid ranking_wrong_center_cap={ranking_wrong_center_cap}.")
    if replay_micro_batch_size < 1:
        raise ValueError(f"Invalid replay_micro_batch_size={replay_micro_batch_size}.")
    if replay_budget_window_size < 1:
        raise ValueError(f"Invalid replay_budget_window_size={replay_budget_window_size}.")
    if replay_budget_recover_window_count < 1:
        raise ValueError(
            f"Invalid replay_budget_recover_window_count={replay_budget_recover_window_count}."
        )
    if replay_candidate_cooldown < 0:
        raise ValueError(f"Invalid replay_candidate_cooldown={replay_candidate_cooldown}.")
    if replay_max_candidates_per_complex < 1:
        raise ValueError(
            f"Invalid replay_max_candidates_per_complex={replay_max_candidates_per_complex}."
        )
    if stop_after_epoch is not None and int(stop_after_epoch) < 1:
        raise ValueError(f"Invalid stop_after_epoch={stop_after_epoch}.")
    if not 0.0 <= float(min_checkpoint_selection_coverage) <= 1.0:
        raise ValueError(
            "min_checkpoint_selection_coverage must be in [0, 1], got "
            f"{min_checkpoint_selection_coverage}."
        )
    if int(max_val_non_oom_failures) < 0:
        raise ValueError(
            f"Invalid max_val_non_oom_failures={max_val_non_oom_failures}."
        )
    if int(max_val_oom_failures) < 0:
        raise ValueError(f"Invalid max_val_oom_failures={max_val_oom_failures}.")
    if not 0.0 <= float(final_topn_min_coverage) <= 1.0:
        raise ValueError(
            "final_topn_min_coverage must be in [0, 1], got "
            f"{final_topn_min_coverage}."
        )

    if val_cost_budget is None:
        raise ValueError("val_cost_budget must be configured explicitly.")
    if blind_pool_cost_budget is None:
        raise ValueError("blind_pool_cost_budget must be configured explicitly.")
    if final_topn_cost_budget is None:
        raise ValueError("final_topn_cost_budget must be configured explicitly.")
    if min_val_cost_budget is None:
        raise ValueError("min_val_cost_budget must be configured explicitly.")

    configured_train_cost_budget = max(1, int(train_cost_budget))
    configured_val_cost_budget = max(1, int(val_cost_budget))
    configured_blind_pool_cost_budget = max(1, int(blind_pool_cost_budget))
    configured_final_topn_cost_budget = max(1, int(final_topn_cost_budget))
    eval_cost_guard_headroom = max(1.0, float(eval_cost_guard_headroom))
    training_config_snapshot = build_training_config_snapshot(locals())
    checkpoint_model_config_kwargs = build_checkpoint_model_config_kwargs(locals())


    def _annotate_loss_context(batch_obj: Any, *, current_epoch: int, total_epochs_count: int, warmup_epochs_count: int, training: bool) -> None:
        apply_loss_context(
            batch_obj,
            current_epoch=current_epoch,
            total_epochs_count=total_epochs_count,
            warmup_epochs_count=warmup_epochs_count,
            training=training,
        )

    def _to_debug_scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 1:
                scalar = value.detach().cpu().item()
                return float(scalar) if isinstance(scalar, (float, int)) else scalar
            return {
                "shape": list(value.shape),
                "min": float(value.detach().amin().cpu().item()) if value.numel() > 0 else 0.0,
                "max": float(value.detach().amax().cpu().item()) if value.numel() > 0 else 0.0,
            }
        return value

    def _summarize_loss_debug(loss_items: dict[str, Any]) -> dict[str, Any]:
        debug_summary: dict[str, Any] = {}
        for key, value in loss_items.items():
            if key.startswith(("loss_", "weight_", "rank_pairs_", "energy_nan_")) or key == "total":
                debug_summary[key] = _to_debug_scalar(value)
        return debug_summary

    def _collect_batch_sample_debug(batch_obj: Any, *, limit: int = 8) -> list[dict[str, Any]]:
        samples = batch_obj.to_data_list() if hasattr(batch_obj, "to_data_list") else [batch_obj]
        summaries: list[dict[str, Any]] = []
        for sample in samples[:limit]:
            ligand_atoms = int(sample["ligand_atom"].num_nodes) if "ligand_atom" in sample.node_types else 0
            protein_residues = int(sample["protein_residue"].num_nodes) if "protein_residue" in sample.node_types else 0
            protein_atoms = int(sample["protein_atom"].num_nodes) if "protein_atom" in sample.node_types else 0
            summaries.append(
                {
                    "pdb_id": str(getattr(sample, "pdb_id", "unknown")),
                    "dataset_index": (
                        int(getattr(sample, "dataset_index"))
                        if getattr(sample, "dataset_index", None) is not None
                        else None
                    ),
                    "ligand_atoms": ligand_atoms,
                    "protein_atoms": protein_atoms,
                    "protein_residues": protein_residues,
                }
            )
        return summaries

    def _collect_nonfinite_grad_param_names(*, limit: int = 12) -> dict[str, Any]:
        bad_names: list[str] = []
        total_bad = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isfinite(param.grad).all():
                continue
            total_bad += 1
            if len(bad_names) < limit:
                bad_names.append(name)
        return {
            "count": total_bad,
            "examples": bad_names,
        }

    def _probe_loss_branch_gradients(
        loss_terms: dict[str, torch.Tensor | None],
        *,
        limit: int = 12,
    ) -> dict[str, Any]:
        branch_summary: dict[str, Any] = {}
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        param_names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        for branch_name, branch_loss in loss_terms.items():
            if branch_loss is None:
                branch_summary[branch_name] = {"status": "missing"}
                continue
            if branch_loss.grad_fn is None:
                branch_summary[branch_name] = {
                    "status": "no_grad_fn",
                    "value": _to_debug_scalar(branch_loss),
                }
                continue
            grads = torch.autograd.grad(
                branch_loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            nonfinite_names: list[str] = []
            total_nonfinite = 0
            max_abs_grad = 0.0
            for param_name, grad in zip(param_names, grads):
                if grad is None:
                    continue
                finite_mask = torch.isfinite(grad)
                if not finite_mask.all():
                    total_nonfinite += 1
                    if len(nonfinite_names) < limit:
                        nonfinite_names.append(param_name)
                    finite_grad = grad[finite_mask]
                    if finite_grad.numel() > 0:
                        max_abs_grad = max(
                            max_abs_grad,
                            float(finite_grad.abs().max().detach().cpu().item()),
                        )
                    continue
                if grad.numel() > 0:
                    max_abs_grad = max(
                        max_abs_grad,
                        float(grad.abs().max().detach().cpu().item()),
                    )
            branch_summary[branch_name] = {
                "status": "nonfinite" if total_nonfinite > 0 else "finite",
                "loss": _to_debug_scalar(branch_loss),
                "nonfinite_param_count": total_nonfinite,
                "nonfinite_examples": nonfinite_names,
                "max_abs_grad": max_abs_grad,
            }
        return branch_summary

    def _log_nonfinite_training_debug(
        *,
        reason: str,
        batch_idx: int,
        source_batch_obj: Any,
        local_batch_obj: Any,
        loss_items: dict[str, Any],
        combined_loss: torch.Tensor,
        grad_norm: torch.Tensor | None = None,
        branch_probe: dict[str, Any] | None = None,
    ) -> None:
        logger.warning(
            "Non-finite training state detected | reason=%s | epoch=%d | batch=%d | "
            "combined_loss=%s | grad_norm=%s",
            reason,
            epoch + 1,
            batch_idx,
            _to_debug_scalar(combined_loss),
            _to_debug_scalar(grad_norm) if grad_norm is not None else None,
        )
        logger.warning("Loss breakdown: %s", _summarize_loss_debug(loss_items))
        logger.warning(
            "Source batch samples: %s",
            _collect_batch_sample_debug(source_batch_obj),
        )
        logger.warning(
            "Local batch samples: %s",
            _collect_batch_sample_debug(local_batch_obj),
        )
        logger.warning(
            "Non-finite gradient parameters: %s",
            _collect_nonfinite_grad_param_names(),
        )
        if branch_probe is not None:
            logger.warning("Branch gradient probe: %s", branch_probe)

    def _log_budget_adjustment(
        adjustment: Any | None,
        *,
        budget_label: str,
        rebuild_loader: bool = False,
    ) -> None:
        if adjustment is None or adjustment.new_budget == adjustment.previous_budget:
            return
        log_fn = logger.warning if adjustment.action == "reduce" else logger.info
        log_fn(
            "%s budget callback | phase=%s | action=%s | %s: %d -> %d | "
            "window_oom=%d/%d | offenders=%d | reason=%s",
            budget_label,
            adjustment.phase_name,
            adjustment.action,
            budget_label,
            adjustment.previous_budget,
            adjustment.new_budget,
            adjustment.window_oom,
            adjustment.window_total,
            adjustment.offender_count,
            adjustment.reason,
        )
        if rebuild_loader:
            logger.info(
                "Next epoch will rebuild the train loader with %s=%d.",
                budget_label,
                adjustment.new_budget,
            )

    def _run_same_center_auxiliary_step(
        *,
        local_samples: list[Any],
        placement_centers_cpu: torch.Tensor,
        root_indices: list[int],
        batch_idx: int,
    ) -> tuple[torch.Tensor, int, bool, bool]:
        """
        以独立辅助反传方式执行 same-center pairwise ranking。

        Args:
            local_samples: 当前 local crop batch 对应的逐图样本列表。
            placement_centers_cpu: 当前 batch 每个图对应的裁剪中心。
            root_indices: 当前根 batch 的底层数据集索引。
            batch_idx: 当前训练 batch 编号。

        Returns:
            tuple[torch.Tensor, int, bool, bool]:
                返回聚合后的 same-center 监控损失、有效配对数、
                是否发生 OOM、以及是否存在不可约 OOM。
        """

        def _run_chunk(
            *,
            chunk_samples: list[Any],
            chunk_centers_cpu: torch.Tensor,
        ) -> tuple[torch.Tensor, int, bool, bool]:
            try:
                anchor_batch = cast(Any, collator.collate(chunk_samples)).to(device)
                x_1_anchor = anchor_batch["ligand_atom"].pos
                chunk_centers = chunk_centers_cpu.to(
                    device=device,
                    dtype=x_1_anchor.dtype,
                )
                with torch.no_grad():
                    t_anchor, x_t_anchor, _ = matcher.sample_location_and_target(
                        x_1=x_1_anchor,
                        data=anchor_batch,
                        current_epoch=epoch,
                        total_epochs=epochs,
                        placement_centers=chunk_centers,
                    )
                anchor_batch["ligand_atom"].pos = x_t_anchor
                anchor_batch.t = t_anchor
                anchor_pred = model(anchor_batch, t_anchor)
                pose_rank_anchor = compute_pose_rank_target(
                    x_t_anchor,
                    x_1_anchor,
                    batch_idx=anchor_batch["ligand_atom"].batch,
                    samples=chunk_samples,
                    dataset_raw_dir=dataset.raw_dir,
                )
                anchor_rank_logit = select_pose_rank_logit(anchor_pred)

                same_center_batch = cast(Any, collator.collate(chunk_samples)).to(device)
                x_1_same = same_center_batch["ligand_atom"].pos
                with torch.no_grad():
                    t_same = torch.clamp(
                        t_anchor * (0.25 + 0.45 * torch.rand_like(t_anchor)),
                        min=1e-3,
                        max=1.0 - 1e-3,
                    )
                    _, x_t_same, _ = matcher.sample_location_and_target(
                        x_1=x_1_same,
                        data=same_center_batch,
                        current_epoch=epoch,
                        total_epochs=epochs,
                        placement_centers=chunk_centers,
                        t_override=t_same,
                    )
                same_center_batch["ligand_atom"].pos = x_t_same
                same_center_batch.t = t_same
                same_center_pred = model(same_center_batch, t_same)
                pose_rank_same = compute_pose_rank_target(
                    x_t_same,
                    x_1_same,
                    batch_idx=same_center_batch["ligand_atom"].batch,
                    samples=chunk_samples,
                    dataset_raw_dir=dataset.raw_dir,
                )
                same_center_rank_logit = select_pose_rank_logit(same_center_pred)
                loss_same, count_same = pairwise_ranking_loss_from_pairs(
                    anchor_rank_logit,
                    pose_rank_anchor,
                    same_center_rank_logit,
                    pose_rank_same,
                    margin=pose_ranking_margin,
                )
                if count_same > 0:
                    auxiliary_loss = (
                        pose_ranking_pair_weight
                        * loss_same
                        * max(1, len(chunk_samples))
                    )
                    auxiliary_loss.backward()
                    monitored_loss = loss_same.detach() * count_same
                else:
                    monitored_loss = loss_same.detach().new_zeros(())
                del (
                    anchor_batch,
                    anchor_pred,
                    pose_rank_anchor,
                    anchor_rank_logit,
                    same_center_batch,
                    same_center_pred,
                    pose_rank_same,
                    same_center_rank_logit,
                    x_1_anchor,
                    x_t_anchor,
                    x_1_same,
                    x_t_same,
                    t_anchor,
                    t_same,
                )
                return monitored_loss, count_same, False, False
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                if use_configured_cuda:
                    torch.cuda.empty_cache()
                if len(chunk_samples) <= 1:
                    logger.warning(
                        "Batch %d: irreducible same-center ranking OOM on singleton micro-batch.",
                        batch_idx,
                    )
                    return torch.zeros((), device=device), 0, True, True
                mid = len(chunk_samples) // 2
                left = _run_chunk(
                    chunk_samples=chunk_samples[:mid],
                    chunk_centers_cpu=chunk_centers_cpu[:mid],
                )
                right = _run_chunk(
                    chunk_samples=chunk_samples[mid:],
                    chunk_centers_cpu=chunk_centers_cpu[mid:],
                )
                return (
                    left[0] + right[0],
                    left[1] + right[1],
                    True,
                    left[3] or right[3],
                )

        same_center_action = same_center_budget_controller.get_batch_cooldown_action(
            root_indices,
            len(local_samples),
        )
        if same_center_action == "skip":
            same_center_budget_controller.note_cooldown_skip(root_indices)
            return torch.zeros((), device=device), 0, False, False

        weighted_loss_total = torch.zeros((), device=device)
        pair_count_total = 0
        had_oom = False
        irreducible_oom = False
        chunk_size = max(
            1,
            min(int(same_center_budget_controller.current_budget), len(local_samples)),
        )
        for start in range(0, len(local_samples), chunk_size):
            end = min(start + chunk_size, len(local_samples))
            chunk_weighted_loss, chunk_pair_count, chunk_had_oom, chunk_irreducible = _run_chunk(
                chunk_samples=local_samples[start:end],
                chunk_centers_cpu=placement_centers_cpu[start:end],
            )
            weighted_loss_total = weighted_loss_total + chunk_weighted_loss
            pair_count_total += chunk_pair_count
            had_oom = had_oom or chunk_had_oom
            irreducible_oom = irreducible_oom or chunk_irreducible

        return weighted_loss_total, pair_count_total, had_oom, irreducible_oom


    effective_min_train_cost_budget = max(1, int(min_train_cost_budget))
    effective_min_val_cost_budget = max(1, int(min_val_cost_budget))

    if effective_min_train_cost_budget > configured_train_cost_budget:
        logger.warning(
            f"min_train_cost_budget ({effective_min_train_cost_budget}) is greater than "
            f"train_cost_budget ({configured_train_cost_budget}); clamping min to max."
        )
        effective_min_train_cost_budget = configured_train_cost_budget

    if effective_min_val_cost_budget > configured_val_cost_budget:
        logger.warning(
            f"min_val_cost_budget ({effective_min_val_cost_budget}) is greater than "
            f"val_cost_budget ({configured_val_cost_budget}); clamping min to val max."
        )
        effective_min_val_cost_budget = configured_val_cost_budget

    if not (0.0 < val_oom_reduce_factor < 1.0):
        raise ValueError(
            f"Invalid val_oom_reduce_factor={val_oom_reduce_factor}."
        )
    if max_oom_retry_splits < 0:
        raise ValueError(f"Invalid max_oom_retry_splits={max_oom_retry_splits}.")

    train_budget_controller = WindowAimdBudgetController(
        phase_name="train",
        base_budget=configured_train_cost_budget,
        min_budget=effective_min_train_cost_budget,
        window_size=train_budget_window_size,
        reduce_threshold=oom_reduce_threshold,
        reduce_factor=oom_reduce_factor,
        recover_window_count=train_budget_recover_window_count,
        recover_step=train_budget_recover_step,
        offender_cooldown=train_offender_cooldown,
        enable_adaptive=enable_train_budget_callback,
    )
    val_partial_budget_controller = WindowAimdBudgetController(
        phase_name="val_partial",
        base_budget=configured_val_cost_budget,
        min_budget=effective_min_val_cost_budget,
        window_size=val_budget_window_size,
        reduce_threshold=val_oom_reduce_threshold,
        reduce_factor=val_oom_reduce_factor,
        recover_window_count=val_budget_recover_window_count,
        recover_step=val_budget_recover_step,
        offender_cooldown=val_offender_cooldown,
        enable_adaptive=enable_val_budget_callback,
    )
    val_full_budget_controller = WindowAimdBudgetController(
        phase_name="val_full",
        base_budget=configured_val_cost_budget,
        min_budget=effective_min_val_cost_budget,
        window_size=val_budget_window_size,
        reduce_threshold=val_oom_reduce_threshold,
        reduce_factor=val_oom_reduce_factor,
        recover_window_count=val_budget_recover_window_count,
        recover_step=val_budget_recover_step,
        offender_cooldown=val_offender_cooldown,
        enable_adaptive=enable_val_budget_callback,
    )
    same_center_budget_controller = WindowAimdBudgetController(
        phase_name="ranking_same_center",
        base_budget=max(1, int(same_center_micro_batch_size)),
        min_budget=1,
        window_size=same_center_budget_window_size,
        reduce_threshold=1,
        reduce_factor=0.5,
        recover_window_count=same_center_budget_recover_window_count,
        recover_step=same_center_budget_recover_step,
        offender_cooldown=same_center_offender_cooldown,
        enable_adaptive=True,
    )
    ranking_budget_controller = WindowAimdBudgetController(
        phase_name="ranking_wrong_center",
        base_budget=max(0, int(ranking_wrong_center_cap)),
        min_budget=0,
        window_size=ranking_budget_window_size,
        reduce_threshold=1,
        reduce_factor=0.5,
        recover_window_count=ranking_budget_recover_window_count,
        recover_step=1,
        offender_cooldown=ranking_offender_cooldown,
        enable_adaptive=True,
    )
    replay_budget_controller = WindowAimdBudgetController(
        phase_name="replay_micro_batch",
        base_budget=max(1, int(replay_micro_batch_size)),
        min_budget=1,
        window_size=replay_budget_window_size,
        reduce_threshold=1,
        reduce_factor=0.5,
        recover_window_count=replay_budget_recover_window_count,
        recover_step=1,
        offender_cooldown=replay_candidate_cooldown,
        enable_adaptive=True,
    )
    persistent_workers = bool(dataloader_persistent_workers and dataloader_num_workers > 0)
    graph_cost_profile_cache: dict[int, dict[str, int]] = {}
    train_phase_multiplier = 1.0
    val_partial_phase_multiplier = 1.35
    val_full_phase_multiplier = 1.75
    topn_phase_multiplier = 2.25

    def _get_graph_cost_profile(root_idx: int) -> dict[str, int]:
        if root_idx not in graph_cost_profile_cache:
            graph_cost_profile_cache[root_idx] = dataset.get_graph_cost_profile(root_idx)
        return graph_cost_profile_cache[root_idx]

    def _estimate_sample_costs(dataset_obj: Any, *, phase_multiplier: float) -> list[int]:
        root_indices = resolve_subset_root_indices(dataset_obj)
        return [
            estimate_graph_cost_units(
                _get_graph_cost_profile(root_idx),
                num_gnn_blocks=num_gnn_blocks,
                dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                phase_multiplier=phase_multiplier,
            )
            for root_idx in root_indices
        ]

    def _summarize_sample_costs(name: str, root_indices: list[int]) -> dict[str, Any]:
        if not root_indices:
            return {
                "name": name,
                "samples": 0,
            }
        profiles = [_get_graph_cost_profile(root_idx) for root_idx in root_indices]
        costs = [
            estimate_graph_cost_units(
                profile,
                num_gnn_blocks=num_gnn_blocks,
                dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                phase_multiplier=train_phase_multiplier,
            )
            for profile in profiles
        ]
        total_edges = [int(profile["total_edges"]) for profile in profiles]
        dynamic_candidates = [
            int(min(profile["protein_residue_nodes"], dynamic_residue_candidate_topk))
            if dynamic_residue_candidate_topk > 0
            else int(profile["protein_residue_nodes"])
            for profile in profiles
        ]
        return {
            "name": name,
            "samples": len(root_indices),
            "cost_mean": float(np.mean(costs)),
            "cost_p95": float(np.percentile(costs, 95)),
            "cost_p99": float(np.percentile(costs, 99)),
            "edges_mean": float(np.mean(total_edges)),
            "edges_p95": float(np.percentile(total_edges, 95)),
            "residue_candidates_mean": float(np.mean(dynamic_candidates)),
        }

    _prev_loaders: list[DataLoader] = []

    def _rebuild_train_loader(train_base_cost_budget: int) -> DataLoader:
        for old_loader in _prev_loaders:
            del old_loader
        _prev_loaders.clear()
        gc.collect()

        logger.info(
            "Using adaptive train sampler cost budget: "
            f"train_cost_budget={train_base_cost_budget}."
        )
        train_sampler = AdaptiveCostBatchSampler(
            sample_costs=_estimate_sample_costs(train_set, phase_multiplier=train_phase_multiplier),
            max_cost=train_base_cost_budget,
            shuffle=True,
            seed=42,
        )
        train_loader_local = DataLoader(
            train_set,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=train_sampler,
        )

        _prev_loaders.append(train_loader_local)
        return train_loader_local

    def _build_eval_loader(subset: Any, base_cost_budget: int, *, phase_multiplier: float) -> DataLoader:
        eval_sampler = AdaptiveCostBatchSampler(
            sample_costs=_estimate_sample_costs(subset, phase_multiplier=phase_multiplier),
            max_cost=base_cost_budget,
            shuffle=False,
            seed=42,
        )
        return DataLoader(
            subset,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=False,
            pin_memory=dataloader_pin_memory,
            batch_sampler=eval_sampler,
        )

    def _sample_validation_subset(*, epoch_idx: int, subset_ratio: float) -> Any:
        """
        按比例抽取当前 epoch 使用的验证子集。

        Args:
            epoch_idx: 当前训练轮次，从 0 开始计数。
            subset_ratio: 本轮验证需要覆盖的验证集比例。

        Returns:
            Any: 若比例达到全量则直接返回完整验证集，否则返回按固定随机种子采样后的 `Subset`。
        """
        if len(val_set) == 0:
            return val_set
        if subset_ratio >= 1.0:
            return val_set
        target_size = min(len(val_set), max(1, math.ceil(len(val_set) * subset_ratio)))
        if target_size >= len(val_set):
            return val_set
        generator = torch.Generator()
        generator.manual_seed(42 + epoch_idx)
        sampled_positions = torch.randperm(len(val_set), generator=generator)[:target_size].tolist()
        return Subset(val_set, sampled_positions)

    def _resolve_validation_plan(epoch_idx: int) -> dict[str, Any]:
        """
        解析当前 epoch 的验证调度方案。

        Args:
            epoch_idx: 当前训练轮次，从 0 开始计数。

        Returns:
            dict[str, Any]: 返回本轮验证模式、触发原因、子集对象、覆盖比例和 ODE 方法等信息。
        """
        is_tail_full = val_full_last_epochs > 0 and epoch_idx >= max(0, epochs - val_full_last_epochs)
        is_periodic_full = val_full_every > 0 and (epoch_idx + 1) % val_full_every == 0
        is_full = bool(is_tail_full or is_periodic_full or val_subset_ratio >= 1.0)
        subset_ratio = 1.0 if is_full else val_subset_ratio
        subset = _sample_validation_subset(epoch_idx=epoch_idx, subset_ratio=subset_ratio)
        subset_size = len(subset)
        effective_ratio = 0.0 if len(val_set) == 0 else subset_size / len(val_set)
        if is_tail_full:
            trigger = "tail"
        elif is_periodic_full:
            trigger = "periodic"
        elif is_full:
            trigger = "ratio"
        else:
            trigger = "partial"
        mode = "full" if is_full else "partial"
        resolved_ode_method = ode_method
        phase_multiplier = val_full_phase_multiplier if is_full else val_partial_phase_multiplier
        if resolved_ode_method == "rk4":
            phase_multiplier *= 1.6
        return {
            "mode": mode,
            "trigger": trigger,
            "is_full": is_full,
            "subset": subset,
            "subset_size": subset_size,
            "subset_ratio": effective_ratio,
            "ode_method": resolved_ode_method,
            "phase_multiplier": phase_multiplier,
            "progress_desc": f"Epoch {epoch_idx + 1} [Val:{mode}]",
        }

    train_loader = _rebuild_train_loader(train_budget_controller.current_budget)
    graph_cost_summary = {
        "train": _summarize_sample_costs("train", train_indices),
        "val": _summarize_sample_costs("val", val_indices),
        "test": _summarize_sample_costs("test", test_indices),
    }
    logger.info("Graph cost summary | train=%s | val=%s | test=%s", graph_cost_summary["train"], graph_cost_summary["val"], graph_cost_summary["test"])
    report_dir = os.path.join(save_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    graph_cost_report_path = os.path.join(report_dir, "graph_cost_summary.json")
    with open(graph_cost_report_path, "w", encoding="utf-8") as f:
        json.dump(graph_cost_summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Validation schedule: partial_ratio=%.2f%% | full_every=%d | full_last_epochs=%d | ode=%s.",
        100.0 * val_subset_ratio,
        val_full_every,
        val_full_last_epochs,
        ode_method,
    )
    if val_subset_ratio < 0.10 and val_full_every == 0 and val_full_last_epochs == 0:
        logger.warning(
            "Validation feedback is configured to be extremely sparse | partial_ratio=%.2f%% | no scheduled full validation.",
            100.0 * val_subset_ratio,
        )
    if blind_pool_start_epoch <= 0 and blind_pool_refresh_every <= 1:
        logger.warning(
            "Blind pool refresh is configured to start very early and run every epoch; early-epoch pool quality may be noisy."
        )
    logger.info(
        "Center guidance schedule: learned_start=%.2f | learned_full=%.2f | source=replay-aligned blend.",
        center_guidance_learned_start,
        center_guidance_learned_end,
    )
    logger.info(
        "Evaluation budgets: "
        f"blind_pool_cost_budget={configured_blind_pool_cost_budget} | "
        f"final_topn_cost_budget={configured_final_topn_cost_budget}."
    )

    logger.info("Initializing Model & Flow Components...")

    model = build_model(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        interaction_profile=interaction_profile,
        normalization_stats=normalization_stats,
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
        dynamic_inter_cutoff=dynamic_inter_cutoff,
        dynamic_inter_knn_k=dynamic_inter_knn_k,
        dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
        dynamic_residue_cutoff=dynamic_residue_cutoff,
        dynamic_residue_knn_k=dynamic_residue_knn_k,
        dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
        dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
    ).to(device)

    matcher = ConditionalFlowMatcher(
        sigma_min=flow_sigma_min,
        spatial_sigma_min=flow_spatial_sigma_min,
        spatial_sigma_max=flow_spatial_sigma_max,
        warmup_epochs=warmup_epochs,
        fd_dt=flow_fd_dt,
        rotation_angle_min=flow_rotation_angle_min,
        rotation_angle_max=flow_rotation_angle_max,
        torsion_scale_min=flow_torsion_scale_min,
        torsion_scale_max=flow_torsion_scale_max,
    )
    criterion = FlowMatchingLoss(
        characteristic_scale=loss_characteristic_scale,
        weight_translation=loss_weight_translation,
        weight_rotation=loss_weight_rotation,
        weight_torsion=loss_weight_torsion,
        weight_energy=loss_weight_energy,
        weight_clash=loss_weight_clash,
        weight_pose_rank=loss_weight_pose_rank,
        curriculum_weights={
            "coarse": {
                "translation": loss_coarse_translation,
                "rotation": loss_coarse_rotation,
                "torsion": loss_coarse_torsion,
                "energy": loss_coarse_energy,
                "clash": loss_coarse_clash,
                "pose_rank": loss_coarse_pose_rank,
            },
            "transition": {
                "translation": loss_transition_translation,
                "rotation": loss_transition_rotation,
                "torsion": loss_transition_torsion,
                "energy": loss_transition_energy,
                "clash": loss_transition_clash,
                "pose_rank": loss_transition_pose_rank,
            },
            "refine": {
                "translation": loss_refine_translation,
                "rotation": loss_refine_rotation,
                "torsion": loss_refine_torsion,
                "energy": loss_refine_energy,
                "clash": loss_refine_clash,
                "pose_rank": loss_refine_pose_rank,
            },
        },
        refine_start=loss_refine_start,
        pose_gate_epoch_start=loss_pose_gate_epoch_start,
        pose_gate_epoch_end=loss_pose_gate_epoch_end,
        pose_gate_tau_start=loss_pose_gate_tau_start,
        pose_gate_tau_end=loss_pose_gate_tau_end,
        pose_gate_temperature=loss_pose_gate_temperature,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    try:
        total_train_batches = len(train_loader)
    except ValueError:
        total_train_batches = max(1, len(train_set) // 4)
    updates_per_epoch = math.ceil(total_train_batches / accumulation_steps)
    total_steps = max(1, epochs * updates_per_epoch)
    requested_warmup_steps = max(0, warmup_epochs) * updates_per_epoch
    warmup_steps = min(requested_warmup_steps, max(0, total_steps - 1))
    if requested_warmup_steps != warmup_steps:
        logger.warning(
            "Warmup steps were clamped to fit the current run | requested=%d | effective=%d | total_steps=%d.",
            requested_warmup_steps,
            warmup_steps,
            total_steps,
        )
    cosine_eta_min = 1e-6
    eta_min_factor = cosine_eta_min / lr if lr > 0 else 0.0

    def _lr_multiplier(step_idx: int) -> float:
        if total_steps <= 1:
            return 1.0
        clamped_step = min(max(step_idx, 0), total_steps - 1)
        if warmup_steps > 0 and clamped_step < warmup_steps:
            warmup_progress = clamped_step / max(1, warmup_steps)
            return 0.01 + 0.99 * warmup_progress
        cosine_progress = (
            (clamped_step - warmup_steps)
            / max(1, total_steps - warmup_steps - 1)
        )
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_weight

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_multiplier)
    logger.info(
        "LR scheduler initialized: total_steps=%d, warmup_steps=%d, requested_warmup_steps=%d.",
        total_steps,
        warmup_steps,
        requested_warmup_steps,
    )

    ema_model: AveragedModel | None = None

    runtime_state = TrainerRuntimeState(
        best_val_loss=float("inf"),
        best_rmsd=float("inf"),
        best_composite_metrics=None,
        best_single_shot_success2a_metrics=None,
        best_rmsd_metrics=None,
        best_selected_metrics=None,
        current_fusion_weights=dict(DEFAULT_FUSION_WEIGHTS),
        total_oom_batches=0,
        runtime_benchmark_history=[],
        clone_safety_checked=False,
    )
    selected_primary_key, selected_higher_is_better, selected_metric_label = resolve_selection_rule(
        checkpoint_selection_mode
    )
    start_epoch = 0
    resume_blind_pool_source_dir: str | None = None
    if resume_ckpt is not None:
        resume_result = load_resume_checkpoint(
            resume_ckpt=resume_ckpt,
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
        start_epoch = resume_result.start_epoch
        resume_blind_pool_source_dir = resume_result.resume_blind_pool_source_dir
        ema_model = resume_result.ema_model
        runtime_state = resume_result.runtime_state
    best_val_loss = runtime_state.best_val_loss
    best_rmsd = runtime_state.best_rmsd
    best_composite_metrics = runtime_state.best_composite_metrics
    best_single_shot_success2a_metrics = (
        runtime_state.best_single_shot_success2a_metrics
    )
    best_rmsd_metrics = runtime_state.best_rmsd_metrics
    best_selected_metrics = runtime_state.best_selected_metrics
    current_fusion_weights = dict(runtime_state.current_fusion_weights)
    total_oom_batches = runtime_state.total_oom_batches
    runtime_benchmark_history = list(runtime_state.runtime_benchmark_history)
    clone_safety_checked = runtime_state.clone_safety_checked
    effective_stop_after_epoch = epochs
    if stop_after_epoch is not None:
        effective_stop_after_epoch = min(int(stop_after_epoch), epochs)
        if effective_stop_after_epoch != int(stop_after_epoch):
            logger.info(
                "stop_after_epoch=%d is beyond configured epochs=%d; clamping to %d.",
                int(stop_after_epoch),
                epochs,
                effective_stop_after_epoch,
            )
    if start_epoch >= effective_stop_after_epoch:
        raise ValueError(
            "No epochs left to run after applying resume/stop settings. "
            f"start_epoch={start_epoch + 1}, stop_after_epoch={effective_stop_after_epoch}, epochs={epochs}."
        )
    blind_pool_cache_dir = os.path.join(save_dir, "blind_pool_cache")
    os.makedirs(blind_pool_cache_dir, exist_ok=True)
    blind_pool_signature = build_blind_pool_signature(
        esm_dim=esm_dim,
        processed_dir=dataset.processed_dir,
        index_file=dataset.index_file,
        interaction_profile=interaction_profile,
    )
    cached_blind_pool: list[dict[str, Any]] = load_blind_pool(
        blind_pool_cache_dir,
        expected_signature=blind_pool_signature,
    )
    if cached_blind_pool:
        logger.info("Loaded existing blind pool: %d complexes.", len(cached_blind_pool))
    elif resume_blind_pool_source_dir is not None:
        cached_blind_pool = load_blind_pool(
            resume_blind_pool_source_dir,
            expected_signature=blind_pool_signature,
        )
        if cached_blind_pool:
            logger.info(
                "Loaded resume blind pool from %s: %d complexes.",
                resume_blind_pool_source_dir,
                len(cached_blind_pool),
            )
    best_selected_updated_this_epoch = False

    for epoch in range(start_epoch, effective_stop_after_epoch):
        epoch_start_time = time.perf_counter()
        epoch_progress = 1.0 if epochs <= 1 else epoch / max(1, epochs - 1)
        gc.collect()
        if use_configured_cuda:
            torch.cuda.empty_cache()
        train_loader = _rebuild_train_loader(train_budget_controller.current_budget)

        model.train()
        criterion.train()

        train_loss_meter = 0.0
        pbar = tqdm(total=len(train_set), desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="graphs")

        actual_batches = 0
        epoch_root_oom_events = 0
        epoch_cost_guard_skips = 0
        epoch_cooldown_skips = 0
        accumulated_graphs = 0
        accumulated_batches = 0
        ENERGY_NAN_FAILFAST_LIMIT = 8
        consecutive_energy_nan_skips = 0
        optimizer.zero_grad()
        epoch_local_losses: list[float] = []
        epoch_source_residues: list[float] = []
        epoch_local_residues: list[float] = []
        epoch_rank_pair_counts = {
            "same_center": 0,
            "wrong_center_low_clash": 0,
            "misleading_center": 0,
            "misleading_affinity": 0,
        }
        epoch_rank_oom_skips = 0
        epoch_rank_peak_mem_mb = 0.0
        epoch_energy_nan_skips = 0

        for batch_idx, batch in enumerate(train_loader):
            root_num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
            root_batch_indices = extract_root_dataset_indices(batch)
            root_event_had_oom = False
            root_event_irreducible_oom = False
            root_event_same_center_oom = False
            root_event_same_center_irreducible_oom = False
            root_event_rank_oom = False
            pbar.update(root_num_graphs)
            pending_batches: list[tuple[Any, int]] = [(batch, 0)]

            while pending_batches:
                batch, split_depth = pending_batches.pop(0)
                num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
                current_root_indices = extract_root_dataset_indices(batch)
                cooldown_action = train_budget_controller.get_batch_cooldown_action(
                    current_root_indices,
                    num_graphs,
                )
                if cooldown_action == "split" and split_depth < max_oom_retry_splits:
                    split_batches = split_collated_batch(batch, collator=collator)
                    if split_batches is not None:
                        pending_batches = [
                            (split_batches[0], split_depth + 1),
                            (split_batches[1], split_depth + 1),
                        ] + pending_batches
                        continue
                if cooldown_action == "skip":
                    epoch_cooldown_skips += 1
                    train_budget_controller.note_cooldown_skip(current_root_indices)
                    logger.warning(
                        "Batch %d: skipping singleton offender batch during cooldown | offenders=%s.",
                        batch_idx,
                        current_root_indices,
                    )
                    continue
                runtime_train_cost_limit = max(1, int(train_budget_controller.current_budget))
                batch_cost_units = estimate_runtime_batch_cost(
                    batch,
                    num_gnn_blocks=num_gnn_blocks,
                    dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                    dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                    dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                    phase_multiplier=train_phase_multiplier,
                )
                if batch_cost_units > runtime_train_cost_limit:
                    split_batches = (
                        split_collated_batch(batch, collator=collator)
                        if split_depth < max_oom_retry_splits
                        else None
                    )
                    if split_batches is not None:
                        pending_batches = [
                            (split_batches[0], split_depth + 1),
                            (split_batches[1], split_depth + 1),
                        ] + pending_batches
                        continue
                    epoch_cost_guard_skips += 1
                    train_budget_controller.mark_offender(current_root_indices, severe=True)
                    logger.warning(
                        f"Batch {batch_idx}: skipping irreducible oversized cost batch "
                        f"(cost={batch_cost_units} > limit={runtime_train_cost_limit})."
                    )
                    continue

                source_batch: Any | None = None
                local_batch: Any | None = None
                predictions: Any | None = None
                loss_dict: dict[str, torch.Tensor] | None = None
                loss: torch.Tensor | None = None
                loss_sum: torch.Tensor | None = None
                targets: dict[str, torch.Tensor] | None = None
                x_1: torch.Tensor | None = None
                x_t: torch.Tensor | None = None
                t: torch.Tensor | None = None
                ligand_centers: torch.Tensor | None = None
                proposal_logits: torch.Tensor | None = None
                residue_prior_feat: torch.Tensor | None = None
                proposal_logits_cpu: torch.Tensor | None = None
                center_guidance_logits_cpu: torch.Tensor | None = None
                proposal_top_scores: torch.Tensor | None = None
                proposal_top_scores_cpu: torch.Tensor | None = None
                residue_pos_for_crop: torch.Tensor | None = None
                residue_batch_for_crop: torch.Tensor | None = None
                residue_prior_feat_cpu: torch.Tensor | None = None
                residue_pos_cpu: torch.Tensor | None = None
                residue_batch_cpu: torch.Tensor | None = None
                crop_centers: torch.Tensor | None = None
                crop_centers_cpu: torch.Tensor | None = None
                wrong_centers: torch.Tensor | None = None
                wrong_centers_cpu: torch.Tensor | None = None
                wrong_center_scores: torch.Tensor | None = None
                wrong_center_scores_cpu: torch.Tensor | None = None
                wrong_center_valid: torch.Tensor | None = None
                wrong_center_valid_cpu: torch.Tensor | None = None
                bootstrap_centers: torch.Tensor | None = None
                bootstrap_centers_cpu: torch.Tensor | None = None
                local_batch_samples: list[Any] | None = None

                try:
                    if use_configured_cuda:
                        torch.cuda.reset_peak_memory_stats(device=device)
                    source_batch = batch
                    if not clone_safety_checked:
                        clone_safety_checked = True
                    train_progress = epoch_progress
                    center_guidance_fraction = _resolve_center_guidance_fraction(
                        train_progress,
                    )
                    center_negative_ready = train_progress >= crop_near_miss_start
                    ligand_centers = scatter_mean(
                        source_batch["ligand_atom"].pos,
                        source_batch["ligand_atom"].batch,
                        dim=0,
                        dim_size=num_graphs,
                    )
                    proposal_logits, residue_pos_for_crop, residue_batch_for_crop, residue_prior_feat = predict_center_proposal_logits(
                        model,
                        source_batch,
                        device=device,
                    )
                    proposal_logits_cpu = proposal_logits.detach().cpu().view(-1)
                    residue_prior_feat_cpu = residue_prior_feat.detach().cpu()
                    center_guidance_logits_cpu = compute_center_guidance_scores(
                        proposal_logits_cpu,
                        residue_prior_feat_cpu,
                        learned_score_fraction=center_guidance_fraction,
                    )
                    residue_pos_cpu = residue_pos_for_crop.detach().cpu()
                    residue_batch_cpu = residue_batch_for_crop.detach().cpu()
                    wrong_centers_cpu, wrong_center_scores_cpu, wrong_center_valid_cpu = select_wrong_center_candidates(
                        ligand_centers.detach().cpu(),
                        center_guidance_logits_cpu,
                        residue_pos_cpu,
                        residue_batch_cpu,
                        positive_radius=center_positive_radius,
                        bucket_topk=crop_candidate_topk,
                        weighted_sampling=True,
                        allow_negative_centers=center_negative_ready,
                    )
                    bootstrap_centers_cpu = select_bootstrap_blind_centers(
                        ligand_centers.detach().cpu(),
                        center_guidance_logits_cpu,
                        residue_pos_cpu,
                        residue_batch_cpu,
                        positive_radius=center_positive_radius,
                        bucket_topk=crop_candidate_topk,
                        allow_negative_centers=center_negative_ready,
                    )
                    proposal_top_scores_cpu = torch.full((num_graphs,), -1e9, dtype=center_guidance_logits_cpu.dtype)
                    for graph_idx in range(num_graphs):
                        graph_mask = residue_batch_cpu == graph_idx
                        graph_logits = center_guidance_logits_cpu[graph_mask]
                        if graph_logits.numel() > 0:
                            proposal_top_scores_cpu[graph_idx] = graph_logits.max()
                    crop_centers_cpu, crop_modes = select_training_crop_centers(
                        ligand_centers.detach().cpu(),
                        center_guidance_logits_cpu,
                        residue_pos_cpu,
                        residue_batch_cpu,
                        progress=train_progress,
                        positive_radius=center_positive_radius,
                        bucket_topk=crop_candidate_topk,
                        weighted_sampling=True,
                        disable_jitter=disable_jitter_crop,
                        disable_hard_negative=disable_hard_negative_crop,
                        proposal_start=crop_proposal_start,
                        near_miss_start=crop_near_miss_start,
                        hard_negative_start=crop_hard_negative_start,
                    )
                    local_batch = build_local_batch_from_centers(
                        source_batch,
                        centers=crop_centers_cpu,
                        crop_radius=float(crop_radius),
                        crop_min_residues=crop_min_residues,
                        crop_atom_margin=crop_atom_margin,
                        graph_builder=graph_builder,
                        collator=collator,
                    )
                    local_batch_samples = (
                        local_batch.to_data_list()
                        if hasattr(local_batch, "to_data_list")
                        else [local_batch]
                    )
                    batch = local_batch.to(device)
                    crop_centers = crop_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                    wrong_center_valid = wrong_center_valid_cpu.to(device=device)
                    wrong_center_scores = wrong_center_scores_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                    wrong_centers = wrong_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                    bootstrap_centers = bootstrap_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                    proposal_top_scores = proposal_top_scores_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                    epoch_source_residues.append(
                        float(source_batch["protein_residue"].pos.size(0)) / max(1, num_graphs)
                    )
                    epoch_local_residues.append(
                        float(local_batch["protein_residue"].pos.size(0)) / max(1, num_graphs)
                    )
                    _annotate_loss_context(
                        batch,
                        current_epoch=epoch,
                        total_epochs_count=epochs,
                        warmup_epochs_count=warmup_epochs,
                        training=True,
                    )

                    with torch.no_grad():
                        x_1 = batch["ligand_atom"].pos
                        t, x_t, targets = matcher.sample_location_and_target(
                            x_1=x_1,
                            data=batch,
                            current_epoch=epoch,
                            total_epochs=epochs,
                            placement_centers=crop_centers,
                        )

                    batch["ligand_atom"].pos = x_t
                    batch.t = t

                    predictions = model(batch, t)

                    targets["binding_affinity_target"] = batch.get("y_energy", None)
                    targets["pose_rank_target"] = compute_pose_rank_target(
                        x_t,
                        x_1,
                        batch_idx=batch["ligand_atom"].batch,
                        samples=local_batch_samples,
                        dataset_raw_dir=dataset.raw_dir,
                    )

                    loss_dict = criterion(predictions, targets, batch)
                    epoch_local_losses.append(float(loss_dict["total"].detach().item()))
                    energy_nan_this_batch = int(loss_dict.get("energy_nan_skipped", torch.tensor(0.0)).item())
                    epoch_energy_nan_skips += energy_nan_this_batch
                    if energy_nan_this_batch > 0:
                        consecutive_energy_nan_skips += 1
                        if consecutive_energy_nan_skips >= ENERGY_NAN_FAILFAST_LIMIT:
                            raise RuntimeError(
                                f"Energy head produced non-finite affinity values on "
                                f"{consecutive_energy_nan_skips} consecutive batches "
                                f"(epoch={epoch+1}, batch={batch_idx})."
                            )
                    else:
                        consecutive_energy_nan_skips = 0
                    loss_pose_rank = torch.tensor(0.0, device=device)
                    same_center_monitored_loss = torch.tensor(0.0, device=device)
                    same_center_pair_count = 0
                    same_center_ranking_active = (
                        pose_ranking_pair_weight > 0.0
                        and train_progress >= ranking_same_center_start
                    )
                    wrong_center_ranking_active = (
                        same_center_ranking_active
                        and train_progress >= ranking_wrong_center_start
                    )
                    if same_center_ranking_active:
                        if not local_batch_samples:
                            loss_dict["loss_pose_rank"] = loss_pose_rank.detach()
                            loss_dict["rank_pairs_same_center"] = torch.tensor(0.0, device=device)
                            loss_dict["rank_pairs_wrong_center"] = torch.tensor(0.0, device=device)
                        else:
                            rank_terms: list[torch.Tensor] = []
                            current_rank_logit = select_pose_rank_logit(predictions)
                            allow_wrong_center_branch = (
                                wrong_center_ranking_active
                                and ranking_budget_controller.current_budget > 0
                                and bool(wrong_center_valid.any())
                            )
                            if allow_wrong_center_branch:
                                try:
                                    wrong_local_batch = build_local_batch_from_centers(
                                        source_batch,
                                        centers=wrong_centers_cpu,
                                        crop_radius=float(crop_radius),
                                        crop_min_residues=crop_min_residues,
                                        crop_atom_margin=crop_atom_margin,
                                        graph_builder=graph_builder,
                                        collator=collator,
                                    )
                                    wrong_local_batch_samples = (
                                        wrong_local_batch.to_data_list()
                                        if hasattr(wrong_local_batch, "to_data_list")
                                        else [wrong_local_batch]
                                    )
                                    wrong_local_batch = wrong_local_batch.to(device)
                                    x_1_wrong = wrong_local_batch["ligand_atom"].pos
                                    with torch.no_grad():
                                        t_wrong = torch.clamp(
                                            t * (0.65 + 0.25 * torch.rand_like(t)),
                                            min=1e-3,
                                            max=0.9,
                                        )
                                        _, x_t_wrong, _ = matcher.sample_location_and_target(
                                            x_1=x_1_wrong,
                                            data=wrong_local_batch,
                                            current_epoch=epoch,
                                            total_epochs=epochs,
                                            placement_centers=wrong_centers,
                                            t_override=t_wrong,
                                        )
                                    wrong_local_batch["ligand_atom"].pos = x_t_wrong
                                    wrong_local_batch.t = t_wrong
                                    wrong_pred = model(wrong_local_batch, t_wrong)
                                    wrong_rank_logit = select_pose_rank_logit(wrong_pred)
                                    pose_rank_wrong = compute_pose_rank_target(
                                        x_t_wrong,
                                        x_1_wrong,
                                        batch_idx=wrong_local_batch["ligand_atom"].batch,
                                        samples=wrong_local_batch_samples,
                                        dataset_raw_dir=dataset.raw_dir,
                                    )
                                    anchor_clash = predictions.get("steric_clash_batch")
                                    wrong_clash = wrong_pred.get("steric_clash_batch")
                                    if anchor_clash is None:
                                        anchor_clash = torch.zeros_like(t)
                                    if wrong_clash is None:
                                        wrong_clash = torch.zeros_like(t)
                                    low_clash_mask = wrong_center_valid & (
                                        wrong_clash.view(-1) <= (anchor_clash.view(-1) + 1.0)
                                    )
                                    loss_wrong, count_wrong = pairwise_ranking_loss_from_pairs(
                                        current_rank_logit,
                                        targets["pose_rank_target"],
                                        wrong_rank_logit,
                                        pose_rank_wrong,
                                        margin=pose_ranking_margin,
                                        extra_mask=low_clash_mask,
                                    )
                                    if count_wrong > 0:
                                        rank_terms.append(loss_wrong)
                                        epoch_rank_pair_counts["wrong_center_low_clash"] += count_wrong

                                    misleading_center_mask = low_clash_mask & (
                                        wrong_center_scores >= (proposal_top_scores - 0.25)
                                    )
                                    loss_center_hard, count_center_hard = pairwise_ranking_loss_from_pairs(
                                        current_rank_logit,
                                        targets["pose_rank_target"],
                                        wrong_rank_logit,
                                        pose_rank_wrong,
                                        margin=pose_ranking_margin,
                                        extra_mask=misleading_center_mask,
                                    )
                                    if count_center_hard > 0:
                                        rank_terms.append(loss_center_hard)
                                        epoch_rank_pair_counts["misleading_center"] += count_center_hard

                                    anchor_aff = predictions.get("binding_affinity")
                                    wrong_aff = wrong_pred.get("binding_affinity")
                                    if anchor_aff is not None and wrong_aff is not None:
                                        misleading_aff_mask = low_clash_mask & (
                                            wrong_aff.view(-1) >= (anchor_aff.view(-1) - 0.25)
                                        )
                                        loss_aff_hard, count_aff_hard = pairwise_ranking_loss_from_pairs(
                                            current_rank_logit,
                                            targets["pose_rank_target"],
                                            wrong_rank_logit,
                                            pose_rank_wrong,
                                            margin=pose_ranking_margin,
                                            extra_mask=misleading_aff_mask,
                                        )
                                        if count_aff_hard > 0:
                                            rank_terms.append(loss_aff_hard)
                                            epoch_rank_pair_counts["misleading_affinity"] += count_aff_hard
                                    del wrong_local_batch, x_1_wrong, t_wrong, x_t_wrong, wrong_pred, pose_rank_wrong
                                except torch.cuda.OutOfMemoryError:
                                    root_event_rank_oom = True
                                    epoch_rank_oom_skips += 1
                                    ranking_budget_controller.mark_offender(current_root_indices)
                                    logger.warning(
                                        "Batch %d: wrong-center ranking OOM, downgrading to same-center-only.",
                                        batch_idx,
                                    )
                                    gc.collect()
                                    if use_configured_cuda:
                                        torch.cuda.empty_cache()

                            if rank_terms:
                                loss_pose_rank = torch.stack(rank_terms).mean()
                            loss_dict["loss_pose_rank"] = loss_pose_rank.detach()
                            loss_dict["rank_pairs_same_center"] = torch.tensor(0.0, device=device)
                            loss_dict["rank_pairs_wrong_center"] = torch.tensor(
                                epoch_rank_pair_counts["wrong_center_low_clash"], device=device
                            )
                            if use_configured_cuda:
                                epoch_rank_peak_mem_mb = max(
                                    epoch_rank_peak_mem_mb,
                                    float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
                                )
                            else:
                                loss_dict["loss_pose_rank"] = loss_pose_rank.detach()
                                loss_dict["rank_pairs_same_center"] = torch.tensor(0.0, device=device)
                                loss_dict["rank_pairs_wrong_center"] = torch.tensor(0.0, device=device)

                    loss_pose_bootstrap = torch.tensor(0.0, device=device)
                    teacher_model = ema_model if ema_model is not None else model
                    if pose_bootstrap_weight > 0.0 and should_run_bootstrap(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        total_epochs=epochs,
                        frequency=pose_bootstrap_frequency,
                        start_ratio=pose_bootstrap_start,
                    ):
                        loss_pose_bootstrap = compute_bootstrap_pose_rank_loss(
                            student_model=model,
                            teacher_model=teacher_model,
                            matcher=matcher,
                            source_batch=source_batch,
                            placement_centers=bootstrap_centers,
                            epoch=epoch,
                            ode_steps=pose_bootstrap_ode_steps,
                        ode_method=ode_method,
                            graph_builder=graph_builder,
                            collator=collator,
                            crop_radius=float(crop_radius),
                            crop_min_residues=crop_min_residues,
                            crop_atom_margin=crop_atom_margin,
                            dataset_raw_dir=dataset.raw_dir,
                        )
                        loss_dict["loss_pose_bootstrap"] = loss_pose_bootstrap.detach()

                    loss_center_value = torch.tensor(0.0, device=device)
                    if (
                        center_proposal_weight > 0.0
                        and proposal_logits is not None
                        and residue_pos_for_crop is not None
                        and residue_batch_for_crop is not None
                        and ligand_centers is not None
                    ):
                        ligand_centers_device = ligand_centers.to(
                            device=residue_pos_for_crop.device,
                            dtype=residue_pos_for_crop.dtype,
                        )
                        center_value_targets = build_center_value_targets(
                            residue_pos_for_crop,
                            residue_batch_for_crop,
                            ligand_centers_device,
                            positive_radius=center_positive_radius,
                        )
                        loss_center_value = compute_center_value_loss(
                            proposal_logits.view(-1),
                            center_value_targets.to(
                                device=proposal_logits.device,
                                dtype=proposal_logits.dtype,
                            ),
                        )
                        loss_dict["_raw_loss_center_value"] = loss_center_value

                    loss_dict["loss_center_value"] = loss_center_value.detach()
                    loss_dict["weight_center_value"] = torch.tensor(center_proposal_weight, device=device)
                    loss_dict["weight_pose_rank_pair"] = torch.tensor(pose_ranking_pair_weight, device=device)
                    loss_dict["weight_pose_bootstrap"] = torch.tensor(pose_bootstrap_weight, device=device)
                    loss = (
                        loss_dict["total"]
                        + center_proposal_weight * loss_center_value
                        + pose_ranking_pair_weight * loss_pose_rank
                        + pose_bootstrap_weight * loss_pose_bootstrap
                    )

                    if loss.grad_fn is None:
                        logger.warning(f"Batch {batch_idx}: loss has no grad_fn, skipping.")
                        continue

                    if torch.isnan(loss) or loss > 200:
                        logger.warning(f"{'NaN' if torch.isnan(loss) else 'Huge'} Loss on batch {batch_idx}, skipping.")
                        for k, v in loss_dict.items():
                            logger.warning(f"  {k}: {v}")
                        _log_nonfinite_training_debug(
                            reason="nonfinite_loss" if torch.isnan(loss) else "huge_loss",
                            batch_idx=batch_idx,
                            source_batch_obj=source_batch,
                            local_batch_obj=batch,
                            loss_items=loss_dict,
                            combined_loss=loss,
                        )
                        continue

                    loss_sum = loss * num_graphs
                    loss_terms_for_debug = {
                        "base_total": loss_dict["total"] * num_graphs,
                        "base_translation": (
                            loss_dict["weight_translation"]
                            * loss_dict["_raw_loss_translation"]
                            * num_graphs
                        ),
                        "base_rotation": (
                            loss_dict["weight_rotation"]
                            * loss_dict["_raw_loss_rotation"]
                            * num_graphs
                        ),
                        "base_torsion": (
                            loss_dict["weight_torsion"] * loss_dict["_raw_loss_torsion"] * num_graphs
                        ),
                        "base_energy": (
                            loss_dict["weight_energy"] * loss_dict["_raw_loss_energy"] * num_graphs
                        ),
                        "base_clash": (
                            loss_dict["weight_clash"] * loss_dict["_raw_loss_clash"] * num_graphs
                        ),
                        "base_pose_rank_bce": (
                            loss_dict["weight_pose_rank"] * loss_dict["_raw_loss_pose_rank_bce"] * num_graphs
                        ),
                        "center_value": (
                            center_proposal_weight * loss_center_value * num_graphs
                            if center_proposal_weight > 0.0
                            else None
                        ),
                        "pose_rank_pair": (
                            pose_ranking_pair_weight * loss_pose_rank * num_graphs
                            if pose_ranking_pair_weight > 0.0
                            else None
                        ),
                        "pose_bootstrap": (
                            pose_bootstrap_weight * loss_pose_bootstrap * num_graphs
                            if pose_bootstrap_weight > 0.0
                            else None
                        ),
                    }
                    loss_sum.backward()
                    if same_center_ranking_active and local_batch_samples:
                        (
                            same_center_monitored_loss,
                            same_center_pair_count,
                            same_center_had_oom,
                            same_center_irreducible_oom,
                        ) = _run_same_center_auxiliary_step(
                            local_samples=local_batch_samples,
                            placement_centers_cpu=crop_centers_cpu,
                            root_indices=current_root_indices,
                            batch_idx=batch_idx,
                        )
                        if same_center_pair_count > 0:
                            epoch_rank_pair_counts["same_center"] += same_center_pair_count
                        if same_center_had_oom:
                            root_event_same_center_oom = True
                            root_event_same_center_irreducible_oom = (
                                root_event_same_center_irreducible_oom
                                or same_center_irreducible_oom
                            )
                            if same_center_irreducible_oom:
                                same_center_budget_controller.mark_offender(
                                    current_root_indices,
                                    severe=True,
                                )
                            epoch_rank_oom_skips += 1
                            logger.warning(
                                "Batch %d: same-center ranking OOM, falling back to smaller same-center micro-batches.",
                                batch_idx,
                            )
                        total_rank_display = loss_pose_rank.detach()
                        if same_center_pair_count > 0:
                            total_rank_display = total_rank_display + (
                                same_center_monitored_loss / same_center_pair_count
                            )
                        loss_dict["loss_pose_rank"] = total_rank_display
                        loss_dict["rank_pairs_same_center"] = torch.tensor(
                            same_center_pair_count,
                            device=device,
                        )
                        if use_configured_cuda:
                            epoch_rank_peak_mem_mb = max(
                                epoch_rank_peak_mem_mb,
                                float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
                            )

                except torch.cuda.OutOfMemoryError:
                    root_event_had_oom = True
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_graphs = 0
                    accumulated_batches = 0
                    gc.collect()
                    if use_configured_cuda:
                        torch.cuda.empty_cache()
                    split_batches = (
                        split_collated_batch(source_batch, collator=collator)
                        if source_batch is not None and split_depth < max_oom_retry_splits
                        else None
                    )
                    if split_batches is not None:
                        logger.warning(
                            f"Batch {batch_idx}: CUDA OOM, retrying with split depth {split_depth + 1}."
                        )
                        pending_batches = [
                            (split_batches[0], split_depth + 1),
                            (split_batches[1], split_depth + 1),
                        ] + pending_batches
                        continue
                    root_event_irreducible_oom = True
                    logger.warning(
                        f"Batch {batch_idx}: irreducible CUDA OOM, skipping batch "
                        f"(cost={batch_cost_units}, limit={runtime_train_cost_limit})."
                    )
                    continue

                actual_batches += 1
                accumulated_graphs += num_graphs
                accumulated_batches += 1

                is_last_in_cycle = (
                    accumulated_batches >= accumulation_steps and accumulated_graphs > 0
                )

                if is_last_in_cycle:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= accumulated_graphs

                    preclip_nonfinite = _collect_nonfinite_grad_param_names()
                    if preclip_nonfinite["count"] > 0:
                        logger.warning(
                            "Batch %d: detected non-finite gradients before clipping, skipping optimizer step.",
                            batch_idx,
                        )
                        _log_nonfinite_training_debug(
                            reason="preclip_nonfinite_grad",
                            batch_idx=batch_idx,
                            source_batch_obj=source_batch,
                            local_batch_obj=batch,
                            loss_items=loss_dict,
                            combined_loss=loss,
                            branch_probe=_probe_loss_branch_gradients(loss_terms_for_debug),
                        )
                        optimizer.zero_grad()
                        accumulated_graphs = 0
                        accumulated_batches = 0
                        train_loss_meter += loss.item()
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.warning(
                            f"Batch {batch_idx}: grad_norm={grad_norm:.4g}, skipping optimizer step."
                        )
                        _log_nonfinite_training_debug(
                            reason="nonfinite_grad_norm",
                            batch_idx=batch_idx,
                            source_batch_obj=source_batch,
                            local_batch_obj=batch,
                            loss_items=loss_dict,
                            combined_loss=loss,
                            grad_norm=grad_norm,
                            branch_probe=_probe_loss_branch_gradients(loss_terms_for_debug),
                        )
                    else:
                        optimizer.step()
                        if ema_model is None:
                            ema_model = AveragedModel(
                                model,
                                avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                            )
                        ema_model.update_parameters(model)
                        scheduler.step()

                    optimizer.zero_grad()
                    accumulated_graphs = 0
                    accumulated_batches = 0

                train_loss_meter += loss.item()
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "L_translation": f"{loss_dict.get('loss_translation', torch.tensor(0)).item():.3f}",
                        "L_rotation": f"{loss_dict.get('loss_rotation', torch.tensor(0)).item():.3f}",
                        "L_torsion": f"{loss_dict.get('loss_torsion', torch.tensor(0)).item():.3f}",
                        "L_ene": f"{loss_dict.get('loss_energy', torch.tensor(0)).item():.3f}",
                        "L_cls": f"{loss_dict.get('loss_clash', torch.tensor(0)).item():.3f}",
                        "L_rank": f"{loss_dict.get('loss_pose_rank', torch.tensor(0)).item():.3f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                del predictions, loss_dict, loss, loss_sum, targets, x_1, x_t, t, batch, source_batch
                del proposal_logits, proposal_logits_cpu, proposal_top_scores, proposal_top_scores_cpu
                del residue_pos_for_crop, residue_batch_for_crop, residue_pos_cpu, residue_batch_cpu
                del crop_centers, crop_centers_cpu, crop_modes
                del wrong_centers, wrong_centers_cpu, wrong_center_scores, wrong_center_scores_cpu
                del wrong_center_valid, wrong_center_valid_cpu, bootstrap_centers, bootstrap_centers_cpu

            train_adjustment = train_budget_controller.record_root_event(
                root_indices=root_batch_indices,
                had_oom=root_event_had_oom,
                irreducible=root_event_irreducible_oom,
            )
            if root_event_had_oom:
                epoch_root_oom_events += 1
                total_oom_batches += 1
            _log_budget_adjustment(
                train_adjustment,
                budget_label="train_cost_budget",
                rebuild_loader=True,
            )
            if pose_ranking_pair_weight > 0.0:
                same_center_adjustment = same_center_budget_controller.record_root_event(
                    root_indices=root_batch_indices,
                    had_oom=root_event_same_center_oom,
                    irreducible=root_event_same_center_irreducible_oom,
                )
                _log_budget_adjustment(
                    same_center_adjustment,
                    budget_label="same_center_micro_batch_size",
                )
                ranking_adjustment = ranking_budget_controller.record_root_event(
                    root_indices=root_batch_indices,
                    had_oom=root_event_rank_oom,
                    irreducible=False,
                )
                _log_budget_adjustment(
                    ranking_adjustment,
                    budget_label="ranking_wrong_center_cap",
                )

        if accumulated_graphs > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= accumulated_graphs

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()

                if ema_model is None:
                    ema_model = AveragedModel(
                        model,
                        avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                    )
                ema_model.update_parameters(model)
                scheduler.step()

            optimizer.zero_grad()

        pbar.close()
        gc.collect()
        if use_configured_cuda:
            torch.cuda.empty_cache()
        avg_train_loss = train_loss_meter / max(1, actual_batches)
        if epoch_local_losses:
            local_mean = float(np.mean(epoch_local_losses))
            local_std = float(np.std(epoch_local_losses))
            epoch_center_guidance_fraction = _resolve_center_guidance_fraction(
                epoch_progress,
            )
            if epoch_center_guidance_fraction <= 1e-6:
                center_guidance_label = "heuristic_prior"
            elif epoch_center_guidance_fraction >= 1.0 - 1e-6:
                center_guidance_label = "learned_center_value"
            else:
                center_guidance_label = (
                    f"blended_center_value({epoch_center_guidance_fraction:.2f})"
                )
            logger.info(
                "Local crop stats | local=%.4f±%.4f | source_res/graph=%.1f | "
                "local_res/graph=%.1f | center_guidance=%s",
                local_mean,
                local_std,
                float(np.mean(epoch_source_residues)) if epoch_source_residues else 0.0,
                float(np.mean(epoch_local_residues)) if epoch_local_residues else 0.0,
                center_guidance_label,
            )
        logger.info(
            "Ranking stats | same_center=%d | wrong_center_low_clash=%d | misleading_center=%d | "
            "misleading_affinity=%d | rank_oom_skips=%d | rank_peak_mem_mb=%.1f | "
            "same_center_micro_batch=%d | wrong_center_cap=%d",
            epoch_rank_pair_counts["same_center"],
            epoch_rank_pair_counts["wrong_center_low_clash"],
            epoch_rank_pair_counts["misleading_center"],
            epoch_rank_pair_counts["misleading_affinity"],
            epoch_rank_oom_skips,
            epoch_rank_peak_mem_mb,
            same_center_budget_controller.current_budget,
            ranking_budget_controller.current_budget,
        )
        if epoch_energy_nan_skips > 0:
            logger.warning(
                "Energy loss skipped due to non-finite affinity values on %d training batches.",
                epoch_energy_nan_skips,
            )

        if epoch_root_oom_events > 0:
            logger.warning(
                "Epoch %d: encountered %d root CUDA OOM events (total root OOM events=%d).",
                epoch + 1,
                epoch_root_oom_events,
                total_oom_batches,
            )
        if epoch_cooldown_skips > 0:
            logger.warning(
                "Epoch %d: skipped %d cooldown-isolated singleton batches.",
                epoch + 1,
                epoch_cooldown_skips,
            )

        if use_configured_cuda:
            torch.cuda.empty_cache()

        gc.collect()
        if use_configured_cuda:
            torch.cuda.empty_cache()

        validation_center_guidance_fraction = _resolve_center_guidance_fraction(
            epoch_progress,
        )
        val_plan = _resolve_validation_plan(epoch)
        val_monitor_subset = val_plan["subset"]
        is_full_val = bool(val_plan["is_full"])
        active_val_controller = (
            val_full_budget_controller
            if is_full_val
            else val_partial_budget_controller
        )
        active_val_budget = (
            active_val_controller.current_budget
        )
        val_monitor_loader = _build_eval_loader(
            val_monitor_subset,
            active_val_budget,
            phase_multiplier=float(val_plan["phase_multiplier"]),
        )
        logger.info(
            "Starting validation for epoch %d/%d | mode=%s | trigger=%s | subset=%d/%d graphs (%.2f%%) | ode=%s.",
            epoch + 1,
            epochs,
            val_plan["mode"],
            val_plan["trigger"],
            val_plan["subset_size"],
            len(val_set),
            100.0 * float(val_plan["subset_ratio"]),
            val_plan["ode_method"],
        )
        val_metrics = cast(dict[str, float], compute_validation_loss(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            criterion=criterion,
            loader=val_monitor_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_rmsd_batches=None,
            dataset=dataset,
            warmup_epochs=warmup_epochs,
            graph_builder=graph_builder,
            collator=collator,
            crop_radius=float(crop_radius),
            crop_min_residues=crop_min_residues,
            crop_atom_margin=crop_atom_margin,
            cost_guard_limit=max(1, int(active_val_budget)),
            ode_steps=val_ode_steps,
            ode_method=cast(str, val_plan["ode_method"]),
            progress_desc=cast(str, val_plan["progress_desc"]),
            num_gnn_blocks=num_gnn_blocks,
            dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
            dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
            dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
            phase_multiplier=float(val_plan["phase_multiplier"]),
            max_oom_retry_splits=max_oom_retry_splits,
            max_non_oom_failures=max_val_non_oom_failures,
            max_oom_failures=max_val_oom_failures,
            center_hit_radius=center_positive_radius,
            center_recall_topk_values=(1, 3, center_proposal_topk),
            center_nms_radius=center_nms_radius,
            learned_score_fraction=validation_center_guidance_fraction,
        ))
        logger.info(
            "Finished validation for epoch %d/%d | mode=%s | subset=%d/%d graphs.",
            epoch + 1,
            epochs,
            val_plan["mode"],
            val_plan["subset_size"],
            len(val_set),
        )
        val_metrics["val_subset_size"] = float(val_plan["subset_size"])
        val_metrics["val_subset_ratio"] = float(val_plan["subset_ratio"])
        val_metrics["val_is_full"] = 1.0 if bool(val_plan["is_full"]) else 0.0
        val_metrics["fusion_pose_weight"] = float(current_fusion_weights["pose_weight"])
        val_metrics["fusion_center_weight"] = float(current_fusion_weights["center_weight"])
        val_metrics["fusion_aff_weight"] = float(current_fusion_weights.get("aff_weight", 0.0))
        val_metrics["fusion_clash_weight"] = float(current_fusion_weights.get("clash_weight", 0.0))
        val_metrics["fusion_bias"] = float(current_fusion_weights["bias"])
        epoch_seconds = max(time.perf_counter() - epoch_start_time, 1e-6)
        runtime_benchmark_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_metrics.get("val_loss", float("nan"))),
                "mean_rmsd_final": float(val_metrics.get("mean_rmsd_final", float("inf"))),
                "single_shot_success_2a": float(val_metrics.get("single_shot_success_2a", 0.0)),
                "single_shot_success_5a": float(val_metrics.get("single_shot_success_5a", 0.0)),
                "local_center_recall@1_4a": float(val_metrics.get("local_center_recall@1_4a", 0.0)),
                "local_center_recall@3_4a": float(val_metrics.get("local_center_recall@3_4a", 0.0)),
                f"local_center_recall@{center_proposal_topk}_4a": float(
                    val_metrics.get(f"local_center_recall@{center_proposal_topk}_4a", 0.0)
                ),
                "local_center_mean_min_dist": float(
                    val_metrics.get("local_center_mean_min_dist", float("inf"))
                ),
                "train_oom_batches": int(epoch_root_oom_events),
                "val_oom_batches": int(val_metrics.get("oom_batches", 0.0)),
                "val_failed_batches": int(val_metrics.get("failed_batches", 0.0)),
                "val_non_oom_failed_batches": int(
                    val_metrics.get("non_oom_failed_batches", 0.0)
                ),
                "val_coverage": float(val_metrics.get("val_coverage", 0.0)),
                "val_selection_coverage": float(
                    val_metrics.get("val_selection_coverage", 0.0)
                ),
                "cost_guard_skips": int(epoch_cost_guard_skips),
                "epoch_seconds": float(epoch_seconds),
                "graphs_per_second": float(len(train_set) / epoch_seconds),
                "train_cost_budget": int(train_budget_controller.current_budget),
                "val_cost_budget": int(active_val_budget),
                "val_partial_cost_budget": int(val_partial_budget_controller.current_budget),
                "val_full_cost_budget": int(val_full_budget_controller.current_budget),
                "validation_mode": str(val_plan["mode"]),
                "validation_trigger": str(val_plan["trigger"]),
                "validation_ode_method": str(val_plan["ode_method"]),
            }
        )
        runtime_report_path = os.path.join(report_dir, "runtime_benchmark_history.json")
        with open(runtime_report_path, "w", encoding="utf-8") as f:
            json.dump(runtime_benchmark_history, f, ensure_ascii=False, indent=2)

        gc.collect()
        if use_configured_cuda:
            torch.cuda.empty_cache()

        avg_val_loss_scalar = float(val_metrics.get("val_loss", float("nan")))
        mean_rmsd = float(val_metrics.get("mean_rmsd_final", float("inf")))
        val_oom_batches = int(val_metrics.get("oom_batches", 0.0))
        val_adjustment = active_val_controller.record_root_event(
            root_indices=[],
            had_oom=val_oom_batches > 0,
            irreducible=False,
        )
        _log_budget_adjustment(
            val_adjustment,
            budget_label="val_cost_budget",
        )

        if not (math.isnan(avg_val_loss_scalar) or math.isinf(avg_val_loss_scalar)):
            best_val_loss = min(best_val_loss, avg_val_loss_scalar)


        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss_scalar:.4f} | "
            f"Val Mean RMSD: {mean_rmsd:.4f} | "
            f"Single-shot Success@2A: {val_metrics.get('single_shot_success_2a', 0.0):.2f} | "
            f"Single-shot Success@5A: {val_metrics.get('single_shot_success_5a', 0.0):.2f} | "
            f"Local Center@1/3/{center_proposal_topk} 4A: "
            f"{val_metrics.get('local_center_recall@1_4a', 0.0):.2f}/"
            f"{val_metrics.get('local_center_recall@3_4a', 0.0):.2f}/"
            f"{val_metrics.get(f'local_center_recall@{center_proposal_topk}_4a', 0.0):.2f} | "
            f"Local Center mean min dist: {val_metrics.get('local_center_mean_min_dist', float('inf')):.2f} A | "
            f"Val coverage: {100.0 * val_metrics.get('val_selection_coverage', 0.0):.2f}% | "
            f"Val failed batches: {int(val_metrics.get('failed_batches', 0.0))} | "
            f"Val mode: {val_plan['mode']} | "
            f"Val subset: {int(val_metrics.get('val_subset_size', 0.0))} ({100.0 * val_metrics.get('val_subset_ratio', 0.0):.2f}%) | "
            f"Root OOM events: epoch={epoch_root_oom_events}, total={total_oom_batches} | "
            f"Cost-guard skips: {epoch_cost_guard_skips} | "
            f"Cooldown skips: {epoch_cooldown_skips} | "
            f"Train budget: {train_budget_controller.current_budget}"
        )

        selection_metrics = build_selection_metrics(val_metrics)
        logger.info(
            "Checkpoint selection metrics | "
            f"Composite: {selection_metrics['composite_score']:.4f} | "
            f"Single-shot Success@2A: {selection_metrics['single_shot_success_2a']:.2f} | "
            f"Single-shot Success@5A: {selection_metrics['single_shot_success_5a']:.2f} | "
            f"Mean RMSD: {selection_metrics['mean_rmsd']:.4f} | "
            f"Val Loss: {selection_metrics['val_loss']:.4f}"
        )
        logger.info(
            "Checkpoint selection mode | mode=%s | primary=%s | value=%.4f",
            checkpoint_selection_mode,
            selected_metric_label,
            selection_metrics[selected_primary_key],
        )
        val_selection_coverage = float(
            val_metrics.get("val_selection_coverage", 0.0)
        )
        checkpoint_selection_eligible = (
            val_selection_coverage >= float(min_checkpoint_selection_coverage)
        )
        val_metrics["checkpoint_selection_eligible"] = (
            1.0 if checkpoint_selection_eligible else 0.0
        )
        if not checkpoint_selection_eligible:
            logger.warning(
                "Skipping best-checkpoint updates: validation selection coverage %.2f%% "
                "is below required %.2f%%.",
                100.0 * val_selection_coverage,
                100.0 * float(min_checkpoint_selection_coverage),
            )

        checkpoint_epoch_state = CheckpointEpochState(
            epoch_idx=epoch,
            avg_train_loss_value=avg_train_loss,
            val_metrics_obj=val_metrics,
            selection_metrics=selection_metrics,
            best_val_loss=best_val_loss,
            best_rmsd=best_rmsd,
            current_fusion_weights=current_fusion_weights,
            normalization_stats=normalization_stats,
            run_name=run_name,
            run_log_file=run_log_file,
        )
        checkpoint_model_state = CheckpointModelState(
            model=model,
            ema_model=ema_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        selection_checkpoint = compose_selection_checkpoint(
            epoch_state=checkpoint_epoch_state,
            model_state=checkpoint_model_state,
            model_config_kwargs=checkpoint_model_config_kwargs,
            training_config=training_config_snapshot,
        )

        is_warmup = epoch < warmup_epochs
        if not is_warmup and checkpoint_selection_eligible:
            if is_better_checkpoint(
                selection_metrics,
                best_selected_metrics,
                primary_key=selected_primary_key,
                primary_higher_is_better=selected_higher_is_better,
            ):
                best_selected_metrics = dict(selection_metrics)
                best_selected_updated_this_epoch = True
                torch.save(selection_checkpoint, os.path.join(save_dir, "best_selected_model.pt"))
                torch.save(selection_checkpoint, os.path.join(save_dir, "best_model.pt"))
                logger.info(
                    "Saved best selected model | mode=%s | %s=%.4f | Single-shot Success@2A=%.2f | Mean RMSD=%.4f",
                    checkpoint_selection_mode,
                    selected_metric_label,
                    selection_metrics[selected_primary_key],
                    selection_metrics["single_shot_success_2a"],
                    selection_metrics["mean_rmsd"],
                )
            if is_better_checkpoint(
                selection_metrics,
                best_composite_metrics,
                primary_key="composite_score",
                primary_higher_is_better=True,
            ):
                best_composite_metrics = dict(selection_metrics)
                torch.save(selection_checkpoint, os.path.join(save_dir, "best_composite_model.pt"))
                logger.info(
                    "Saved best composite model | "
                    f"Composite={selection_metrics['composite_score']:.4f}, "
                    f"Single-shot Success@2A={selection_metrics['single_shot_success_2a']:.2f}, "
                    f"Single-shot Success@5A={selection_metrics['single_shot_success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if is_better_checkpoint(
                selection_metrics,
                best_single_shot_success2a_metrics,
                primary_key="single_shot_success_2a",
                primary_higher_is_better=True,
            ):
                best_single_shot_success2a_metrics = dict(selection_metrics)
                torch.save(selection_checkpoint, os.path.join(save_dir, "best_single_shot_success2a_model.pt"))
                logger.info(
                    "Saved best single-shot Success@2A model | "
                    f"Single-shot Success@2A={selection_metrics['single_shot_success_2a']:.2f}, "
                    f"Single-shot Success@5A={selection_metrics['single_shot_success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if is_better_checkpoint(
                selection_metrics,
                best_rmsd_metrics,
                primary_key="mean_rmsd",
                primary_higher_is_better=False,
            ):
                best_rmsd_metrics = dict(selection_metrics)
                best_rmsd = selection_metrics["mean_rmsd"]
                selection_checkpoint["best_rmsd"] = best_rmsd
                torch.save(selection_checkpoint, os.path.join(save_dir, "best_rmsd_model.pt"))
                logger.info(f"Saved best Mean RMSD model: {best_rmsd:.4f}")

        if should_refresh_pool(
            epoch,
            refresh_every=blind_pool_refresh_every,
            min_start_epoch=blind_pool_start_epoch,
            best_updated_this_epoch=best_selected_updated_this_epoch,
            refresh_on_best_update=blind_pool_refresh_on_best_update,
        ):
            logger.info("Refreshing blind candidate pool at epoch %d ...", epoch + 1)
            pool_model = ema_model if ema_model is not None else model
            pool_loader = _build_eval_loader(
                train_set,
                configured_blind_pool_cost_budget,
                phase_multiplier=topn_phase_multiplier,
            )
            try:
                new_pool = refresh_blind_candidate_pool(
                    model=pool_model,
                    matcher=matcher,
                    loader=pool_loader,
                    device=device,
                    graph_builder=graph_builder,
                    collator=collator,
                    center_topk=center_proposal_topk,
                    refine_topk=center_refine_topk,
                    center_nms_radius=center_nms_radius,
                    stage1_pose_samples=stage1_pose_samples,
                    stage2_pose_samples=stage2_pose_samples,
                    crop_radius=float(crop_radius),
                    ode_steps=val_ode_steps,
                    ode_method=ode_method,
                    warmup_epochs=warmup_epochs,
                    center_hit_radius=center_positive_radius,
                    crop_min_residues=crop_min_residues,
                    crop_atom_margin=crop_atom_margin,
                    max_complexes=blind_pool_max_complexes,
                    fusion_weights=current_fusion_weights,
                    learned_score_fraction=_resolve_center_guidance_fraction(
                        epoch_progress,
                    ),
                    cost_guard_limit=max(
                        1,
                        int(configured_blind_pool_cost_budget * eval_cost_guard_headroom),
                    ),
                    num_gnn_blocks=num_gnn_blocks,
                    dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                    dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                    dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                    phase_multiplier=topn_phase_multiplier,
                    max_oom_retry_splits=max_oom_retry_splits,
                    pool_epoch=epoch,
                    generator_ckpt_id=f"epoch_{epoch}",
                    dataset_raw_dir=dataset.raw_dir,
                )
                if new_pool:
                    cached_blind_pool = new_pool
                    save_blind_pool(
                        new_pool, blind_pool_cache_dir, epoch=epoch,
                        meta={
                            "signature": blind_pool_signature,
                            "center_proposal_topk": center_proposal_topk,
                            "center_refine_topk": center_refine_topk,
                            "stage1_pose_samples": stage1_pose_samples,
                            "stage2_pose_samples": stage2_pose_samples,
                            "ode_steps": val_ode_steps,
                            "ode_method": ode_method,
                            "crop_radius": float(crop_radius),
                        },
                    )
                    pool_stats = get_pool_stats(cached_blind_pool)
                    logger.info("Blind pool stats: %s", pool_stats)

            except Exception as e:
                logger.warning("Blind pool refresh failed: %s", e)
            finally:
                del pool_loader
                gc.collect()
                if use_configured_cuda:
                    torch.cuda.empty_cache()

        if (
            cached_blind_pool
            and blind_pool_cache_rank_weight > 0
            and epoch_progress >= replay_start_ratio
        ):
            try:
                import random as _rng
                replay_dataset = BlindCandidateReplayDataset(
                    cached_blind_pool,
                    candidates_per_complex=max(4, blind_pool_pairs_per_complex * 2),
                    positive_rmsd_threshold=2.0,
                    hard_negative_clash_threshold=5.0,
                )
                if len(replay_dataset) > 0:
                    model.train()
                    replay_sample_size = min(16, len(replay_dataset))
                    replay_indices = _rng.sample(range(len(replay_dataset)), replay_sample_size)
                    replay_items = [replay_dataset[i] for i in replay_indices]
                    optimizer.zero_grad(set_to_none=True)

                    replay_losses = replay_and_compute_losses(
                        model=model,
                        replay_items=replay_items,
                        train_set=train_set,
                        graph_builder=graph_builder,
                        collator=collator,
                        device=device,
                        crop_radius=float(crop_radius),
                        crop_min_residues=crop_min_residues,
                        crop_atom_margin=crop_atom_margin,
                        margin=pose_ranking_margin,
                        lambda_bce=blind_pool_cache_bce_weight,
                        lambda_pair=blind_pool_cache_rank_weight,
                        lambda_list=0.5,
                        lambda_center_value=center_proposal_weight,
                        micro_batch_size=replay_budget_controller.current_budget,
                        max_candidates_per_complex=replay_max_candidates_per_complex,
                        budget_controller=replay_budget_controller,
                        backward_losses=True,
                        backward_normalizer=float(replay_sample_size),
                    )

                    replay_total = replay_losses["rerank_total"]
                    center_val_loss = replay_losses.get("center_value_loss", torch.tensor(0.0, device=device))
                    combined_replay_loss = replay_total + center_proposal_weight * center_val_loss

                    has_replay_grad = any(
                        param.grad is not None for param in model.parameters()
                    )
                    if has_replay_grad and combined_replay_loss.item() > 0:
                        replay_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        if not (torch.isnan(replay_grad_norm) or torch.isinf(replay_grad_norm)):
                            optimizer.step()
                            if ema_model is not None:
                                ema_model.update_parameters(model)
                    optimizer.zero_grad(set_to_none=True)

                    logger.info(
                        "Replay reranker | bce=%.4f | pairwise=%.4f | listwise=%.4f | "
                        "center_value=%.4f | n_pairs=%d | total=%.4f | micro_batch=%d | oom_events=%d | cooldown_skips=%d",
                        replay_losses["rerank_bce"].item(),
                        replay_losses["rerank_pairwise"].item(),
                        replay_losses["rerank_listwise"].item(),
                        center_val_loss.item(),
                        int(replay_losses["rerank_n_pairs"].item()),
                        combined_replay_loss.item(),
                        replay_budget_controller.current_budget,
                        int(replay_losses.get("replay_oom_events", torch.tensor(0.0, device=device)).item()),
                        int(replay_losses.get("replay_cooldown_skips", torch.tensor(0.0, device=device)).item()),
                    )
                    model.eval()

            except Exception as e:
                optimizer.zero_grad(set_to_none=True)
                logger.warning("Replay reranker training failed: %s\n%s", e, traceback.format_exc())

        runtime_state = TrainerRuntimeState(
            best_val_loss=best_val_loss,
            best_rmsd=best_rmsd,
            best_composite_metrics=best_composite_metrics,
            best_single_shot_success2a_metrics=best_single_shot_success2a_metrics,
            best_rmsd_metrics=best_rmsd_metrics,
            best_selected_metrics=best_selected_metrics,
            current_fusion_weights=current_fusion_weights,
            total_oom_batches=total_oom_batches,
            runtime_benchmark_history=runtime_benchmark_history,
            clone_safety_checked=clone_safety_checked,
        )
        trainer_state_snapshot = build_trainer_state_snapshot(
            runtime_state=runtime_state,
            train_budget_controller=train_budget_controller,
            val_partial_budget_controller=val_partial_budget_controller,
            val_full_budget_controller=val_full_budget_controller,
            same_center_budget_controller=same_center_budget_controller,
            ranking_budget_controller=ranking_budget_controller,
            replay_budget_controller=replay_budget_controller,
        )
        checkpoint_epoch_state = CheckpointEpochState(
            epoch_idx=epoch,
            avg_train_loss_value=avg_train_loss,
            val_metrics_obj=val_metrics,
            selection_metrics=selection_metrics,
            best_val_loss=best_val_loss,
            best_rmsd=best_rmsd,
            current_fusion_weights=current_fusion_weights,
            normalization_stats=normalization_stats,
            run_name=run_name,
            run_log_file=run_log_file,
        )
        checkpoint_model_state = CheckpointModelState(
            model=model,
            ema_model=ema_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        resume_checkpoint = compose_resume_checkpoint(
            epoch_state=checkpoint_epoch_state,
            model_state=checkpoint_model_state,
            model_config_kwargs=checkpoint_model_config_kwargs,
            training_config=training_config_snapshot,
            trainer_state=trainer_state_snapshot,
            rng_state=capture_rng_state(),
        )
        torch.save(resume_checkpoint, os.path.join(save_dir, "latest_model.pt"))
        if (epoch + 1) % 10 == 0:
            torch.save(resume_checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

        best_selected_updated_this_epoch = False

    if run_test_after_training:
        if len(test_set) == 0:
            logger.warning("Test set is empty; skipping final test evaluation.")
        else:
            gc.collect()
            if use_configured_cuda:
                torch.cuda.empty_cache()
            preferred_ckpt_paths = [
                os.path.join(save_dir, "best_selected_model.pt"),
                os.path.join(save_dir, "best_composite_model.pt"),
                os.path.join(save_dir, "best_model.pt"),
                os.path.join(save_dir, "best_rmsd_model.pt"),
            ]
            for best_ckpt_path in preferred_ckpt_paths:
                if os.path.exists(best_ckpt_path):
                    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    best_state = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
                    if best_state is not None:
                        model.load_state_dict(best_state)
                        logger.info(f"Loaded best checkpoint for final test evaluation: {best_ckpt_path}")
                        break

            topn_loader = _build_eval_loader(
                test_set,
                configured_final_topn_cost_budget,
                phase_multiplier=topn_phase_multiplier,
            )
            test_metrics: dict[str, float] = {}

            logger.info("Starting final blind Top-N test evaluation.")
            topn_eval = evaluate_topn_success(
                model=model,
                matcher=matcher,
                loader=topn_loader,
                device=device,
                graph_builder=graph_builder,
                collator=collator,
                topk_values=test_topk_values,
                center_topk=center_proposal_topk,
                refine_topk=center_refine_topk,
                center_nms_radius=center_nms_radius,
                stage1_pose_samples=stage1_pose_samples,
                stage2_pose_samples=stage2_pose_samples,
                crop_radius=float(crop_radius),
                ode_steps=val_ode_steps,
                ode_method=ode_method,
                warmup_epochs=warmup_epochs,
                cost_guard_limit=max(1, int(
                    configured_final_topn_cost_budget * eval_cost_guard_headroom
                )),
                num_gnn_blocks=num_gnn_blocks,
                dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                phase_multiplier=topn_phase_multiplier,
                max_oom_retry_splits=max_oom_retry_splits,
                center_hit_radius=center_positive_radius,
                crop_min_residues=crop_min_residues,
                crop_atom_margin=crop_atom_margin,
                fusion_weights=current_fusion_weights,
                return_candidate_records=True,
                progress_desc="Final [Test-TopN]",
                dataset_raw_dir=dataset.raw_dir,
                min_coverage=final_topn_min_coverage,
                fail_on_non_oom_error=True,
            )
            logger.info("Finished final blind Top-N test evaluation.")
            test_candidate_records = cast(list[dict[str, Any]], topn_eval.get("candidate_records", []))
            topn_metrics = summarize_blind_candidate_records(
                test_candidate_records,
                topk_values=test_topk_values,
                fusion_weights=current_fusion_weights,
            )
            topn_metrics["fusion_pose_weight"] = float(current_fusion_weights["pose_weight"])
            topn_metrics["fusion_center_weight"] = float(current_fusion_weights["center_weight"])
            topn_metrics["fusion_aff_weight"] = float(current_fusion_weights.get("aff_weight", 0.0))
            topn_metrics["fusion_clash_weight"] = float(current_fusion_weights.get("clash_weight", 0.0))
            topn_metrics["fusion_bias"] = float(current_fusion_weights["bias"])
            topn_metrics["topn_expected_complexes"] = float(topn_eval.get("topn_expected_complexes", 0.0))
            topn_metrics["topn_processed_complexes"] = float(topn_eval.get("topn_processed_complexes", 0.0))
            topn_metrics["topn_failed_complexes"] = float(topn_eval.get("topn_failed_complexes", 0.0))
            topn_metrics["topn_coverage"] = float(topn_eval.get("topn_coverage", 0.0))
            topn_metrics["topn_cost_guard_skips"] = float(topn_eval.get("topn_cost_guard_skips", 0.0))
            topn_metrics["topn_oom_batches"] = float(topn_eval.get("topn_oom_batches", 0.0))
            topn_metrics["topn_failed_batches"] = float(topn_eval.get("topn_failed_batches", 0.0))
            topn_metrics["topn_non_oom_failed_batches"] = float(
                topn_eval.get("topn_non_oom_failed_batches", 0.0)
            )
            test_metrics.update(topn_metrics)

            report_dir = os.path.join(save_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "test_metrics.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved final test report to {report_path}")
            logger.info(f"[Test Summary] {test_metrics}")
