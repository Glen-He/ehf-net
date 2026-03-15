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
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from ehfnet.data import ProteinLigandDataset
from ehfnet.data.datasets import ScaffoldSplitter
from ehfnet.data.featurizers import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.graph import GraphCollator
from ehfnet.runtime import build_dataset, build_model, resolve_interaction_profile
from ehfnet.training.batch_helpers import (
    apply_loss_context,
    build_local_batch_from_centers,
    compute_pose_quality_target,
    select_pose_ranking_logit,
)
from ehfnet.training.blind_pool import (
    BlindCandidateReplayDataset,
    build_blind_pool_signature,
    get_pool_stats,
    load_blind_pool,
    refresh_blind_candidate_pool,
    replay_and_compute_losses,
    save_blind_pool,
    should_refresh_pool,
)
from ehfnet.training.candidate_generation import generate_blind_candidates
from ehfnet.training.center_sampling import (
    compute_bootstrap_pose_quality_loss,
    select_bootstrap_blind_centers,
    select_training_crop_centers,
    select_wrong_center_candidates,
    should_run_bootstrap,
)
from ehfnet.training.checkpoint_io import (
    build_selection_metrics,
    compose_checkpoint,
    is_better_checkpoint,
    resolve_selection_rule,
)
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.inference import (
    DEFAULT_FUSION_WEIGHTS,
    calibrate_linear_fusion_weights,
    compute_center_guidance_scores,
    predict_center_proposal_logits,
    select_diverse_center_indices,
    summarize_blind_candidate_records,
)
from ehfnet.training.losses import FlowMatchingLoss
from ehfnet.training.normalization import compute_train_split_normalization_stats
from ehfnet.training.rerank_losses import pairwise_ranking_loss_from_pairs
from ehfnet.training.validation import compute_validation_loss, evaluate_topn_success

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
    dynamic_residue_cutoff: float,
    dynamic_residue_knn_k: int,
    flow_sigma_min: float,
    flow_spatial_sigma_min: float,
    flow_spatial_sigma_max: float,
    flow_fd_dt: float,
    flow_rotation_angle_min: float,
    flow_rotation_angle_max: float,
    flow_torsion_scale_min: float,
    flow_torsion_scale_max: float,
    loss_characteristic_scale: float,
    loss_weight_trans: float,
    loss_weight_rot: float,
    loss_weight_torsion: float,
    loss_weight_energy: float,
    loss_weight_clash: float,
    loss_weight_pose_quality: float,
    loss_coarse_trans: float,
    loss_coarse_rot: float,
    loss_coarse_torsion: float,
    loss_coarse_energy: float,
    loss_coarse_clash: float,
    loss_coarse_pose_quality: float,
    loss_transition_trans: float,
    loss_transition_rot: float,
    loss_transition_torsion: float,
    loss_transition_energy: float,
    loss_transition_clash: float,
    loss_transition_pose_quality: float,
    loss_refine_trans: float,
    loss_refine_rot: float,
    loss_refine_torsion: float,
    loss_refine_energy: float,
    loss_refine_clash: float,
    loss_refine_pose_quality: float,
    loss_refine_start: float,
    loss_pose_gate_epoch_start: float,
    loss_pose_gate_epoch_end: float,
    loss_pose_gate_tau_start: float,
    loss_pose_gate_tau_end: float,
    loss_pose_gate_temperature: float,
    device: str | torch.device,
    crop_radius: float,
    warmup_epochs: int,
    rmsd_check_ratio: float,
    accumulation_steps: int,
    max_nodes_per_batch: int,
    val_max_nodes_per_batch: int,
    test_max_nodes_per_batch: int,
    topn_max_nodes_per_batch: int,
    edge_budget_factor: int,
    eval_edge_guard_headroom: float,
    ema_decay: float,
    dataloader_num_workers: int,
    dataloader_pin_memory: bool,
    dataloader_persistent_workers: bool,
    split_train_frac: float,
    split_val_frac: float,
    split_test_frac: float,
    split_seed: int,
    split_cache_file: str,
    force_resplit: bool,
    ablation_mode: str,
    run_test_after_training: bool,
    test_topk_values: tuple[int, ...],
    test_pose_samples: int,
    enable_oom_adaptive_batch: bool,
    oom_reduce_threshold: int,
    oom_reduce_factor: float,
    min_max_nodes_per_batch: int,
    enable_val_oom_adaptive_batch: bool,
    val_oom_reduce_threshold: int,
    val_oom_reduce_factor: float,
    min_val_max_nodes_per_batch: int,
    oom_recover_epochs: int,
    oom_recover_factor: float,
    center_proposal_weight: float,
    center_positive_radius: float,
    center_proposal_topk: int,
    center_refine_topk: int,
    center_nms_radius: float,
    stage1_pose_samples: int,
    stage2_pose_samples: int,
    crop_candidate_topk: int,
    crop_min_residues: int,
    crop_atom_margin: float,
    disable_jitter_crop: bool,
    disable_hard_negative_crop: bool,
    pose_ranking_pair_weight: float,
    pose_ranking_margin: float,
    pose_bootstrap_weight: float,
    pose_bootstrap_frequency: int,
    pose_bootstrap_ode_steps: int,
    enable_fusion_calibration: bool,
    val_ode_steps: int,
    checkpoint_selection_mode: str,
    fusion_search_center_weights: tuple[float, ...],
    fusion_search_aff_weights: tuple[float, ...],
    fusion_search_clash_weights: tuple[float, ...],
    blind_pool_refresh_every: int,
    blind_pool_start_epoch: int,
    blind_pool_max_complexes: int,
    blind_pool_cache_bce_weight: float,
    blind_pool_cache_rank_weight: float,
    blind_pool_pairs_per_complex: int,
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
        dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
        dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。
        flow_sigma_min: 流匹配时间噪声下界。
        flow_spatial_sigma_min: 平移扰动课程的最小尺度。
        flow_spatial_sigma_max: 平移扰动课程的最大尺度。
        flow_fd_dt: 流匹配目标构造时使用的有限差分步长。
        flow_rotation_angle_min: 课程初期允许的最大旋转角。
        flow_rotation_angle_max: 课程后期允许的最大旋转角。
        flow_torsion_scale_min: 课程初期的扭转扰动缩放系数。
        flow_torsion_scale_max: 课程后期的扭转扰动缩放系数。
        loss_characteristic_scale: 平衡平移与旋转量纲的特征长度尺度。
        loss_weight_trans: 平移损失的全局权重。
        loss_weight_rot: 旋转损失的全局权重。
        loss_weight_torsion: 扭转损失的全局权重。
        loss_weight_energy: 亲和力损失的全局权重。
        loss_weight_clash: 位阻损失的全局权重。
        loss_weight_pose_quality: 构象质量损失的全局权重。
        loss_coarse_trans: 粗阶段平移损失权重。
        loss_coarse_rot: 粗阶段旋转损失权重。
        loss_coarse_torsion: 粗阶段扭转损失权重。
        loss_coarse_energy: 粗阶段亲和力损失权重。
        loss_coarse_clash: 粗阶段位阻损失权重。
        loss_coarse_pose_quality: 粗阶段构象质量损失权重。
        loss_transition_trans: 过渡阶段平移损失权重。
        loss_transition_rot: 过渡阶段旋转损失权重。
        loss_transition_torsion: 过渡阶段扭转损失权重。
        loss_transition_energy: 过渡阶段亲和力损失权重。
        loss_transition_clash: 过渡阶段位阻损失权重。
        loss_transition_pose_quality: 过渡阶段构象质量损失权重。
        loss_refine_trans: 细化阶段平移损失权重。
        loss_refine_rot: 细化阶段旋转损失权重。
        loss_refine_torsion: 细化阶段扭转损失权重。
        loss_refine_energy: 细化阶段亲和力损失权重。
        loss_refine_clash: 细化阶段位阻损失权重。
        loss_refine_pose_quality: 细化阶段构象质量损失权重。
        loss_refine_start: 进入细化阶段时对应的训练进度阈值。
        loss_pose_gate_epoch_start: 构象相关损失开始打开门控的训练进度。
        loss_pose_gate_epoch_end: 构象相关损失完全打开门控的训练进度。
        loss_pose_gate_tau_start: 构象门控在初期使用的时间阈值。
        loss_pose_gate_tau_end: 构象门控在后期使用的时间阈值。
        loss_pose_gate_temperature: 构象时间门控的温度系数。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        crop_radius: 局部裁剪半径。
        warmup_epochs: 课程学习预热轮数。
        rmsd_check_ratio: 验证阶段执行 RMSD 推演的 batch 比例。
        accumulation_steps: 梯度累积步数。
        max_nodes_per_batch: 训练阶段每个 batch 允许的最大节点预算。
        val_max_nodes_per_batch: 验证阶段每个 batch 允许的最大节点预算。
        test_max_nodes_per_batch: 测试阶段每个 batch 允许的最大节点预算。
        topn_max_nodes_per_batch: Top-N 评估阶段每个 batch 允许的最大节点预算。
        edge_budget_factor: 按节点数估计边数预算时使用的放大系数。
        eval_edge_guard_headroom: 评估阶段为边数保护预留的额外裕量。
        ema_decay: EMA 模型更新的衰减系数。
        dataloader_num_workers: DataLoader 使用的 worker 数。
        dataloader_pin_memory: 是否为 DataLoader 启用 pin_memory。
        dataloader_persistent_workers: 是否为 DataLoader 启用持久 worker。
        split_train_frac: 训练集划分比例。
        split_val_frac: 验证集划分比例。
        split_test_frac: 测试集划分比例。
        split_seed: 数据划分使用的随机种子。
        split_cache_file: 数据划分缓存文件路径。
        force_resplit: 是否忽略已有划分缓存并重新划分数据集。
        ablation_mode: 当前训练使用的消融模式名称。
        run_test_after_training: 训练结束后是否自动执行测试评估。
        test_topk_values: 测试阶段统计 Top-N 成功率时使用的 N 列表。
        test_pose_samples: 测试阶段每个复合物采样的候选构象数。
        enable_oom_adaptive_batch: 是否启用训练阶段的 OOM 自适应降批。
        oom_reduce_threshold: 单个 epoch 中触发自动降批所需的 OOM 次数阈值。
        oom_reduce_factor: 触发 OOM 后缩小节点预算的比例系数。
        min_max_nodes_per_batch: 训练阶段自动降批后的最小节点预算。
        enable_val_oom_adaptive_batch: 是否启用验证阶段独立的 OOM 自适应降批。
        val_oom_reduce_threshold: 验证阶段触发降批所需的 OOM 次数阈值。
        val_oom_reduce_factor: 验证阶段缩小节点预算的比例系数。
        min_val_max_nodes_per_batch: 验证阶段自动降批后的最小节点预算。
        oom_recover_epochs: 连续无 OOM 后尝试恢复预算所需的 epoch 数。
        oom_recover_factor: 预算恢复时使用的放大系数。
        center_proposal_weight: 中心提议损失在验证汇总中的权重。
        center_positive_radius: 中心判定为正样本时使用的距离半径。
        center_proposal_topk: 中心提议阶段保留的 Top-K 数量。
        center_refine_topk: 中心细化阶段阶段保留的 Top-K 数量。
        center_nms_radius: 中心去重时使用的最小间距半径。
        stage1_pose_samples: 第一阶段局部对接生成的候选构象数。
        stage2_pose_samples: 第二阶段精排生成的候选构象数。
        crop_candidate_topk: crop候选阶段保留的 Top-K 数量。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        disable_jitter_crop: 是否关闭jittercrop。
        disable_hard_negative_crop: 是否关闭hard负例crop。
        pose_ranking_pair_weight: 构象rankingpair相关的权重。
        pose_ranking_margin: 构象rankingmargin。
        pose_bootstrap_weight: 构象bootstrap相关的权重。
        pose_bootstrap_frequency: 构象bootstrapfrequency。
        pose_bootstrap_ode_steps: 构象bootstrapode的步数。
        enable_fusion_calibration: 是否启用融合calibration。
        val_ode_steps: valode的步数。
        checkpoint_selection_mode: checkpoint 选择规则名称。
        fusion_search_center_weights: 搜索中心融合权重时使用的候选集合。
        fusion_search_aff_weights: 搜索亲和力融合权重时使用的候选集合。
        fusion_search_clash_weights: 搜索位阻融合权重时使用的候选集合。
        blind_pool_refresh_every: blind pool 的刷新间隔。
        blind_pool_start_epoch: 允许开始刷新 blind pool 的最小训练轮次。
        blind_pool_max_complexes: 单次刷新 blind pool 时最多处理的复合物数量。
        blind_pool_cache_bce_weight: blind pool 回放中 BCE 损失的权重。
        blind_pool_cache_rank_weight: blind pool 回放中排序损失的权重。
        blind_pool_pairs_per_complex: 每个复合物在 blind pool 中采样的配对数量。
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

    from torch_geometric.loader import DynamicBatchSampler

    if accumulation_steps < 1:
        raise ValueError(f"Invalid accumulation_steps={accumulation_steps}.")

    if not (0.0 < oom_reduce_factor < 1.0):
        raise ValueError(f"Invalid oom_reduce_factor={oom_reduce_factor}.")

    if val_max_nodes_per_batch is None:
        raise ValueError("val_max_nodes_per_batch must be configured explicitly.")
    if test_max_nodes_per_batch is None:
        raise ValueError("test_max_nodes_per_batch must be configured explicitly.")
    if topn_max_nodes_per_batch is None:
        raise ValueError("topn_max_nodes_per_batch must be configured explicitly.")
    if min_val_max_nodes_per_batch is None:
        raise ValueError("min_val_max_nodes_per_batch must be configured explicitly.")

    configured_train_max_nodes_per_batch = max(1, int(max_nodes_per_batch))
    configured_val_max_nodes_per_batch = max(1, int(val_max_nodes_per_batch))
    configured_test_max_nodes_per_batch = max(1, int(test_max_nodes_per_batch))
    configured_topn_max_nodes_per_batch = max(1, int(topn_max_nodes_per_batch))
    train_edge_budget_factor = max(1, int(edge_budget_factor))
    eval_edge_guard_headroom = max(1.0, float(eval_edge_guard_headroom))


    def _annotate_loss_context(batch_obj: Any, *, current_epoch: int, total_epochs_count: int, warmup_epochs_count: int, training: bool) -> None:
        apply_loss_context(
            batch_obj,
            current_epoch=current_epoch,
            total_epochs_count=total_epochs_count,
            warmup_epochs_count=warmup_epochs_count,
            training=training,
        )


    effective_min_train_nodes_per_batch = max(1, int(min_max_nodes_per_batch))
    effective_min_val_nodes_per_batch = max(1, int(min_val_max_nodes_per_batch))

    if effective_min_train_nodes_per_batch > configured_train_max_nodes_per_batch:
        logger.warning(
            f"min_max_nodes_per_batch ({effective_min_train_nodes_per_batch}) is greater than "
            f"max_nodes_per_batch ({configured_train_max_nodes_per_batch}); clamping min to max."
        )
        effective_min_train_nodes_per_batch = configured_train_max_nodes_per_batch

    if effective_min_val_nodes_per_batch > configured_val_max_nodes_per_batch:
        logger.warning(
            f"min_val_max_nodes_per_batch ({effective_min_val_nodes_per_batch}) is greater than "
            f"val_max_nodes_per_batch ({configured_val_max_nodes_per_batch}); clamping min to val max."
        )
        effective_min_val_nodes_per_batch = configured_val_max_nodes_per_batch

    if not (0.0 < val_oom_reduce_factor < 1.0):
        raise ValueError(
            f"Invalid val_oom_reduce_factor={val_oom_reduce_factor}."
        )

    current_train_max_nodes_per_batch = configured_train_max_nodes_per_batch
    current_val_max_nodes_per_batch = configured_val_max_nodes_per_batch
    persistent_workers = bool(dataloader_persistent_workers and dataloader_num_workers > 0)

    _prev_loaders: list[DataLoader] = []

    def _build_loaders(train_max_nodes: int, val_max_nodes: int) -> tuple[DataLoader, DataLoader]:
        for old_loader in _prev_loaders:
            del old_loader
        _prev_loaders.clear()
        gc.collect()

        train_edge_budget = max(1, int(train_max_nodes * train_edge_budget_factor))
        val_edge_budget = max(1, int(val_max_nodes * train_edge_budget_factor))
        logger.info(
            f"Using DynamicBatchSampler budgets: train_max_num={train_edge_budget} (mode=edge), "
            f"val_max_num={val_edge_budget} (mode=edge)."
        )

        train_sampler = DynamicBatchSampler(
            cast(Any, train_set),
            max_num=train_edge_budget,
            mode="edge",
            shuffle=True,
        )
        train_loader_local = DataLoader(
            train_set,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=train_sampler,
        )

        val_sampler = DynamicBatchSampler(
            cast(Any, val_set),
            max_num=val_edge_budget,
            mode="edge",
            shuffle=False,
        )
        val_loader_local = DataLoader(
            val_set,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=val_sampler,
        )

        _prev_loaders.extend([train_loader_local, val_loader_local])
        return train_loader_local, val_loader_local

    def _build_eval_loader(subset: Any, max_nodes: int) -> DataLoader:
        eval_edge_budget = max(1, int(max_nodes * train_edge_budget_factor))
        eval_sampler = DynamicBatchSampler(
            cast(Any, subset),
            max_num=eval_edge_budget,
            mode="edge",
            shuffle=False,
        )
        return DataLoader(
            subset,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=eval_sampler,
        )

    train_loader, val_loader = _build_loaders(
        current_train_max_nodes_per_batch,
        current_val_max_nodes_per_batch,
    )

    try:
        total_val_batches = len(val_loader)

    except ValueError:
        total_val_batches = max(1, len(val_set) // 4)

    rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)

    if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
        rmsd_check_batches = 1

    logger.info(f"Validation Sampling: Check RMSD for {rmsd_check_batches}/{total_val_batches} batches ({rmsd_check_ratio*100:.1f}%)")
    logger.info(
        "Evaluation budgets: "
        f"test_nodes={configured_test_max_nodes_per_batch}, "
        f"test_edges={max(1, int(configured_test_max_nodes_per_batch * train_edge_budget_factor))}, "
        f"topn_nodes={configured_topn_max_nodes_per_batch}, "
        f"topn_edges={max(1, int(configured_topn_max_nodes_per_batch * train_edge_budget_factor))}."
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
        dynamic_residue_cutoff=dynamic_residue_cutoff,
        dynamic_residue_knn_k=dynamic_residue_knn_k,
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
        weight_trans=loss_weight_trans,
        weight_rot=loss_weight_rot,
        weight_torsion=loss_weight_torsion,
        weight_energy=loss_weight_energy,
        weight_clash=loss_weight_clash,
        weight_pose_quality=loss_weight_pose_quality,
        curriculum_weights={
            "coarse": {
                "trans": loss_coarse_trans,
                "rot": loss_coarse_rot,
                "torsion": loss_coarse_torsion,
                "energy": loss_coarse_energy,
                "clash": loss_coarse_clash,
                "pose_quality": loss_coarse_pose_quality,
            },
            "transition": {
                "trans": loss_transition_trans,
                "rot": loss_transition_rot,
                "torsion": loss_transition_torsion,
                "energy": loss_transition_energy,
                "clash": loss_transition_clash,
                "pose_quality": loss_transition_pose_quality,
            },
            "refine": {
                "trans": loss_refine_trans,
                "rot": loss_refine_rot,
                "torsion": loss_refine_torsion,
                "energy": loss_refine_energy,
                "clash": loss_refine_clash,
                "pose_quality": loss_refine_pose_quality,
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
    total_steps = epochs * updates_per_epoch
    warmup_steps = max(1, warmup_epochs) * updates_per_epoch
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps],
    )
    logger.info(
        f"LR scheduler initialized: total_steps={total_steps}, warmup_steps={warmup_steps}."
    )

    ema_model: AveragedModel | None = None

    best_val_loss = float("inf")
    best_rmsd = float("inf")
    best_composite_metrics: dict[str, float] | None = None
    best_success2a_metrics: dict[str, float] | None = None
    best_rmsd_metrics: dict[str, float] | None = None
    best_selected_metrics: dict[str, float] | None = None
    current_fusion_weights = dict(DEFAULT_FUSION_WEIGHTS)
    total_oom_batches = 0
    consecutive_clean_epochs = 0
    consecutive_clean_val_epochs = 0
    oom_blacklisted_pdb_ids: set[str] = set()
    oom_counts_by_pdb: dict[str, int] = {}
    OOM_BLACKLIST_THRESHOLD = 2
    clone_safety_checked = False
    selected_primary_key, selected_higher_is_better, selected_metric_label = resolve_selection_rule(checkpoint_selection_mode)
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
    best_selected_updated_this_epoch = False

    def _extract_batch_pdb_ids(batch_obj: Any) -> list[str]:
        pdb_attr = getattr(batch_obj, "pdb_id", None)

        if pdb_attr is None:
            return []

        if isinstance(pdb_attr, str):
            return [pdb_attr]

        if isinstance(pdb_attr, (list, tuple)):
            return [str(pid) for pid in pdb_attr]

        try:
            return [str(pid) for pid in list(pdb_attr)]
        except Exception:
            return [str(pdb_attr)]

    def _estimate_batch_total_edges(batch_obj: Any) -> int:
        total_edges = 0

        edge_types = getattr(batch_obj, "edge_types", None)
        if not edge_types:
            return 0

        for edge_type in edge_types:
            edge_store = batch_obj[edge_type]
            edge_index = getattr(edge_store, "edge_index", None)
            if edge_index is not None and edge_index.ndim == 2:
                total_edges += int(edge_index.size(1))

        return total_edges

    for epoch in range(epochs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        criterion.train()

        train_loss_meter = 0.0
        pbar = tqdm(total=len(train_set), desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="graphs")

        actual_batches = 0
        epoch_oom_batches = 0
        epoch_edge_guard_skips = 0
        accumulated_graphs = 0
        accumulated_batches = 0
        consecutive_oom = 0
        CIRCUIT_BREAKER_LIMIT = 10
        ENERGY_NAN_FAILFAST_LIMIT = 8
        epoch_fused = False
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
            num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
            pbar.update(num_graphs)
            batch_pdb_ids = _extract_batch_pdb_ids(batch)

            if oom_blacklisted_pdb_ids and batch_pdb_ids:
                if any(pid in oom_blacklisted_pdb_ids for pid in batch_pdb_ids):
                    blacklisted_in_batch = [pid for pid in batch_pdb_ids if pid in oom_blacklisted_pdb_ids]
                    logger.warning(
                        f"Batch {batch_idx}: skipping batch containing OOM-blacklisted samples "
                        f"({len(blacklisted_in_batch)}/{len(batch_pdb_ids)} in batch)."
                    )
                    consecutive_oom = 0
                    continue

            total_edges_cpu = _estimate_batch_total_edges(batch)
            edge_guard_limit = max(1, int(current_train_max_nodes_per_batch * train_edge_budget_factor))
            if total_edges_cpu > edge_guard_limit:
                epoch_edge_guard_skips += 1
                logger.warning(
                    f"Batch {batch_idx}: preflight skip due to edge-heavy batch "
                    f"(total_edges={total_edges_cpu} > limit={edge_guard_limit})."
                )
                consecutive_oom = 0
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

            try:
                source_batch = batch
                if not clone_safety_checked:
                    clone_safety_checked = True
                center_value_ready = bool(cached_blind_pool)
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
                train_progress = 1.0 if epochs <= 1 else epoch / max(1, epochs - 1)
                proposal_logits_cpu = proposal_logits.detach().cpu().view(-1)
                residue_prior_feat_cpu = residue_prior_feat.detach().cpu()
                center_guidance_logits_cpu = compute_center_guidance_scores(
                    proposal_logits_cpu,
                    residue_prior_feat_cpu,
                    use_learned_scores=center_value_ready,
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
                    allow_negative_centers=center_value_ready,
                )
                bootstrap_centers_cpu = select_bootstrap_blind_centers(
                    ligand_centers.detach().cpu(),
                    center_guidance_logits_cpu,
                    residue_pos_cpu,
                    residue_batch_cpu,
                    positive_radius=center_positive_radius,
                    bucket_topk=crop_candidate_topk,
                    allow_negative_centers=center_value_ready,
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
                    allow_proposal_buckets=center_value_ready,
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
                targets["pose_quality_target"] = compute_pose_quality_target(
                    x_t, x_1, batch_idx=batch["ligand_atom"].batch
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
                if pose_ranking_pair_weight > 0.0:
                    try:
                        rank_terms: list[torch.Tensor] = []
                        current_rank_logit = select_pose_ranking_logit(predictions)
                        same_center_batch = batch.clone()
                        with torch.no_grad():
                            t_same = torch.clamp(
                                t * (0.25 + 0.45 * torch.rand_like(t)),
                                min=1e-3,
                                max=1.0 - 1e-3,
                            )
                            _, x_t_same, _ = matcher.sample_location_and_target(
                                x_1=x_1,
                                data=same_center_batch,
                                current_epoch=epoch,
                                total_epochs=epochs,
                                placement_centers=crop_centers,
                                t_override=t_same,
                            )
                        same_center_batch["ligand_atom"].pos = x_t_same
                        same_center_batch.t = t_same
                        same_center_pred = model(same_center_batch, t_same)
                        pose_quality_same = compute_pose_quality_target(
                            x_t_same, x_1, batch_idx=batch["ligand_atom"].batch
                        )
                        same_center_rank_logit = select_pose_ranking_logit(same_center_pred)
                        loss_same, count_same = pairwise_ranking_loss_from_pairs(
                            current_rank_logit,
                            targets["pose_quality_target"],
                            same_center_rank_logit,
                            pose_quality_same,
                            margin=pose_ranking_margin,
                        )
                        if count_same > 0:
                            rank_terms.append(loss_same)
                            epoch_rank_pair_counts["same_center"] += count_same

                        if bool(wrong_center_valid.any()):
                            wrong_local_batch = build_local_batch_from_centers(
                                source_batch,
                                centers=wrong_centers_cpu,
                                crop_radius=float(crop_radius),
                                crop_min_residues=crop_min_residues,
                                crop_atom_margin=crop_atom_margin,
                                graph_builder=graph_builder,
                                collator=collator,
                            ).to(device)
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
                            wrong_rank_logit = select_pose_ranking_logit(wrong_pred)
                            pose_quality_wrong = compute_pose_quality_target(
                                x_t_wrong,
                                x_1_wrong,
                                batch_idx=wrong_local_batch["ligand_atom"].batch,
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
                                targets["pose_quality_target"],
                                wrong_rank_logit,
                                pose_quality_wrong,
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
                                targets["pose_quality_target"],
                                wrong_rank_logit,
                                pose_quality_wrong,
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
                                    targets["pose_quality_target"],
                                    wrong_rank_logit,
                                    pose_quality_wrong,
                                    margin=pose_ranking_margin,
                                    extra_mask=misleading_aff_mask,
                                )
                                if count_aff_hard > 0:
                                    rank_terms.append(loss_aff_hard)
                                    epoch_rank_pair_counts["misleading_affinity"] += count_aff_hard

                            del wrong_local_batch, x_1_wrong, t_wrong, x_t_wrong, wrong_pred, pose_quality_wrong

                        if rank_terms:
                            loss_pose_rank = torch.stack(rank_terms).mean()
                        loss_dict["loss_pose_rank"] = loss_pose_rank.detach()
                        loss_dict["rank_pairs_same_center"] = torch.tensor(
                            epoch_rank_pair_counts["same_center"], device=device
                        )
                        loss_dict["rank_pairs_wrong_center"] = torch.tensor(
                            epoch_rank_pair_counts["wrong_center_low_clash"], device=device
                        )
                        if torch.cuda.is_available():
                            epoch_rank_peak_mem_mb = max(
                                epoch_rank_peak_mem_mb,
                                float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
                            )
                        del same_center_batch, same_center_pred, pose_quality_same, t_same, x_t_same
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"Batch {batch_idx}: ranking forward OOM, skipping pairwise loss.")
                        loss_pose_rank = torch.tensor(0.0, device=device)
                        epoch_rank_oom_skips += 1
                        gc.collect()
                        torch.cuda.empty_cache()

                loss_pose_bootstrap = torch.tensor(0.0, device=device)
                teacher_model = ema_model if ema_model is not None else model
                if pose_bootstrap_weight > 0.0 and should_run_bootstrap(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_epochs=epochs,
                    frequency=pose_bootstrap_frequency,
                    start_ratio=0.30,
                ):
                    loss_pose_bootstrap = compute_bootstrap_pose_quality_loss(
                        student_model=model,
                        teacher_model=teacher_model,
                        matcher=matcher,
                        source_batch=source_batch,
                        placement_centers=bootstrap_centers,
                        epoch=epoch,
                        ode_steps=pose_bootstrap_ode_steps,
                        graph_builder=graph_builder,
                        collator=collator,
                        crop_radius=float(crop_radius),
                        crop_min_residues=crop_min_residues,
                        crop_atom_margin=crop_atom_margin,
                    )
                    loss_dict["loss_pose_bootstrap"] = loss_pose_bootstrap.detach()

                loss_dict["loss_center_value"] = torch.tensor(0.0, device=device)
                loss_dict["weight_center_value"] = torch.tensor(center_proposal_weight, device=device)
                loss_dict["weight_pose_rank"] = torch.tensor(pose_ranking_pair_weight, device=device)
                loss_dict["weight_pose_bootstrap"] = torch.tensor(pose_bootstrap_weight, device=device)
                loss = (
                    loss_dict["total"]
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
                    continue

                loss_sum = loss * num_graphs
                loss_sum.backward()

            except torch.cuda.OutOfMemoryError:
                epoch_oom_batches += 1
                total_oom_batches += 1
                consecutive_oom += 1

                optimizer.zero_grad(set_to_none=True)
                accumulated_graphs = 0
                batch = None
                source_batch = None
                local_batch = None
                predictions = None
                loss_dict = None
                loss = None
                loss_sum = None
                targets = None
                x_1 = None
                x_t = None
                t = None
                ligand_centers = None
                proposal_logits = None
                residue_prior_feat = None
                proposal_logits_cpu = None
                center_guidance_logits_cpu = None
                proposal_top_scores = None
                proposal_top_scores_cpu = None
                residue_pos_for_crop = None
                residue_batch_for_crop = None
                residue_prior_feat_cpu = None
                residue_pos_cpu = None
                residue_batch_cpu = None
                crop_centers = None
                crop_centers_cpu = None
                wrong_centers = None
                wrong_centers_cpu = None
                wrong_center_scores = None
                wrong_center_scores_cpu = None
                wrong_center_valid = None
                wrong_center_valid_cpu = None
                bootstrap_centers = None
                bootstrap_centers_cpu = None
                gc.collect()
                torch.cuda.empty_cache()

                if consecutive_oom >= 3:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    torch.cuda.empty_cache()

                if consecutive_oom == 5:
                    logger.warning(
                        f"Batch {batch_idx}: {consecutive_oom} consecutive OOMs, "
                        f"skipping model CPU roundtrip to avoid secondary OOM during restore."
                    )

                newly_blacklisted: list[str] = []
                if batch_pdb_ids:
                    for pid in batch_pdb_ids:
                        count = oom_counts_by_pdb.get(pid, 0) + 1
                        oom_counts_by_pdb[pid] = count
                        if count >= OOM_BLACKLIST_THRESHOLD and pid not in oom_blacklisted_pdb_ids:
                            oom_blacklisted_pdb_ids.add(pid)
                            newly_blacklisted.append(pid)

                if newly_blacklisted:
                    consecutive_oom = 0
                    logger.warning(
                        f"Batch {batch_idx}: blacklisted {len(newly_blacklisted)} repeatedly OOM samples; "
                        f"total blacklisted={len(oom_blacklisted_pdb_ids)}."
                    )

                if consecutive_oom >= CIRCUIT_BREAKER_LIMIT:
                    logger.error(
                        f"Epoch {epoch+1}: circuit breaker triggered after {consecutive_oom} "
                        f"consecutive OOMs at batch {batch_idx}. Breaking out of epoch."
                    )
                    epoch_fused = True
                    break

                if consecutive_oom <= 2:
                    logger.warning(
                        f"Batch {batch_idx}: CUDA OOM, skipping and clearing cache "
                        f"(batch_total_edges={total_edges_cpu}, edge_guard_limit={edge_guard_limit})."
                    )
                continue

            consecutive_oom = 0
            actual_batches += 1
            accumulated_graphs += num_graphs
            accumulated_batches += 1

            is_last_in_cycle = accumulated_batches >= accumulation_steps and accumulated_graphs > 0

            if is_last_in_cycle:
                for param in model.parameters():

                    if param.grad is not None:
                        param.grad /= accumulated_graphs

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Batch {batch_idx}: grad_norm={grad_norm:.4g}, skipping optimizer step.")

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
                    "L_tr": f"{loss_dict.get('loss_trans', torch.tensor(0)).item():.3f}",
                    "L_rot": f"{loss_dict.get('loss_rot', torch.tensor(0)).item():.3f}",
                    "L_tor": f"{loss_dict.get('loss_torsion', torch.tensor(0)).item():.3f}",
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        avg_train_loss = train_loss_meter / max(1, actual_batches)
        if epoch_local_losses:
            local_mean = float(np.mean(epoch_local_losses))
            local_std = float(np.std(epoch_local_losses))
            logger.info(
                "Local crop stats | local=%.4f±%.4f | source_res/graph=%.1f | "
                "local_res/graph=%.1f | center_guidance=%s",
                local_mean,
                local_std,
                float(np.mean(epoch_source_residues)) if epoch_source_residues else 0.0,
                float(np.mean(epoch_local_residues)) if epoch_local_residues else 0.0,
                "learned_center_value" if cached_blind_pool else "prior_bootstrap",
            )
        logger.info(
            "Ranking stats | same_center=%d | wrong_center_low_clash=%d | misleading_center=%d | "
            "misleading_affinity=%d | rank_oom_skips=%d | rank_peak_mem_mb=%.1f",
            epoch_rank_pair_counts["same_center"],
            epoch_rank_pair_counts["wrong_center_low_clash"],
            epoch_rank_pair_counts["misleading_center"],
            epoch_rank_pair_counts["misleading_affinity"],
            epoch_rank_oom_skips,
            epoch_rank_peak_mem_mb,
        )
        if epoch_energy_nan_skips > 0:
            logger.warning(
                "Energy loss skipped due to non-finite affinity values on %d training batches.",
                epoch_energy_nan_skips,
            )

        if epoch_oom_batches > 0:
            logger.warning(
                f"Epoch {epoch+1}: encountered {epoch_oom_batches} CUDA OOM batches "
                f"(total OOM batches={total_oom_batches})"
                + (" [circuit breaker triggered]" if epoch_fused else "")
                + "."
            )

        should_reduce = (
            enable_oom_adaptive_batch
            and (epoch_fused or epoch_oom_batches >= oom_reduce_threshold)
            and current_train_max_nodes_per_batch > effective_min_train_nodes_per_batch
        )
        if should_reduce:
            factor = min(oom_reduce_factor, 0.7) if epoch_fused else oom_reduce_factor
            reduced_max_nodes = max(
                int(current_train_max_nodes_per_batch * factor),
                int(effective_min_train_nodes_per_batch),
            )

            if reduced_max_nodes < current_train_max_nodes_per_batch:
                logger.warning(
                    f"Epoch {epoch+1}: OOM threshold reached ({epoch_oom_batches}/{oom_reduce_threshold}). "
                    f"Reducing train max_nodes_per_batch: {current_train_max_nodes_per_batch} -> {reduced_max_nodes}."
                )
                current_train_max_nodes_per_batch = reduced_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                logger.info("Rebuilt loaders with tighter node budget; keeping scheduler state continuous.")

                try:
                    total_val_batches = len(val_loader)

                except ValueError:
                    total_val_batches = max(1, len(val_set) // 4)
                rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)

                if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
                    rmsd_check_batches = 1
                logger.info(
                    f"Updated validation sampling: {rmsd_check_batches}/{total_val_batches} batches for RMSD."
                )
                consecutive_clean_epochs = 0

        if epoch_oom_batches == 0:
            consecutive_clean_epochs += 1

        else:
            consecutive_clean_epochs = 0

        if (
            enable_oom_adaptive_batch
            and consecutive_clean_epochs >= oom_recover_epochs
            and current_train_max_nodes_per_batch < configured_train_max_nodes_per_batch
        ):
            recovered_max_nodes = min(
                int(current_train_max_nodes_per_batch * oom_recover_factor),
                int(configured_train_max_nodes_per_batch),
            )
            if recovered_max_nodes > current_train_max_nodes_per_batch:
                logger.info(
                    f"Epoch {epoch+1}: {consecutive_clean_epochs} consecutive clean epochs. "
                    f"Recovering train max_nodes_per_batch: {current_train_max_nodes_per_batch} -> {recovered_max_nodes}."
                )
                current_train_max_nodes_per_batch = recovered_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                logger.info("Rebuilt loaders with recovered budget; keeping scheduler state continuous.")

                try:
                    total_val_batches = len(val_loader)

                except ValueError:
                    total_val_batches = max(1, len(val_set) // 4)
                rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)

                if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
                    rmsd_check_batches = 1

                consecutive_clean_epochs = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        blind_eval = evaluate_topn_success(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            loader=val_loader,
            device=device,
            graph_builder=graph_builder,
            collator=collator,
            topk_values=test_topk_values,
            num_pose_samples=max(test_pose_samples, max(test_topk_values)),
            center_topk=center_proposal_topk,
            refine_topk=center_refine_topk,
            center_nms_radius=center_nms_radius,
            stage1_pose_samples=stage1_pose_samples,
            stage2_pose_samples=stage2_pose_samples,
            crop_radius=float(crop_radius),
            ode_steps=val_ode_steps,
            warmup_epochs=warmup_epochs,
            edge_guard_limit=max(1, int(
                current_val_max_nodes_per_batch * train_edge_budget_factor * 1.5
            )),
            center_hit_radius=center_positive_radius,
            crop_min_residues=crop_min_residues,
            crop_atom_margin=crop_atom_margin,
            fusion_weights=current_fusion_weights,
            return_candidate_records=True,
        )
        blind_candidate_records = cast(list[dict[str, Any]], blind_eval.get("candidate_records", []))
        if enable_fusion_calibration and blind_candidate_records:
            current_fusion_weights = calibrate_linear_fusion_weights(
                blind_candidate_records,
                topk_values=test_topk_values,
                search_center_weights=fusion_search_center_weights,
                search_aff_weights=fusion_search_aff_weights,
                search_clash_weights=fusion_search_clash_weights,
            )
        blind_metrics = summarize_blind_candidate_records(
            blind_candidate_records,
            topk_values=test_topk_values,
            fusion_weights=current_fusion_weights,
        )
        blind_metrics["topn_edge_guard_skips"] = float(blind_eval.get("topn_edge_guard_skips", 0.0))
        blind_metrics["topn_pose_samples"] = float(blind_eval.get("topn_pose_samples", 0.0))
        blind_metrics["fusion_pose_weight"] = float(current_fusion_weights["pose_weight"])
        blind_metrics["fusion_center_weight"] = float(current_fusion_weights["center_weight"])
        blind_metrics["fusion_aff_weight"] = float(current_fusion_weights.get("aff_weight", 0.0))
        blind_metrics["fusion_clash_weight"] = float(current_fusion_weights.get("clash_weight", 0.0))
        blind_metrics["fusion_bias"] = float(current_fusion_weights["bias"])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_val_loss_scalar = float(blind_metrics.get("reranked_top1_mean_best_rmsd", float("nan")))
        mean_rmsd = float(blind_metrics.get("reranked_top1_mean_best_rmsd", float("inf")))
        val_metrics = dict(blind_metrics)
        val_metrics["val_loss"] = avg_val_loss_scalar
        val_metrics["mean_rmsd_final"] = mean_rmsd

        val_oom_batches = 0

        if (
            enable_val_oom_adaptive_batch
            and val_oom_batches >= val_oom_reduce_threshold
            and current_val_max_nodes_per_batch > effective_min_val_nodes_per_batch
        ):
            reduced_val_max_nodes = max(
                int(current_val_max_nodes_per_batch * val_oom_reduce_factor),
                int(effective_min_val_nodes_per_batch),
            )
            if reduced_val_max_nodes < current_val_max_nodes_per_batch:
                logger.warning(
                    f"Epoch {epoch+1}: validation OOM threshold reached ({val_oom_batches}/{val_oom_reduce_threshold}). "
                    f"Reducing val max_nodes_per_batch: {current_val_max_nodes_per_batch} -> {reduced_val_max_nodes}."
                )
                current_val_max_nodes_per_batch = reduced_val_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )

        if val_oom_batches == 0:
            consecutive_clean_val_epochs += 1
        else:
            consecutive_clean_val_epochs = 0

        if (
            enable_val_oom_adaptive_batch
            and consecutive_clean_val_epochs >= oom_recover_epochs
            and current_val_max_nodes_per_batch < configured_val_max_nodes_per_batch
        ):
            recovered_val_max_nodes = min(
                int(current_val_max_nodes_per_batch * oom_recover_factor),
                int(configured_val_max_nodes_per_batch),
            )
            if recovered_val_max_nodes > current_val_max_nodes_per_batch:
                logger.info(
                    f"Epoch {epoch+1}: {consecutive_clean_val_epochs} consecutive validation clean epochs. "
                    f"Recovering val max_nodes_per_batch: {current_val_max_nodes_per_batch} -> {recovered_val_max_nodes}."
                )
                current_val_max_nodes_per_batch = recovered_val_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                consecutive_clean_val_epochs = 0

        if not (math.isnan(avg_val_loss_scalar) or math.isinf(avg_val_loss_scalar)):
            best_val_loss = min(best_val_loss, avg_val_loss_scalar)


        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Blind TieBreak Loss: {avg_val_loss_scalar:.4f} | "
            f"Val-Blind Top1 RMSD: {mean_rmsd:.4f} | "
            f"Oracle@1<2A: {val_metrics.get('oracle_top1_success_2a', 0.0):.2f} | "
            f"Rerank@1<2A: {val_metrics.get('reranked_top1_success_2a', 0.0):.2f} | "
            f"CenterRecall@8: {val_metrics.get('center_recall@8', 0.0):.2f} | "
            f"OOM batches: epoch={epoch_oom_batches}, total={total_oom_batches} | "
            f"Edge-guard skips: {epoch_edge_guard_skips} | "
            f"OOM-blacklisted samples: {len(oom_blacklisted_pdb_ids)}"
        )

        selection_metrics = build_selection_metrics(val_metrics)
        logger.info(
            "Checkpoint selection metrics | "
            f"Composite: {selection_metrics['composite_score']:.4f} | "
            f"BlindCombo: {selection_metrics['blind_combo_score']:.4f} | "
            f"Rerank@1<2A: {selection_metrics['success_2a']:.2f} | "
            f"Rerank@5<2A: {selection_metrics['success_5a']:.2f} | "
            f"Oracle@5<2A: {selection_metrics['oracle_top5_success_2a']:.2f} | "
            f"CenterRecall@8: {selection_metrics['center_recall@8']:.2f} | "
            f"ProposalGap: {selection_metrics['proposal_gap']:.2f} | "
            f"RankingGap: {selection_metrics['ranking_gap']:.2f} | "
            f"Mean RMSD: {selection_metrics['mean_rmsd']:.4f} | "
            f"TieBreak Loss: {selection_metrics['val_loss']:.4f}"
        )
        logger.info(
            "Checkpoint selection mode | mode=%s | primary=%s | value=%.4f",
            checkpoint_selection_mode,
            selected_metric_label,
            selection_metrics[selected_primary_key],
        )

        checkpoint = compose_checkpoint(
            epoch_idx=epoch,
            avg_train_loss_value=avg_train_loss,
            val_metrics_obj=val_metrics,
            selection_metrics=selection_metrics,
            model=model,
            ema_model=ema_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            best_rmsd=best_rmsd,
            current_fusion_weights=current_fusion_weights,
            normalization_stats=normalization_stats,
            run_name=run_name,
            run_log_file=run_log_file,
            esm_dim=esm_dim,
            interaction_profile=interaction_profile,
            hidden_dim=hidden_dim,
            num_gnn_blocks=num_gnn_blocks,
            m_dim_scalar=m_dim_scalar,
            dropout_rate=dropout_rate,
            lig_atom_cont_count=lig_atom_cont_count,
            lig_mol_cont_count=lig_mol_cont_count,
            pro_atom_cont_count=pro_atom_cont_count,
            pro_res_cont_count=pro_res_cont_count,
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
            prediction_max_neighbors=prediction_max_neighbors,
            prediction_min_max_neighbors=prediction_min_max_neighbors,
            prediction_knn_fallback_k=prediction_knn_fallback_k,
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

        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pt"))

        is_warmup = epoch < warmup_epochs
        if not is_warmup:
            if is_better_checkpoint(
                selection_metrics,
                best_selected_metrics,
                primary_key=selected_primary_key,
                primary_higher_is_better=selected_higher_is_better,
            ):
                best_selected_metrics = dict(selection_metrics)
                best_selected_updated_this_epoch = True
                torch.save(checkpoint, os.path.join(save_dir, "best_selected_model.pt"))
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                logger.info(
                    "Saved best selected model | mode=%s | %s=%.4f | Rerank@1<2A=%.2f | Oracle@5<2A=%.2f",
                    checkpoint_selection_mode,
                    selected_metric_label,
                    selection_metrics[selected_primary_key],
                    selection_metrics["success_2a"],
                    selection_metrics["oracle_top5_success_2a"],
                )
            if is_better_checkpoint(
                selection_metrics,
                best_composite_metrics,
                primary_key="composite_score",
                primary_higher_is_better=True,
            ):
                best_composite_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_composite_model.pt"))
                logger.info(
                    "Saved best composite model | "
                    f"Composite={selection_metrics['composite_score']:.4f}, "
                    f"Rerank@1<2A={selection_metrics['success_2a']:.2f}, "
                    f"Rerank@5<2A={selection_metrics['success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if is_better_checkpoint(
                selection_metrics,
                best_success2a_metrics,
                primary_key="success_2a",
                primary_higher_is_better=True,
            ):
                best_success2a_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_success2a_model.pt"))
                logger.info(
                    "Saved best Success@2A model | "
                    f"Rerank@1<2A={selection_metrics['success_2a']:.2f}, "
                    f"Rerank@5<2A={selection_metrics['success_5a']:.2f}, "
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
                checkpoint["best_rmsd"] = best_rmsd
                torch.save(checkpoint, os.path.join(save_dir, "best_rmsd_model.pt"))
                logger.info(f"Saved best Mean RMSD model: {best_rmsd:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

        if should_refresh_pool(
            epoch,
            refresh_every=blind_pool_refresh_every,
            min_start_epoch=blind_pool_start_epoch,
            best_updated_this_epoch=best_selected_updated_this_epoch,
        ):
            logger.info("Refreshing blind candidate pool at epoch %d ...", epoch + 1)
            pool_model = ema_model if ema_model is not None else model
            pool_loader = _build_eval_loader(train_set, configured_topn_max_nodes_per_batch)
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
                    warmup_epochs=warmup_epochs,
                    center_hit_radius=center_positive_radius,
                    crop_min_residues=crop_min_residues,
                    crop_atom_margin=crop_atom_margin,
                    max_complexes=blind_pool_max_complexes,
                    fusion_weights=current_fusion_weights,
                    use_learned_center_scores=bool(cached_blind_pool),
                    pool_epoch=epoch,
                    generator_ckpt_id=f"epoch_{epoch}",
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if cached_blind_pool and blind_pool_cache_rank_weight > 0:
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
                        use_pose_rank_head=True,
                    )

                    replay_total = replay_losses["rerank_total"]
                    center_val_loss = replay_losses.get("center_value_loss", torch.tensor(0.0, device=device))
                    combined_replay_loss = replay_total + center_proposal_weight * center_val_loss

                    if combined_replay_loss.requires_grad and combined_replay_loss.item() > 0:
                        optimizer.zero_grad()
                        combined_replay_loss.backward()
                        replay_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        if not (torch.isnan(replay_grad_norm) or torch.isinf(replay_grad_norm)):
                            optimizer.step()
                            if ema_model is not None:
                                ema_model.update_parameters(model)
                        optimizer.zero_grad()

                    logger.info(
                        "Replay reranker | bce=%.4f | pairwise=%.4f | listwise=%.4f | "
                        "center_value=%.4f | n_pairs=%d | total=%.4f",
                        replay_losses["rerank_bce"].item(),
                        replay_losses["rerank_pairwise"].item(),
                        replay_losses["rerank_listwise"].item(),
                        center_val_loss.item(),
                        int(replay_losses["rerank_n_pairs"].item()),
                        combined_replay_loss.item(),
                    )
                    model.eval()

            except Exception as e:
                logger.warning("Replay reranker training failed: %s\n%s", e, traceback.format_exc())

        best_selected_updated_this_epoch = False

    if run_test_after_training:
        if len(test_set) == 0:
            logger.warning("Test set is empty; skipping final test evaluation.")
        else:
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

            test_loader = _build_eval_loader(test_set, configured_test_max_nodes_per_batch)
            topn_loader = _build_eval_loader(test_set, configured_topn_max_nodes_per_batch)
            test_metrics: dict[str, float] = {}

            topn_eval = evaluate_topn_success(
                model=model,
                matcher=matcher,
                loader=topn_loader,
                device=device,
                graph_builder=graph_builder,
                collator=collator,
                topk_values=test_topk_values,
                num_pose_samples=max(test_pose_samples, max(test_topk_values)),
                center_topk=center_proposal_topk,
                refine_topk=center_refine_topk,
                center_nms_radius=center_nms_radius,
                stage1_pose_samples=stage1_pose_samples,
                stage2_pose_samples=stage2_pose_samples,
                crop_radius=float(crop_radius),
                ode_steps=val_ode_steps,
                warmup_epochs=warmup_epochs,
                edge_guard_limit=max(1, int(
                    configured_topn_max_nodes_per_batch * train_edge_budget_factor * eval_edge_guard_headroom
                )),
                center_hit_radius=center_positive_radius,
                crop_min_residues=crop_min_residues,
                crop_atom_margin=crop_atom_margin,
                fusion_weights=current_fusion_weights,
                return_candidate_records=True,
            )
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
            topn_metrics["topn_edge_guard_skips"] = float(topn_eval.get("topn_edge_guard_skips", 0.0))
            topn_metrics["topn_pose_samples"] = float(topn_eval.get("topn_pose_samples", 0.0))
            topn_metrics["val_loss"] = float(topn_metrics.get("reranked_top1_mean_best_rmsd", float("nan")))
            test_metrics.update(topn_metrics)

            report_dir = os.path.join(save_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "test_metrics.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved final test report to {report_path}")
            logger.info(f"[Test Summary] {test_metrics}")
