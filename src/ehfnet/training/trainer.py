"""
训练循环

提供 EHFNet 模型的训练和验证功能。
"""

import os
import math
import json
import traceback
import torch
import logging
import gc
import numpy as np
from typing import Any, cast
from pathlib import Path
from scipy import stats as scipy_stats
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch_scatter import scatter_mean

from ehfnet.models import EHFNet
from ehfnet.graph import GraphCollator
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.datasets.splitter import ScaffoldSplitter
from ehfnet.training.losses import FlowMatchingLoss
from ehfnet.training.flow_matcher import ConditionalFlowMatcher


logger = logging.getLogger(__name__)


def apply_loss_context(
    batch_obj: Any,
    *,
    current_epoch: int,
    total_epochs_count: int,
    warmup_epochs_count: int,
    training: bool,
) -> None:
    progress = 1.0 if total_epochs_count <= 1 else current_epoch / max(1, total_epochs_count - 1)
    warmup_end = min(1.0, warmup_epochs_count / max(1, total_epochs_count))
    batch_obj.loss_progress = float(max(0.0, min(1.0, progress)))
    batch_obj.loss_warmup_end = float(max(0.0, min(1.0, warmup_end)))
    batch_obj.loss_is_training = bool(training)


def train(
    *,
    data_root: str,
    index_file: str,
    save_dir: str = "./checkpoints",
    esm_path: str | None = None,
    epochs: int = 100,

    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    clip_grad: float = 10.0,
    hidden_dim: int = 128,
    num_gnn_blocks: int = 6,
    lig_atom_cont_count: int = 9,
    lig_mol_cont_count: int = 9,
    pro_atom_cont_count: int = 5,
    pro_res_cont_count: int = 974,     # 14 (torsion) + 960 (ESM)
    esm_dim: int = 960,
    device: str | torch.device = "auto",
    pocket_radius: float | None = 20.0,
    normalization_stats: dict | None = None,
    warmup_epochs: int = 20,
    rmsd_check_ratio: float = 0.2,
    accumulation_steps: int = 1,
    max_nodes_per_batch: int = 10000,
    val_max_nodes_per_batch: int | None = None,
    test_max_nodes_per_batch: int | None = None,
    topn_max_nodes_per_batch: int | None = None,
    ema_decay: float = 0.999,
    dataloader_num_workers: int = 4,
    dataloader_pin_memory: bool = True,
    dataloader_persistent_workers: bool = True,
    split_train_frac: float = 0.7,
    split_val_frac: float = 0.1,
    split_test_frac: float = 0.2,
    split_seed: int = 42,
    split_cache_file: str | None = None,
    force_resplit: bool = False,
    ablation_mode: str = "none",
    run_test_after_training: bool = True,
    test_topk_values: tuple[int, ...] = (1, 5, 10),
    test_pose_samples: int = 10,
    enable_oom_adaptive_batch: bool = True,
    oom_reduce_threshold: int = 3,
    oom_reduce_factor: float = 0.85,
    min_max_nodes_per_batch: int = 12000,
    enable_val_oom_adaptive_batch: bool = True,
    val_oom_reduce_threshold: int = 3,
    val_oom_reduce_factor: float = 0.85,
    min_val_max_nodes_per_batch: int | None = None,
    oom_recover_epochs: int = 3,
    oom_recover_factor: float = 1.1,
    run_name: str | None = None,
    run_log_file: str | None = None,
):
    """
    训练 EHFNet 模型

    Args:
        data_root: PDBBind 数据根目录
        index_file: 索引 CSV 文件路径
        save_dir: 模型保存目录
        esm_path: 预计算的 ESM 嵌入路径
        epochs: 训练轮数

        lr: 学习率
        weight_decay: 权重衰减
        clip_grad: 梯度裁剪阈值
        hidden_dim: 隐藏层维度
        num_gnn_blocks: GNN 块数量
        lig_atom_cont_count: 配体原子连续特征数量
        lig_mol_cont_count: 配体分子连续特征数量
        pro_atom_cont_count: 蛋白原子连续特征数量
        pro_res_cont_count: 蛋白残基连续特征数量
        esm_dim: ESM embedding 维度
        device: 训练设备 ("cpu", "cuda", "cuda:0", "cuda:1" 等)，默认为 "auto" (自动检测)
        pocket_radius: 口袋提取半径 (Å)
        normalization_stats: 归一化统计数据
        warmup_epochs: 空间课程学习预热轮数
        rmsd_check_ratio: 验证集中计算 RMSD 的样本比例 (0.0 ~ 1.0)
                          例如 0.1 表示随机抽取 10% 的 batch 进行耗时的 RMSD 推演
        accumulation_steps: 梯度累积步数。当显存较小时，可设为 2/4 模拟更大 batch_size
        max_nodes_per_batch: 训练集 DynamicBatchSampler 节点预算
        val_max_nodes_per_batch: 验证集节点预算（None 时使用 min(train_budget, 6000)）
        test_max_nodes_per_batch: 测试集节点预算（None 时沿用验证预算）
        topn_max_nodes_per_batch: Top-N 评估节点预算（None 时沿用测试预算）
        ema_decay: EMA 衰减率，默认 0.999；小规模试跑可设为 0.99 加快吸收
        dataloader_num_workers: DataLoader worker 数
        dataloader_pin_memory: 是否启用 pin_memory
        dataloader_persistent_workers: 是否启用 persistent_workers（仅 num_workers>0 时有效）
        split_train_frac: 训练集比例（建议 0.7）
        split_val_frac: 验证集比例（建议 0.1）
        split_test_frac: 测试集比例（建议 0.2）
        split_seed: Scaffold 划分随机种子
        split_cache_file: 划分缓存 JSON 路径；为 None 时自动生成默认路径
        force_resplit: 是否忽略缓存并强制重新划分
        ablation_mode: 消融模式（"none" 或 "inter_multiscale_off"）
        run_test_after_training: 训练结束后是否自动在 test 集上评估并输出报告
        test_topk_values: Top-N 成功率统计的 N 列表，如 (1,5,10)
        test_pose_samples: 每个复合物采样的候选 pose 数（应 >= max(test_topk_values)）
        enable_oom_adaptive_batch: 是否启用 OOM 触发的自动降批保护
        oom_reduce_threshold: 单个 epoch 触发多少次 OOM 后，下个 epoch 自动降低 batch 节点预算
        oom_reduce_factor: 自动降批系数，(0,1) 内有效（级联熔断时自动使用 min(factor, 0.7) 更激进的缩减）
        min_max_nodes_per_batch: 自动降批的下限，避免降得过小影响训练质量
        enable_val_oom_adaptive_batch: 是否启用验证阶段 OOM 触发的独立降批
        val_oom_reduce_threshold: 单个 epoch 验证 OOM 达到该阈值后，下轮降低验证预算
        val_oom_reduce_factor: 验证预算缩减系数，(0,1) 内有效
        min_val_max_nodes_per_batch: 验证自动降批下限，None 时默认与训练下限一致
        oom_recover_epochs: 连续多少个无 OOM epoch 后尝试回升 batch 预算
        oom_recover_factor: 回升系数 (>1)，如 1.1 表示每次回升 10%
    """

    # 1. 准备环境
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
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

    # 2. 准备数据
    logger.info("Initializing Dataset...")
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])

    dataset = PDBBindDataset(
        root=data_root,
        index_file=index_file,
        esm_root=esm_path,
        esm="auto",
        esm_dim=esm_dim,
        pocket_radius=pocket_radius,
        interaction_profile="atom_only" if ablation_mode == "inter_multiscale_off" else "full",
    )

    # 统一亲和力统计来源：以当前 Dataset 统计为准，避免外部 stats 与训练集不一致
    if normalization_stats is None:
        normalization_stats = {}

    normalization_stats["affinity"] = {
        "mean": torch.tensor(dataset.affinity_stats["mean"], dtype=torch.float32),
        "std": torch.tensor(dataset.affinity_stats["std"], dtype=torch.float32),
    }

    if not math.isclose(split_train_frac + split_val_frac + split_test_frac, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Split fractions must sum to 1.0, got "
            f"{split_train_frac + split_val_frac + split_test_frac:.6f}."
        )

    if split_cache_file is None:
        split_dir = Path(data_root) / "splits"
        split_name = (
            f"scaffold_{int(split_train_frac*100)}_"
            f"{int(split_val_frac*100)}_{int(split_test_frac*100)}_seed{split_seed}.json"
        )
        split_cache_file = str(split_dir / split_name)

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

    logger.info(
        f"Final Dataset Sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}"
    )

    from torch_geometric.loader import DynamicBatchSampler

    if accumulation_steps < 1:
        logger.warning(f"Invalid accumulation_steps={accumulation_steps}, fallback to 1.")
        accumulation_steps = 1

    if not (0.0 < oom_reduce_factor < 1.0):
        logger.warning(f"Invalid oom_reduce_factor={oom_reduce_factor}, fallback to 0.8.")
        oom_reduce_factor = 0.8

    configured_train_max_nodes_per_batch = max(1, int(max_nodes_per_batch))
    configured_val_max_nodes_per_batch = (
        max(1, int(val_max_nodes_per_batch))
        if val_max_nodes_per_batch is not None
        else min(configured_train_max_nodes_per_batch, 6000)
    )
    configured_test_max_nodes_per_batch = (
        max(1, int(test_max_nodes_per_batch))
        if test_max_nodes_per_batch is not None
        else configured_val_max_nodes_per_batch
    )
    configured_topn_max_nodes_per_batch = (
        max(1, int(topn_max_nodes_per_batch))
        if topn_max_nodes_per_batch is not None
        else configured_test_max_nodes_per_batch
    )
    # [修复] 降低边预算因子
    # 旧值 60 + protein_atom k=128 bug 导致 batch 静态边已接近显存上限，
    # 而编码器 forward 中还会通过 radius() 动态创建跨图边（对 Sampler 不可见），
    # 使实际 GPU 边数远超预算 → OOM。
    # 修复 k→32 后静态边/图大幅下降，Sampler 会装入更多图，
    # 必须同步下调因子为动态边预留 ~40-50% 显存 headroom。
    # 注意：batch 变小不影响精度——accumulation_steps=8 保证了有效梯度规模。
    train_edge_budget_factor = 40
    eval_edge_guard_headroom = 1.5

    def _safe_metric(value: Any, default: float, *, higher_is_better: bool = True) -> float:
        try:
            metric = float(value)
        except Exception:
            return default

        if math.isnan(metric) or math.isinf(metric):
            return default

        return metric

    def _build_selection_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        success_2a = _safe_metric(metrics.get("success_2a"), 0.0)
        success_5a = _safe_metric(metrics.get("success_5a"), 0.0)
        mean_rmsd = _safe_metric(metrics.get("mean_rmsd_final"), 1e9, higher_is_better=False)
        val_loss = _safe_metric(metrics.get("val_loss"), 1e9, higher_is_better=False)
        composite_score = 1.5 * success_2a + 0.15 * success_5a - 0.8 * mean_rmsd

        return {
            "composite_score": composite_score,
            "success_2a": success_2a,
            "success_5a": success_5a,
            "mean_rmsd": mean_rmsd,
            "val_loss": val_loss,
        }

    def _is_better_checkpoint(
        candidate: dict[str, float],
        incumbent: dict[str, float] | None,
        *,
        primary_key: str,
        primary_higher_is_better: bool,
        tol: float = 1e-6,
    ) -> bool:
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

    def _annotate_loss_context(batch_obj: Any, *, current_epoch: int, total_epochs_count: int, warmup_epochs_count: int, training: bool) -> None:
        apply_loss_context(
            batch_obj,
            current_epoch=current_epoch,
            total_epochs_count=total_epochs_count,
            warmup_epochs_count=warmup_epochs_count,
            training=training,
        )

    def _compose_checkpoint(
        *,
        epoch_idx: int,
        avg_train_loss_value: float,
        val_metrics_obj: dict[str, Any],
        selection_metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch_idx,
            "run_name": run_name,
            "run_log_file": run_log_file,
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
        }

    effective_min_train_nodes_per_batch = max(1, int(min_max_nodes_per_batch))
    effective_min_val_nodes_per_batch = max(
        1,
        int(min_val_max_nodes_per_batch)
        if min_val_max_nodes_per_batch is not None
        else int(min_max_nodes_per_batch),
    )

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
        logger.warning(f"Invalid val_oom_reduce_factor={val_oom_reduce_factor}, fallback to 0.8.")
        val_oom_reduce_factor = 0.8

    current_train_max_nodes_per_batch = configured_train_max_nodes_per_batch
    current_val_max_nodes_per_batch = configured_val_max_nodes_per_batch
    persistent_workers = bool(dataloader_persistent_workers and dataloader_num_workers > 0)

    # 用于追踪旧 loader 引用，确保重建时显式销毁 persistent_workers
    _prev_loaders: list[DataLoader] = []

    def _build_loaders(train_max_nodes: int, val_max_nodes: int) -> tuple[DataLoader, DataLoader]:
        # 显式销毁旧 loader，释放 persistent_workers 进程
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

        # [修复] 验证集也使用 edge 模式进行批处理
        # 旧逻辑使用 mode="node"，导致节点数受控但边数完全不可控，
        # 几乎 100% 的验证 batch 都因超出 edge_guard_limit 而被 preflight skip
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

    # [新增逻辑] 动态计算 Batch 数量 (DynamicBatchSampler 没有固定的 len)
    try:
        total_val_batches = len(val_loader)

    except ValueError:
        total_val_batches = max(1, len(val_set) // 4)
        
    rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)
    
    # 确保至少检查 1 个 batch (如果 ratio > 0)
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

    # 3. 准备模型组件
    logger.info("Initializing Model & Flow Components...")

    model = EHFNet(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        normalization_stats=normalization_stats,
    ).to(device)

    matcher = ConditionalFlowMatcher(
        sigma_min=1e-3,
        warmup_epochs=warmup_epochs,
    )
    criterion = FlowMatchingLoss().to(device)
    # 速度分解由 matcher 内部完成，trainer 不持有分解器

    # 4. 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Warmup + 余弦退火（Step 级），防止初期梯度冲击 + 中后期平滑收敛
    try:
        total_train_batches = len(train_loader)
    except ValueError:
        total_train_batches = max(1, len(train_set) // 4)
    updates_per_epoch = math.ceil(total_train_batches / accumulation_steps)
    total_steps = epochs * updates_per_epoch
    warmup_steps = max(1, warmup_epochs) * updates_per_epoch  # LR warmup 与 Curriculum 同步
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

    # 5. 训练循环
    best_val_loss = float("inf")
    best_rmsd = float("inf")
    best_composite_metrics: dict[str, float] | None = None
    best_success2a_metrics: dict[str, float] | None = None
    best_rmsd_metrics: dict[str, float] | None = None
    total_oom_batches = 0
    consecutive_clean_epochs = 0  # 连续无 OOM 的 epoch 计数，用于回升决策
    consecutive_clean_val_epochs = 0
    oom_blacklisted_pdb_ids: set[str] = set()
    oom_counts_by_pdb: dict[str, int] = {}
    OOM_BLACKLIST_THRESHOLD = 2

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
        accumulated_graphs = 0  # 当前累积周期内的总图数
        consecutive_oom = 0     # 连续 OOM 计数，用于级联熔断
        CIRCUIT_BREAKER_LIMIT = 10  # 连续 OOM 达到此值则熔断当前 epoch
        epoch_fused = False     # 本 epoch 是否被熔断
        optimizer.zero_grad()   # 在循环外初始化梯度清零

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
            
            # 【全覆盖 OOM 安全网】
            # 包裹 batch.to(device) → 采样 → 前向 → 损失 → 反向 的完整流程
            # OOM 可能发生在任何 GPU 操作点：数据迁移、EGNN 消息传递、梯度计算
            try:
                batch = batch.to(device)
                _annotate_loss_context(
                    batch,
                    current_epoch=epoch,
                    total_epochs_count=epochs,
                    warmup_epochs_count=warmup_epochs,
                    training=True,
                )

                # 流匹配训练步骤
                # 生成训练目标不需要梯度
                with torch.no_grad():
                    x_1 = batch["ligand_atom"].pos
                    t, x_t, targets = matcher.sample_location_and_target(
                        x_1=x_1,
                        data=batch,
                        current_epoch=epoch,
                        total_epochs=epochs,
                    )

                batch["ligand_atom"].pos = x_t
                batch.t = t

                # FP32 前向传播
                predictions = model(batch, t)

                # 补充结合能 target
                targets["binding_affinity_target"] = batch.get("y_energy", None)

                loss_dict = criterion(predictions, targets, batch)
                loss = loss_dict["total"]

                # 防御性检查
                if loss.grad_fn is None:
                    logger.warning(f"Batch {batch_idx}: loss has no grad_fn, skipping.")
                    continue

                if torch.isnan(loss) or loss > 200:
                    logger.warning(f"{'NaN' if torch.isnan(loss) else 'Huge'} Loss on batch {batch_idx}, skipping.")
                    for k, v in loss_dict.items():
                        logger.warning(f"  {k}: {v}")
                    continue

                # 样本级梯度累积
                loss_sum = loss * num_graphs
                loss_sum.backward()

            except torch.cuda.OutOfMemoryError:
                epoch_oom_batches += 1
                total_oom_batches += 1
                consecutive_oom += 1

                # 【分级 CUDA 恢复】
                # Level 1: 基础清理（首次 OOM）
                optimizer.zero_grad(set_to_none=True)
                accumulated_graphs = 0
                # 必须删除所有引用 GPU tensor 的局部变量，否则碎片无法回收
                for _var in ('batch', 'predictions', 'loss_dict', 'loss', 'loss_sum', 'targets', 'x_1', 'x_t', 't'):
                    if _var in locals():
                        try:
                            del locals()[_var]
                        except Exception:
                            pass
                gc.collect()
                torch.cuda.empty_cache()

                # Level 2: 深度恢复（连续 OOM >= 3）
                if consecutive_oom >= 3:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    torch.cuda.empty_cache()

                # Level 3: 模型参数 CPU 往返（连续 OOM >= 5）—— 彻底重组显存布局
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

                # 【级联熔断器】连续 OOM 达到阈值，立即退出当前 epoch
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

            # OOM 恢复：成功处理一个 batch 后重置连续 OOM 计数
            consecutive_oom = 0
            actual_batches += 1
            accumulated_graphs += num_graphs

            # 仅在完成一个完整累积周期后才更新参数
            is_last_in_cycle = (batch_idx + 1) % accumulation_steps == 0

            if is_last_in_cycle and accumulated_graphs > 0:
                
                # 将累积梯度除以真实图总数，得到无偏的样本级平均梯度
                for param in model.parameters():

                    if param.grad is not None:
                        param.grad /= accumulated_graphs

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                # Inf/NaN 梯度直接跳过，防止权重被污染

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Batch {batch_idx}: grad_norm={grad_norm:.4g}, skipping optimizer step.")

                else:
                    optimizer.step()
                    # 惰性构建 EMA（首次 step 后 lazy 参数已全部初始化）
                    if ema_model is None:
                        ema_model = AveragedModel(
                            model,
                            avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                        )
                    ema_model.update_parameters(model)
                    scheduler.step()

                # 清零梯度和计数器，开始新的累积周期
                optimizer.zero_grad()
                accumulated_graphs = 0

            # 记录日志
            train_loss_meter += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L_tr": f"{loss_dict.get('loss_trans', torch.tensor(0)).item():.3f}",
                    "L_rot": f"{loss_dict.get('loss_rot', torch.tensor(0)).item():.3f}",
                    "L_tor": f"{loss_dict.get('loss_torsion', torch.tensor(0)).item():.3f}",
                    "L_ene": f"{loss_dict.get('loss_energy', torch.tensor(0)).item():.3f}",
                    "L_cls": f"{loss_dict.get('loss_clash', torch.tensor(0)).item():.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            del predictions, loss_dict, loss, loss_sum, targets, x_1, x_t, t, batch

        # 循环结束，如果有剩余积累的梯度，进行最后一次 step
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

        if epoch_oom_batches > 0:
            logger.warning(
                f"Epoch {epoch+1}: encountered {epoch_oom_batches} CUDA OOM batches "
                f"(total OOM batches={total_oom_batches})"
                + (" [circuit breaker triggered]" if epoch_fused else "")
                + "."
            )

        # 【自适应降批】熔断 epoch 或 OOM 超阈值时降低节点预算
        # 熔断意味着级联发生，必须立即响应
        should_reduce = (
            enable_oom_adaptive_batch
            and (epoch_fused or epoch_oom_batches >= oom_reduce_threshold)
            and current_train_max_nodes_per_batch > effective_min_train_nodes_per_batch
        )
        if should_reduce:
            # 熔断时使用更激进的缩减（0.7x），普通 OOM 使用标准缩减
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
                consecutive_clean_epochs = 0  # 降批后重置回升计数

        # 【回升机制】连续无 OOM 时逐步恢复 batch 预算
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

                consecutive_clean_epochs = 0  # 回升后重置计数

        # 验证
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [新增] 训练结束，验证开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_val_loss = compute_validation_loss(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_rmsd_batches=rmsd_check_batches,
            dataset=dataset,
            warmup_epochs=warmup_epochs,
            # [修复] edge_guard 应基于实际的 val_edge_budget 并预留动态边余量
            # 旧算法用 val_max_nodes * 60 = 360K，但批内实际边数远超该值
            # 现在改为 val_edge_budget * 1.5，为前向传播动态边预留 50% headroom
            edge_guard_limit=max(1, int(
                current_val_max_nodes_per_batch * train_edge_budget_factor * 1.5
            )),
        )
        
        # [新增] 验证结束，下一轮开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 提取指标
        if isinstance(avg_val_loss, dict):
            val_metrics = avg_val_loss
            avg_val_loss_scalar = val_metrics["val_loss"]
            mean_rmsd = val_metrics["mean_rmsd_final"]
        else:
            avg_val_loss_scalar = avg_val_loss
            mean_rmsd = float("inf")
            val_metrics = {}

        val_oom_batches = int(val_metrics.get("oom_batches", 0)) if isinstance(val_metrics, dict) else 0

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

        # ReduceLROnPlateau 已移除，scheduler 已在 Step 级自动推进

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss_scalar:.4f} | Val RMSD: {mean_rmsd:.4f} | "
            f"Median: {val_metrics.get('median_rmsd_final', float('inf')):.4f} | "
            f"Centroid: {val_metrics.get('centroid_dist_mean', float('inf')):.4f} | "
            f"Pearson: {val_metrics.get('pearson_r', 0):.4f} | "
            f"OOM batches: epoch={epoch_oom_batches}, total={total_oom_batches} | "
            f"Edge-guard skips: {epoch_edge_guard_skips} | "
            f"OOM-blacklisted samples: {len(oom_blacklisted_pdb_ids)}"
        )

        selection_metrics = _build_selection_metrics(val_metrics)
        logger.info(
            "Checkpoint selection metrics | "
            f"Composite: {selection_metrics['composite_score']:.4f} | "
            f"Success@2A: {selection_metrics['success_2a']:.2f} | "
            f"Success@5A: {selection_metrics['success_5a']:.2f} | "
            f"Mean RMSD: {selection_metrics['mean_rmsd']:.4f} | "
            f"Val Loss: {selection_metrics['val_loss']:.4f}"
        )

        checkpoint = _compose_checkpoint(
            epoch_idx=epoch,
            avg_train_loss_value=avg_train_loss,
            val_metrics_obj=val_metrics,
            selection_metrics=selection_metrics,
        )

        # 1. 始终保存最新模型（作为保底）
        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pt"))

        # 2. Best model：Curriculum 结束后按多指标分别保存
        is_warmup = epoch < warmup_epochs
        if not is_warmup:
            if _is_better_checkpoint(
                selection_metrics,
                best_composite_metrics,
                primary_key="composite_score",
                primary_higher_is_better=True,
            ):
                best_composite_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_composite_model.pt"))
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                logger.info(
                    "Saved best composite model | "
                    f"Composite={selection_metrics['composite_score']:.4f}, "
                    f"Success@2A={selection_metrics['success_2a']:.2f}, "
                    f"Success@5A={selection_metrics['success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if _is_better_checkpoint(
                selection_metrics,
                best_success2a_metrics,
                primary_key="success_2a",
                primary_higher_is_better=True,
            ):
                best_success2a_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_success2a_model.pt"))
                logger.info(
                    "Saved best Success@2A model | "
                    f"Success@2A={selection_metrics['success_2a']:.2f}, "
                    f"Success@5A={selection_metrics['success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if _is_better_checkpoint(
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

        # 3. 每 10 轮保存一个永久备份
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

    # 6. 训练完成后的独立测试集评估（用于最终报告/专利材料）
    if run_test_after_training:
        if len(test_set) == 0:
            logger.warning("Test set is empty; skipping final test evaluation.")
        else:
            preferred_ckpt_paths = [
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
            test_metrics_raw = compute_validation_loss(
                model=model,
                matcher=matcher,
                criterion=criterion,
                loader=test_loader,
                device=device,
                epoch=epochs,
                total_epochs=epochs,
                max_rmsd_batches=max(10_000_000, len(test_set)),
                dataset=dataset,
                warmup_epochs=warmup_epochs,
                edge_guard_limit=max(1, int(
                    configured_test_max_nodes_per_batch * train_edge_budget_factor * eval_edge_guard_headroom
                )),
            )
            if isinstance(test_metrics_raw, dict):
                test_metrics = dict(test_metrics_raw)
            else:
                test_metrics = {"val_loss": float(test_metrics_raw)}

            topn_loader = _build_eval_loader(test_set, configured_topn_max_nodes_per_batch)

            topn_metrics = evaluate_topn_success(
                model=model,
                matcher=matcher,
                loader=topn_loader,
                device=device,
                topk_values=test_topk_values,
                num_pose_samples=max(test_pose_samples, max(test_topk_values)),
                ode_steps=50,
                warmup_epochs=warmup_epochs,
                edge_guard_limit=max(1, int(
                    configured_topn_max_nodes_per_batch * train_edge_budget_factor * eval_edge_guard_headroom
                )),
            )
            test_metrics.update(topn_metrics)

            report_dir = os.path.join(save_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "test_metrics.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved final test report to {report_path}")
            logger.info(f"[Test Summary] {test_metrics}")


@torch.no_grad()
def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int = 1,
    max_rmsd_batches: int = 10,
    dataset: PDBBindDataset | None = None,
    warmup_epochs: int = 20,
    edge_guard_limit: int | None = None,
) -> dict | float:
    """
    验证函数：计算 Loss 并统计全量 RMSD 指标
    """
    model.eval()
    total_loss = 0.0
    all_rmsd_init: list[torch.Tensor] = []
    all_rmsd_final: list[torch.Tensor] = []
    all_centroid_dist: list[torch.Tensor] = []   # 质心距离
    affinity_preds: list[torch.Tensor] = []
    affinity_targets: list[torch.Tensor] = []
    
    valid_batches = 0
    oom_batches = 0
    edge_guard_skips = 0
    
    # 固定随机种子 (保持验证集生成的一致性)
    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

    # 使用 tqdm 显示验证进度，按精确样本数统计
    val_total_graphs = len(cast(Any, loader).dataset) if hasattr(loader, "dataset") else 0
    pbar = tqdm(total=val_total_graphs, desc=f"Epoch {(epoch or 0) + 1} [Val]", leave=False, unit="graphs")

    for i, batch in enumerate(loader):
        num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
        pbar.update(num_graphs)

        if edge_guard_limit is not None:
            total_edges_cpu = 0
            edge_types = getattr(batch, "edge_types", None)
            if edge_types:
                for edge_type in edge_types:
                    edge_store = batch[edge_type]
                    edge_index = getattr(edge_store, "edge_index", None)
                    if edge_index is not None and edge_index.ndim == 2:
                        total_edges_cpu += int(edge_index.size(1))

            if total_edges_cpu > edge_guard_limit:
                edge_guard_skips += 1
                logger.warning(
                    f"Validation batch {i}: preflight skip due to edge-heavy batch "
                    f"(total_edges={total_edges_cpu} > limit={edge_guard_limit})."
                )
                continue

        try:
            batch = batch.to(device)
            apply_loss_context(
                batch,
                current_epoch=epoch if epoch is not None else total_epochs - 1,
                total_epochs_count=total_epochs,
                warmup_epochs_count=warmup_epochs,
                training=False,
            )
            x_1 = batch["ligand_atom"].pos

            # 1. 计算 Loss (用于早停和模型选择)
            t, x_t, targets = matcher.sample_location_and_target(
                x_1=x_1,
                data=batch,
                current_epoch=epoch if epoch is not None else 0,
                total_epochs=total_epochs
            )

            batch["ligand_atom"].pos = x_t
            batch.t = t  # 注入时间步，供 Loss 时间掩码使用

            predictions = model(batch, t)

            # matcher 已返回分解好的 SE(3) 目标，直接补全结合能
            targets["binding_affinity_target"] = batch.get("y_energy", None)

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"]
            
            # 过滤爆炸 Loss
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                total_loss += loss.item()
                valid_batches += 1
                
            # [新增] 收集亲和力预测 (用于计算 RMSE/Pearson/Spearman)
            # 与 losses.py 保持一致：仅在 t > 0.8 时收集，避免噪声位姿下的能量预测污染统计
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                # 遵循 losses.py 中的物理约束阈值
                if t is not None:
                    valid_mask = t > 0.8
                else:
                    valid_mask = torch.ones_like(batch.get("y_energy", torch.zeros(1)), dtype=torch.bool)
                
                if valid_mask.any():
                    pred_aff = predictions.get("binding_affinity", None)
                    if pred_aff is not None:
                        # 仅选取 t > 0.5 的预测值
                        pred_aff_valid = pred_aff[valid_mask]
                        # 双重检查：预测值本身也不能含 NaN
                        if not torch.isnan(pred_aff_valid).any():
                            affinity_preds.append(pred_aff_valid.cpu())
                            # target 统一为 raw（若已提供 raw 则直接用，否则做一次反归一化）
                            if hasattr(batch, "y_energy_raw"):
                                target_raw_valid = batch.y_energy_raw[valid_mask]
                                affinity_targets.append(target_raw_valid.cpu())
                            else:
                                y_norm = batch.get("y_energy", None)
                                if y_norm is not None and dataset is not None:
                                    target_raw_valid = dataset.denormalize_affinity(y_norm[valid_mask].cpu())
                                    affinity_targets.append(target_raw_valid)
            
            # 2. 全量 RMSD 推演
            # -----------------------------------------------------------
            if i < max_rmsd_batches:
                try:
                    # 克隆数据用于推演
                    infer_batch = batch.clone()
                    infer_batch["ligand_atom"].pos = x_1 
                    
                    # 验证始终使用全难度扰动（与最终推理条件一致）
                    x_0_infer = matcher._generate_random_pose(
                        x_ref=x_1,
                        batch=infer_batch["ligand_atom"].batch,
                        B=int(infer_batch["ligand_atom"].batch.max().item()) + 1,
                        masses=infer_batch["ligand_atom"].masses,
                        torsion_indices=getattr(infer_batch, "torsion_indices", None),
                        torsion_moving_mask=getattr(infer_batch, "torsion_moving_mask", None),
                        epoch=warmup_epochs,
                    )
                    
                    # 记录初始 RMSD
                    sq_diff_init = ((x_0_infer - x_1) ** 2).sum(dim=-1)
                    msd_init = scatter_mean(sq_diff_init, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_init = torch.sqrt(msd_init)
                    # [修改] 强制转 CPU，切断 GPU 显存占用
                    all_rmsd_init.append(rmsd_init.detach().cpu())

                    # 执行推演（Euler 50 步，用于训练期间的趋势监控）
                    infer_batch["ligand_atom"].pos = x_0_infer
                    final_pos, _ = matcher.ode_solve(
                        model=model,
                        data=infer_batch,
                        steps=50,
                        method="euler",
                        store_trajectory=False,
                    )
                    
                    # 记录最终 RMSD
                    sq_diff_final = ((final_pos - x_1) ** 2).sum(dim=-1)
                    msd_final = scatter_mean(sq_diff_final, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_final = torch.sqrt(msd_final)
                    all_rmsd_final.append(rmsd_final.detach().cpu())

                    # 记录质心距离（Centroid Distance）
                    B_infer = int(infer_batch["ligand_atom"].batch.max().item()) + 1
                    pred_centroid = scatter_mean(final_pos, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    true_centroid = scatter_mean(x_1, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    centroid_dist = torch.norm(pred_centroid - true_centroid, dim=-1)  # [B_infer]
                    all_centroid_dist.append(centroid_dist.detach().cpu())

                    del infer_batch, x_0_infer, final_pos, sq_diff_init, msd_init, rmsd_init
                    del sq_diff_final, msd_final, rmsd_final, pred_centroid, true_centroid, centroid_dist
                    
                except Exception as e:
                    logger.warning(f"RMSD inference failed for batch {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            # -----------------------------------------------------------

        except torch.cuda.OutOfMemoryError:
            oom_batches += 1
            logger.warning(f"Validation batch {i}: CUDA OOM, skipping and clearing cache.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        except Exception as e:
            logger.warning(f"Validation batch failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        del predictions, targets, loss_dict, loss, x_1, x_t, t, batch

    # ========== 综合评估指标计算 ==========
    metrics: dict[str, float] = {}

    # --- 亲和力预测指标 ---
    pearson_r = 0.0
    spearman_rho = 0.0
    rmse_val = float("inf")
    mae_val = float("inf")

    if len(affinity_preds) > 0 and dataset is not None:
        cat_preds = torch.cat(affinity_preds).view(-1)
        cat_targets = torch.cat(affinity_targets).view(-1)
        
        # 模型输出是 norm，验证时仅在这里做一次反归一化
        raw_preds: torch.Tensor = dataset.denormalize_affinity(cat_preds)
        
        mse_val = F.mse_loss(raw_preds, cat_targets)
        rmse_val = torch.sqrt(mse_val).item()
        mae_val = F.l1_loss(raw_preds, cat_targets).item()

        # Pearson R（线性相关性，对应 CASF-2016 Scoring Power）
        # Spearman ρ（排序一致性，对应 CASF-2016 Ranking Power）
        pred_np = raw_preds.numpy()
        target_np = cat_targets.numpy()
        if len(pred_np) > 2 and np.std(pred_np) > 1e-6:
            pearson_res = scipy_stats.pearsonr(pred_np, target_np)
            spearman_res = scipy_stats.spearmanr(pred_np, target_np)
            pearson_r = float(cast(Any, pearson_res)[0])
            spearman_rho = float(cast(Any, spearman_res)[0])
        
        logger.info(f"[Validation Affinity] RMSE: {rmse_val:.4f} pKd | MAE: {mae_val:.4f} pKd")
        logger.info(f"  Pearson R: {pearson_r:.4f} | Spearman ρ: {spearman_rho:.4f}")

    metrics["affinity_rmse"] = rmse_val
    metrics["affinity_mae"] = mae_val
    metrics["pearson_r"] = pearson_r
    metrics["spearman_rho"] = spearman_rho

    # --- 对接精度指标 ---
    mean_final = float("inf")
    median_final = float("inf")
    success_2a = 0.0
    success_5a = 0.0
    mean_centroid = float("inf")
    median_centroid = float("inf")

    if len(all_rmsd_final) > 0:
        cat_rmsd_init = torch.cat(all_rmsd_init)
        cat_rmsd_final = torch.cat(all_rmsd_final)
        
        mean_init = cat_rmsd_init.mean().item()
        mean_final = cat_rmsd_final.mean().item()
        median_final = cat_rmsd_final.median().item()
        
        # 成功率（多阈值）
        success_2a = (cat_rmsd_final < 2.0).float().mean().item() * 100
        success_5a = (cat_rmsd_final < 5.0).float().mean().item() * 100

        # 质心距离
        if len(all_centroid_dist) > 0:
            cat_centroid = torch.cat(all_centroid_dist)
            mean_centroid = cat_centroid.mean().item()
            median_centroid = cat_centroid.median().item()
        
        logger.info("-" * 60)
        logger.info(f"[Validation Full Stats] Epoch {(epoch or 0) + 1}")
        logger.info(f"  Mean RMSD: {mean_init:.2f} -> {mean_final:.2f} Å | Median: {median_final:.2f} Å")
        logger.info(f"  Success Rate (<2Å): {success_2a:.2f}% | (<5Å): {success_5a:.2f}%")
        logger.info(f"  Centroid Distance: Mean {mean_centroid:.2f} Å | Median {median_centroid:.2f} Å")
        logger.info("-" * 60)

    metrics["mean_rmsd_final"] = mean_final
    metrics["median_rmsd_final"] = median_final
    metrics["success_2a"] = success_2a
    metrics["success_5a"] = success_5a
    metrics["centroid_dist_mean"] = mean_centroid
    metrics["centroid_dist_median"] = median_centroid
    metrics["oom_batches"] = float(oom_batches)
    metrics["valid_batches"] = float(valid_batches)
    metrics["edge_guard_skips"] = float(edge_guard_skips)

    if valid_batches == 0:
        metrics["val_loss"] = float("nan")
        return metrics

    # 显式清理现场
    del all_rmsd_init, all_rmsd_final, all_centroid_dist
    del affinity_preds, affinity_targets
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics["val_loss"] = total_loss / valid_batches
    return metrics


@torch.no_grad()
def evaluate_topn_success(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    loader: DataLoader,
    device: torch.device,
    topk_values: tuple[int, ...] = (1, 5, 10),
    num_pose_samples: int = 10,
    ode_steps: int = 50,
    warmup_epochs: int = 20,
    edge_guard_limit: int | None = None,
) -> dict[str, float]:
    """
    基于多候选 pose 生成 + 亲和力排序，统计 Top-N 对接成功率。

    评估流程（按 batch）：
    1. 每个复合物生成 num_pose_samples 个候选 pose；
    2. 用模型亲和力 head 对候选 pose 打分并排序；
    3. 对每个 N，统计 Top-N 内最优 RMSD 是否 < 2Å / < 5Å。
    """

    model.eval()

    if num_pose_samples <= 0:
        raise ValueError(f"num_pose_samples must be > 0, got {num_pose_samples}")

    topk_unique = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    if not topk_unique:
        raise ValueError("topk_values must contain at least one positive integer")

    max_k = max(topk_unique)
    if num_pose_samples < max_k:
        raise ValueError(
            f"num_pose_samples ({num_pose_samples}) must be >= max top-k ({max_k})"
        )

    success_counts_2a = {k: 0.0 for k in topk_unique}
    success_counts_5a = {k: 0.0 for k in topk_unique}
    mean_best_rmsd = {k: [] for k in topk_unique}
    total_graphs = 0
    edge_guard_skips = 0

    for batch_idx, batch in enumerate(loader):
        try:
            if edge_guard_limit is not None:
                total_edges_cpu = 0
                edge_types = getattr(batch, "edge_types", None)
                if edge_types:
                    for edge_type in edge_types:
                        edge_store = batch[edge_type]
                        edge_index = getattr(edge_store, "edge_index", None)
                        if edge_index is not None and edge_index.ndim == 2:
                            total_edges_cpu += int(edge_index.size(1))

                if total_edges_cpu > edge_guard_limit:
                    edge_guard_skips += 1
                    logger.warning(
                        f"Top-N eval batch {batch_idx}: preflight skip due to edge-heavy batch "
                        f"(total_edges={total_edges_cpu} > limit={edge_guard_limit})."
                    )
                    continue

            batch = batch.to(device)
            x_ref = batch["ligand_atom"].pos
            lig_batch = batch["ligand_atom"].batch
            masses = batch["ligand_atom"].masses
            B = int(lig_batch.max().item()) + 1

            torsion_indices = getattr(batch, "torsion_indices", None)
            torsion_moving_mask = getattr(batch, "torsion_moving_mask", None)

            candidate_rmsd: list[torch.Tensor] = []
            candidate_scores: list[torch.Tensor] = []

            for pose_id in range(num_pose_samples):
                infer_batch = batch.clone()
                x0 = matcher._generate_random_pose(
                    x_ref=x_ref,
                    batch=lig_batch,
                    B=B,
                    masses=masses,
                    torsion_indices=torsion_indices,
                    torsion_moving_mask=torsion_moving_mask,
                    epoch=warmup_epochs + pose_id,
                )

                infer_batch["ligand_atom"].pos = x0
                final_pos, _ = matcher.ode_solve(
                    model=model,
                    data=infer_batch,
                    steps=ode_steps,
                    method="euler",
                    store_trajectory=False,
                )

                sq_diff = ((final_pos - x_ref) ** 2).sum(dim=-1)
                rmsd_per_graph = torch.sqrt(scatter_mean(sq_diff, lig_batch, dim=0, dim_size=B))
                candidate_rmsd.append(rmsd_per_graph.detach().cpu())

                score_batch = infer_batch.clone()
                score_batch["ligand_atom"].pos = final_pos
                score_t = torch.ones(B, device=device, dtype=final_pos.dtype)
                score_out = model(score_batch, score_t)
                score = score_out["binding_affinity"].view(-1)
                candidate_scores.append(score.detach().cpu())

                del infer_batch, score_batch, x0, final_pos, sq_diff, rmsd_per_graph, score_t, score_out, score

            rmsd_mat = torch.stack(candidate_rmsd, dim=1)    # [B, P]
            score_mat = torch.stack(candidate_scores, dim=1) # [B, P]
            rank_idx = torch.argsort(score_mat, dim=1, descending=True)

            for k in topk_unique:
                topk_idx = rank_idx[:, :k]
                topk_rmsd = torch.gather(rmsd_mat, dim=1, index=topk_idx)
                best_rmsd_k = torch.min(topk_rmsd, dim=1).values

                success_counts_2a[k] += float((best_rmsd_k < 2.0).float().sum().item())
                success_counts_5a[k] += float((best_rmsd_k < 5.0).float().sum().item())
                mean_best_rmsd[k].append(best_rmsd_k)

            total_graphs += B

            del batch, x_ref, lig_batch, masses, candidate_rmsd, candidate_scores, rmsd_mat, score_mat, rank_idx

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Top-N eval batch {batch_idx}: CUDA OOM, skipping.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as exc:
            logger.warning(f"Top-N eval batch {batch_idx} failed: {exc}\n{traceback.format_exc()}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    if total_graphs == 0:
        return {
            "topn_total_graphs": 0.0,
        }

    metrics: dict[str, float] = {
        "topn_total_graphs": float(total_graphs),
        "topn_pose_samples": float(num_pose_samples),
        "topn_edge_guard_skips": float(edge_guard_skips),
    }

    for k in topk_unique:
        best_rmsd_all = torch.cat(mean_best_rmsd[k], dim=0) if mean_best_rmsd[k] else torch.tensor([], dtype=torch.float32)
        metrics[f"top{k}_success_2a"] = (success_counts_2a[k] / total_graphs) * 100.0
        metrics[f"top{k}_success_5a"] = (success_counts_5a[k] / total_graphs) * 100.0
        metrics[f"top{k}_mean_best_rmsd"] = float(best_rmsd_all.mean().item()) if best_rmsd_all.numel() > 0 else float("inf")
        metrics[f"top{k}_median_best_rmsd"] = float(best_rmsd_all.median().item()) if best_rmsd_all.numel() > 0 else float("inf")

    return metrics
