"""
训练入口脚本

该脚本用于解析命令行参数，配置日志记录，并启动 EHFNet 模型的训练过程。
"""

import argparse
import os
import sys
import logging

from datetime import datetime
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# [修复] 必须在 import torch 之前设置 CUDA allocator 配置，否则可能不生效
# expandable_segments:True -> 允许分配器动态扩展显存段，缓解碎片化
# max_split_size_mb:128 -> 避免大块显存被切得太碎
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from ehfnet.training.trainer import train
import torch

# [新增] 全局开启 TF32 (提速神器)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser(description="Train EHFNet for molecular docking prediction")
    
    # 数据相关参数
    # 使用动态路径作为默认值
    default_data_root = PROJECT_ROOT / "data/processed/pdbbind"
    default_index_file = default_data_root / "index.csv"
    
    parser.add_argument("--data_root", type=str, default=str(default_data_root), help="Path to PDBBind dataset root directory")
    parser.add_argument("--index_file", type=str, default=str(default_index_file), help="Path to index CSV or PDBBind index file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--esm_path", type=str, default=None, help="Path to precomputed ESM embeddings (optional)")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    
    # 模型相关参数
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_gnn_blocks", type=int, default=4, help="Number of GNN blocks")
    
    # 特征相关参数（通常固定，但为灵活性暴露）
    parser.add_argument("--lig_atom_cont_count", type=int, default=9, help="Ligand atom continuous feature count")
    parser.add_argument("--lig_mol_cont_count", type=int, default=9, help="Ligand molecule continuous feature count")
    parser.add_argument("--pro_atom_cont_count", type=int, default=5, help="Protein atom continuous feature count")
    parser.add_argument("--esm_dim", type=int, default=960, help="ESM embedding dimension (default: 960 for ESMC-300M)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--pocket_radius", type=float, default=12.0, help="Radius (A) for protein pocket extraction (default: 12.0)")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs for spatial curriculum learning (default: 20)")
    parser.add_argument("--rmsd_ratio", type=float, default=0.2, help="Ratio of validation set to compute RMSD (0.0-1.0)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_nodes_per_batch", type=int, default=20000, help="Max nodes per batch for DynamicBatchSampler.")
    parser.add_argument("--val_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for validation loader (default: min(train_budget, 6000))")
    parser.add_argument("--test_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for final test loader (default: same as val budget)")
    parser.add_argument("--topn_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for Top-N evaluation loader (default: same as test budget)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate (default: 0.999; use 0.99 for quick smoke tests)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True, help="Enable DataLoader pin_memory")
    parser.add_argument("--no_dataloader_pin_memory", dest="dataloader_pin_memory", action="store_false", help="Disable DataLoader pin_memory")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=True, help="Enable DataLoader persistent_workers")
    parser.add_argument("--no_dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store_false", help="Disable DataLoader persistent_workers")
    parser.add_argument("--enable_oom_adaptive_batch", action="store_true", default=True, help="Auto-reduce max_nodes_per_batch when frequent CUDA OOM occurs")
    parser.add_argument("--disable_oom_adaptive_batch", dest="enable_oom_adaptive_batch", action="store_false", help="Disable adaptive OOM batch protection")
    parser.add_argument("--oom_reduce_threshold", type=int, default=3, help="Reduce batch node budget when OOM batches in an epoch reach this threshold")
    parser.add_argument("--oom_reduce_factor", type=float, default=0.85, help="Factor to shrink max_nodes_per_batch after OOM threshold (0-1)")
    parser.add_argument("--min_max_nodes_per_batch", type=int, default=12000, help="Lower bound for adaptive max_nodes_per_batch")
    parser.add_argument("--enable_val_oom_adaptive_batch", action="store_true", default=True, help="Auto-reduce validation node budget when validation OOM is frequent")
    parser.add_argument("--disable_val_oom_adaptive_batch", dest="enable_val_oom_adaptive_batch", action="store_false", help="Disable validation OOM adaptive protection")
    parser.add_argument("--val_oom_reduce_threshold", type=int, default=3, help="Reduce validation node budget when validation OOM batches reach this threshold")
    parser.add_argument("--val_oom_reduce_factor", type=float, default=0.85, help="Factor to shrink validation max_nodes_per_batch after OOM threshold (0-1)")
    parser.add_argument("--min_val_max_nodes_per_batch", type=int, default=None, help="Lower bound for adaptive validation max_nodes_per_batch (default: same as min_max_nodes_per_batch)")
    parser.add_argument("--oom_recover_epochs", type=int, default=3, help="Consecutive clean epochs before attempting batch budget recovery")
    parser.add_argument("--oom_recover_factor", type=float, default=1.1, help="Factor to grow max_nodes_per_batch during recovery (>1)")
    parser.add_argument("--split_train_frac", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--split_val_frac", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--split_test_frac", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for scaffold split")
    parser.add_argument("--split_cache_file", type=str, default=None, help="Path to persisted split JSON")
    parser.add_argument("--force_resplit", action="store_true", help="Force regenerate split JSON")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "inter_multiscale_off"],
        help="Ablation mode: none (full model) or inter_multiscale_off (atom-atom only)",
    )
    parser.add_argument("--run_test_after_training", action="store_true", default=True, help="Run final test-set evaluation after training")
    parser.add_argument("--skip_test_after_training", dest="run_test_after_training", action="store_false", help="Skip final test-set evaluation")
    parser.add_argument("--test_topk", type=str, default="1,5,10", help="Comma-separated top-k values for final test evaluation")
    parser.add_argument("--test_pose_samples", type=int, default=10, help="Number of candidate poses per complex for Top-N evaluation")

    args = parser.parse_args()

    try:
        parsed_topk = tuple(int(x.strip()) for x in args.test_topk.split(",") if x.strip())
        if not parsed_topk:
            raise ValueError("empty top-k list")
        args.test_topk_values = parsed_topk
    except Exception as e:
        raise ValueError(f"Invalid --test_topk='{args.test_topk}': {e}") from e
    
    # 动态计算 pro_res_cont_count: 14 (扭转角) + esm_dim
    args.pro_res_cont_count = 14 + args.esm_dim

    # 配置 logging
    # 将日志保存在 logs/train 目录下，并使用时间戳防止覆盖
    log_dir = os.path.join("logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logging.info(f"Logging to {log_file}")

    # 为当前运行创建独立输出目录，避免覆盖历史 checkpoint/report
    base_save_dir = args.save_dir
    args.save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Run artifacts will be saved to {args.save_dir}")
    logger.info(f"Starting training with arguments: {args}")

    # 加载归一化统计数据
    stats_file = os.path.join(args.data_root, "normalization_stats.pt")
    if not os.path.exists(stats_file):
        logger.warning(f"Normalization stats not found at {stats_file}. Computing now...")
        processed_dir = os.path.join(args.data_root, "processed")
        
        try:
            # 确保 processed 目录存在
            if not os.path.exists(processed_dir):
                 logger.warning(f"Processed data dir {processed_dir} not found. Stats will be computed next time.")
                 normalization_stats = None

            else:
                # 调用脚本计算
                from scripts.compute_dataset_stats import compute_stats
                compute_stats(processed_dir, stats_file)
                normalization_stats = torch.load(stats_file, weights_only=False)
                logger.info("Stats computed and loaded.")

        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            normalization_stats = None
    else:
        normalization_stats = torch.load(stats_file, weights_only=False)
        logger.info(f"Loaded normalization stats from {stats_file}")

    try:
        train(
            data_root=args.data_root,
            index_file=args.index_file,
            save_dir=args.save_dir,
            esm_path=args.esm_path,
            epochs=args.epochs,

            lr=args.lr,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            hidden_dim=args.hidden_dim,
            num_gnn_blocks=args.num_gnn_blocks,
            lig_atom_cont_count=args.lig_atom_cont_count,
            lig_mol_cont_count=args.lig_mol_cont_count,
            pro_atom_cont_count=args.pro_atom_cont_count,
            pro_res_cont_count=args.pro_res_cont_count,
            esm_dim=args.esm_dim,
            device=args.device,
            pocket_radius=args.pocket_radius,
            normalization_stats=normalization_stats,
            warmup_epochs=args.warmup_epochs,
            rmsd_check_ratio=args.rmsd_ratio,
            accumulation_steps=args.accumulation_steps,
            max_nodes_per_batch=args.max_nodes_per_batch,
            val_max_nodes_per_batch=args.val_max_nodes_per_batch,
            test_max_nodes_per_batch=args.test_max_nodes_per_batch,
            topn_max_nodes_per_batch=args.topn_max_nodes_per_batch,
            ema_decay=args.ema_decay,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            split_train_frac=args.split_train_frac,
            split_val_frac=args.split_val_frac,
            split_test_frac=args.split_test_frac,
            split_seed=args.split_seed,
            split_cache_file=args.split_cache_file,
            force_resplit=args.force_resplit,
            ablation_mode=args.ablation_mode,
            run_test_after_training=args.run_test_after_training,
            test_topk_values=args.test_topk_values,
            test_pose_samples=args.test_pose_samples,
            enable_oom_adaptive_batch=args.enable_oom_adaptive_batch,
            oom_reduce_threshold=args.oom_reduce_threshold,
            oom_reduce_factor=args.oom_reduce_factor,
            min_max_nodes_per_batch=args.min_max_nodes_per_batch,
            enable_val_oom_adaptive_batch=args.enable_val_oom_adaptive_batch,
            val_oom_reduce_threshold=args.val_oom_reduce_threshold,
            val_oom_reduce_factor=args.val_oom_reduce_factor,
            min_val_max_nodes_per_batch=args.min_val_max_nodes_per_batch,
            oom_recover_epochs=args.oom_recover_epochs,
            oom_recover_factor=args.oom_recover_factor,
            run_name=run_name,
            run_log_file=log_file,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
