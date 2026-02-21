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

from ehfnet.training.trainer import train
import torch

# [新增] 在 import torch 之前或刚开始设置
# expandable_segments:True -> 允许分配器动态扩展显存段，极大缓解碎片化
# max_split_size_mb:128 -> 避免大块显存被切得太碎
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# [新增] 全局开启 TF32 (提速神器)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    torch.autograd.set_detect_anomaly(True)
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    
    # 模型相关参数
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_gnn_blocks", type=int, default=6, help="Number of GNN blocks")
    
    # 特征相关参数（通常固定，但为灵活性暴露）
    parser.add_argument("--lig_atom_cont_count", type=int, default=9, help="Ligand atom continuous feature count")
    parser.add_argument("--lig_mol_cont_count", type=int, default=9, help="Ligand molecule continuous feature count")
    parser.add_argument("--pro_atom_cont_count", type=int, default=5, help="Protein atom continuous feature count")
    parser.add_argument("--esm_dim", type=int, default=960, help="ESM embedding dimension (default: 960 for ESMC-300M)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--pocket_radius", type=float, default=20.0, help="Radius (A) for protein pocket extraction (default: 20.0)")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs for spatial curriculum learning (default: 20)")
    parser.add_argument("--rmsd_ratio", type=float, default=0.1, help="Ratio of validation set to compute RMSD (0.0-1.0)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate (default: 0.999; use 0.99 for quick smoke tests)")

    args = parser.parse_args()
    
    # 动态计算 pro_res_cont_count: 14 (扭转角) + esm_dim
    args.pro_res_cont_count = 14 + args.esm_dim

    # 配置 logging
    # 将日志保存在 logs/train 目录下，并使用时间戳防止覆盖
    log_dir = os.path.join("logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logging.info(f"Logging to {log_file}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
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
            batch_size=args.batch_size,
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
            ema_decay=args.ema_decay,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
