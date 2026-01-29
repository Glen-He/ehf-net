"""
训练入口脚本

该脚本用于解析命令行参数，配置日志记录，并启动 EHFNet 模型的训练过程。
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# 若从源码直接运行（未安装），将 src 加入 python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ehfnet.training.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train EHFNet for molecular docking prediction")
    
    # 数据相关参数
    parser.add_argument("--data_root", type=str, default="/pavo/glen/Code/EHFNet/data/processed/pdbbind", help="Path to PDBBind dataset root directory")
    parser.add_argument("--index_file", type=str, default="/pavo/glen/Code/EHFNet/data/processed/pdbbind/index.csv", help="Path to index CSV or PDBBind index file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--esm_path", type=str, default=None, help="Path to precomputed ESM embeddings (optional)")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=10.0, help="Gradient clipping value")
    
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
    # parser.add_argument("--pro_res_cont_count", type=int, default=974, help="Protein residue continuous feature count (14 torsion + 960 ESM)")

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
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
