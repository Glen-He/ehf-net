import argparse
import os
import sys
import logging

# 若从源码直接运行（未安装），将 src 加入 python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ehfnet.training.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train EHFNet for molecular docking prediction")
    
    # 数据相关参数
    parser.add_argument("--data_root", type=str, required=True, help="Path to PDBBind dataset root directory")
    parser.add_argument("--index_file", type=str, required=True, help="Path to index CSV or PDBBind index file")
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
    parser.add_argument("--pro_res_cont_count", type=int, default=1166, help="Protein residue continuous feature count (14 torsion + 1152 ESM)")

    args = parser.parse_args()

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "train.log") if os.path.exists(args.save_dir) else "train.log")
        ]
    )
    
    # 创建保存目录（确保后续日志可写入指定目录；即使 Trainer 会创建，这里也先兜底）
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
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
