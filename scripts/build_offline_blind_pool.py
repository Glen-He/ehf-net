"""
离线候选池生成脚本。

用最优 checkpoint 在全训练集（或大子集）上跑完整 blind pipeline，
生成覆盖全面的候选池，用于最终 reranker fine-tuning。

Usage:
    python scripts/build_offline_blind_pool.py \
        --checkpoint checkpoints/best_model.pt \
        --data_root data/processed/hiqbind \
        --output_dir cache/offline_pool \
        --max_complexes 0 \
        --ode_steps 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from ehfnet.models import EHFNet
from ehfnet.graph import GraphCollator
from ehfnet.datasets.protein_ligand import ProteinLigandDataset
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.blind_pool import (
    build_blind_pool_compatibility,
    save_blind_pool,
    get_pool_stats,
)
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from ehfnet.training.checkpoint_schema import validate_checkpoint_compatibility
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build offline blind candidate pool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录，须含 index.csv")
    parser.add_argument("--output_dir", type=str, default="cache/offline_pool")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_complexes", type=int, default=0, help="0 = all")
    parser.add_argument("--center_topk", type=int, default=8)
    parser.add_argument("--refine_topk", type=int, default=3)
    parser.add_argument("--stage1_pose_samples", type=int, default=2)
    parser.add_argument("--stage2_pose_samples", type=int, default=8)
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--crop_radius", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--esm_path", type=str, default=None)
    args = parser.parse_args()
    index_file = os.path.join(args.data_root, "index.csv")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = validate_checkpoint_compatibility(ckpt)
    model_state = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    if model_state is None:
        raise ValueError(f"Checkpoint {args.checkpoint} does not contain model weights.")
    normalization_stats = ckpt.get("normalization_stats")

    model = EHFNet(
        hidden_dim=int(model_config["hidden_dim"]),
        time_dim=int(model_config["time_dim"]),
        num_gnn_blocks=int(model_config["num_gnn_blocks"]),
        lig_atom_cont_count=int(model_config["lig_atom_cont_count"]),
        lig_mol_cont_count=int(model_config["lig_mol_cont_count"]),
        pro_atom_cont_count=int(model_config["pro_atom_cont_count"]),
        pro_res_cont_count=int(model_config["pro_res_cont_count"]),
        m_dim_scalar=int(model_config.get("m_dim_scalar", 16)),
        dropout_rate=float(model_config.get("dropout_rate", 0.0)),
        num_rbf=int(model_config.get("num_rbf", 50)),
        r_cutoff=float(model_config.get("r_cutoff", 10.0)),
        force_cutoff=float(model_config.get("force_cutoff", 6.0)),
        fix_protein=bool(model_config.get("fix_protein", True)),
        interaction_profile=str(model_config["interaction_profile"]),
        normalization_stats=normalization_stats,
        dynamic_inter_cutoff=float(model_config.get("dynamic_inter_cutoff", 10.0)),
        dynamic_inter_knn_k=int(model_config.get("dynamic_inter_knn_k", 8)),
        dynamic_residue_cutoff=float(model_config.get("dynamic_residue_cutoff", 14.0)),
        dynamic_residue_knn_k=int(model_config.get("dynamic_residue_knn_k", 6)),
    ).to(device)

    if model_state:
        model.load_state_dict(model_state, strict=True)
    model.eval()
    logger.info("Model loaded from %s", args.checkpoint)

    matcher = ConditionalFlowMatcher(sigma_min=1e-3, warmup_epochs=20)

    dataset = ProteinLigandDataset(
        root=args.data_root,
        index_file=index_file,
        esm_root=args.esm_path,
        esm="auto",
        esm_dim=int(model_config["esm_dim"]),
        interaction_profile=str(model_config["interaction_profile"]),
    )
    graph_builder = dataset.graph_builder
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator.collate,
        num_workers=2,
        shuffle=False,
    )

    max_c = args.max_complexes if args.max_complexes > 0 else None

    logger.info(
        "Generating offline pool | complexes=%s | centers=%d | s1=%d | s2=%d | ode=%d",
        max_c or "all", args.center_topk, args.stage1_pose_samples,
        args.stage2_pose_samples, args.ode_steps,
    )

    records = generate_candidates_from_loader(
        model=model,
        matcher=matcher,
        loader=loader,
        device=device,
        graph_builder=graph_builder,
        collator=collator,
        center_topk=args.center_topk,
        refine_topk=args.refine_topk,
        stage1_pose_samples=args.stage1_pose_samples,
        stage2_pose_samples=args.stage2_pose_samples,
        crop_radius=args.crop_radius,
        ode_steps=args.ode_steps,
        max_complexes=max_c,
        generator_ckpt_id=args.checkpoint,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pool_compatibility = build_blind_pool_compatibility(
        esm_dim=int(model_config["esm_dim"]),
        processed_dir=dataset.processed_dir,
        index_file=dataset.index_file,
        interaction_profile=str(model_config["interaction_profile"]),
    )
    save_blind_pool(records, args.output_dir, epoch=0, meta={
        "compatibility": pool_compatibility,
        "checkpoint": args.checkpoint,
        "total_complexes": len(records),
        "mode": "offline_blind",
    })

    stats = get_pool_stats(records)
    logger.info("Offline pool complete: %s", stats)


if __name__ == "__main__":
    main()
