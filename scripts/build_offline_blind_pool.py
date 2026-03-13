"""
离线候选池生成脚本。

用最优 checkpoint 在全训练集（或大子集）上跑完整 blind pipeline，
生成覆盖全面的候选池，用于最终 reranker fine-tuning。

Usage:
    python scripts/build_offline_blind_pool.py \
        --checkpoint checkpoints/best_model.pt \
        --data_root data/processed/pdbbind \
        --index_file data/processed/pdbbind/index.csv \
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
from ehfnet.graph import create_graph_tools, GraphCollator
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.blind_pool import save_blind_pool, get_pool_stats
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build offline blind candidate pool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    normalization_stats = ckpt.get("normalization_stats")

    hidden_dim = ckpt.get("hidden_dim", 128)
    num_gnn_blocks = ckpt.get("num_gnn_blocks", 4)
    esm_dim = ckpt.get("esm_dim", 960)

    model = EHFNet(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=9,
        lig_mol_cont_count=9,
        pro_atom_cont_count=5,
        pro_res_cont_count=14 + esm_dim,
        normalization_stats=normalization_stats,
    ).to(device)

    if model_state:
        model.load_state_dict(model_state, strict=False)
    model.eval()
    logger.info("Model loaded from %s", args.checkpoint)

    graph_builder, _, _ = create_graph_tools()
    collator = GraphCollator()
    matcher = ConditionalFlowMatcher(sigma_min=1e-3, warmup_epochs=20)

    dataset = PDBBindDataset(
        root=args.data_root,
        index_file=args.index_file,
        graph_builder=graph_builder,
        protein_context_mode="full",
        esm_embedding_path=args.esm_path,
    )

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
    save_blind_pool(records, args.output_dir, epoch=0, meta={
        "checkpoint": args.checkpoint,
        "total_complexes": len(records),
        "mode": "offline_full",
    })

    stats = get_pool_stats(records)
    logger.info("Offline pool complete: %s", stats)


if __name__ == "__main__":
    main()
