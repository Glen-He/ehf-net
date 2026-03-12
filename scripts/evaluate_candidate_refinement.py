"""
评估 candidate-conditioned refinement backbone。

评估协议：
1. 从外部 candidate pool 读取候选 pose；
2. 逐个 candidate 做 refinement；
3. 对比 raw docking score 排序 与 refined affinity 排序；
4. 输出 oracle-in-pool、Top-k、RMSD 改善和 clash 改善。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from ehfnet.datasets.candidate_store import CandidateStore
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.graph import GraphCollator
from ehfnet.models import EHFNet
from ehfnet.training.flow_matcher import ConditionalFlowMatcher


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-conditioned refinement")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--candidate_root", type=str, required=True)
    parser.add_argument("--candidate_source", type=str, default="vina")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reports/candidate_refinement")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_gnn_blocks", type=int, default=4)
    parser.add_argument("--lig_atom_cont_count", type=int, default=9)
    parser.add_argument("--lig_mol_cont_count", type=int, default=9)
    parser.add_argument("--pro_atom_cont_count", type=int, default=5)
    parser.add_argument("--pro_res_cont_count", type=int, default=974)
    parser.add_argument("--esm_dim", type=int, default=960)
    parser.add_argument("--pocket_radius", type=float, default=12.0)
    parser.add_argument("--refine_steps", type=int, default=20)
    parser.add_argument("--topk", type=str, default="1,5,10")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--split_name", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--limit_complexes", type=int, default=0)
    parser.add_argument("--limit_candidates_per_complex", type=int, default=0)
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_placeholder_normalization_stats(args: argparse.Namespace, affinity_stats: dict[str, float]) -> dict[str, dict[str, torch.Tensor]]:
    def zeros_ones(length: int) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros(length, dtype=torch.float32),
            "std": torch.ones(length, dtype=torch.float32),
        }

    return {
        "ligand_atom": zeros_ones(args.lig_atom_cont_count),
        "ligand_molecule": zeros_ones(args.lig_mol_cont_count),
        "protein_atom": zeros_ones(args.pro_atom_cont_count),
        "protein_residue": zeros_ones(args.pro_res_cont_count),
        "affinity": {
            "mean": torch.tensor(float(affinity_stats["mean"]), dtype=torch.float32),
            "std": torch.tensor(float(affinity_stats["std"]), dtype=torch.float32),
        },
    }


def topk_metrics(best_rmsd_by_rank: list[float], threshold: float) -> float:
    if not best_rmsd_by_rank:
        return 0.0
    return float(np.mean(np.asarray(best_rmsd_by_rank) < threshold) * 100.0)


def resolve_eval_pdb_ids(dataset: PDBBindDataset, split_file: str | None, split_name: str | None) -> list[str]:
    valid_pdb_ids = set(getattr(dataset, "_valid_pdb_ids", []))
    if not split_file or not split_name:
        return sorted(valid_pdb_ids)

    with Path(split_file).open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    split_indices = split_payload["indices"][split_name]
    split_df = dataset.index_df.iloc[split_indices]
    return [pdb_id for pdb_id in split_df["pdb_id"].astype(str).str.lower().tolist() if pdb_id in valid_pdb_ids]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    setup_logging(output_dir)

    topk_values = tuple(sorted({int(x.strip()) for x in args.topk.split(",") if x.strip()}))
    device = torch.device(args.device)
    dataset = PDBBindDataset(
        root=args.data_root,
        index_file=args.index_file,
        esm_root=None,
        esm="auto",
        esm_dim=args.esm_dim,
        pocket_radius=args.pocket_radius,
        candidate_root=None,
        interaction_profile="full",
    )
    candidate_store = CandidateStore(args.candidate_root, args.candidate_source)
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])

    model = EHFNet(
        hidden_dim=args.hidden_dim,
        time_dim=args.hidden_dim,
        num_gnn_blocks=args.num_gnn_blocks,
        lig_atom_cont_count=args.lig_atom_cont_count,
        lig_mol_cont_count=args.lig_mol_cont_count,
        pro_atom_cont_count=args.pro_atom_cont_count,
        pro_res_cont_count=args.pro_res_cont_count,
        normalization_stats=build_placeholder_normalization_stats(args, dataset.affinity_stats),
    ).to(device)
    matcher = ConditionalFlowMatcher(sigma_min=1e-3, warmup_epochs=20)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    if state_dict is None:
        raise KeyError(f"No model_state_dict found in checkpoint: {args.checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()

    oracle_best_before: list[float] = []
    oracle_best_after: list[float] = []
    raw_top1_rmsd: list[float] = []
    refined_top1_rmsd: list[float] = []
    refined_topk_best: dict[int, list[float]] = {k: [] for k in topk_values}
    raw_topk_best: dict[int, list[float]] = {k: [] for k in topk_values}
    all_candidate_rmsd_before: list[float] = []
    all_candidate_rmsd_after: list[float] = []
    clash_before_flags: list[float] = []
    clash_after_flags: list[float] = []
    raw_rank_eligible = 0
    evaluated_complexes = 0

    pdb_ids = resolve_eval_pdb_ids(dataset, args.split_file, args.split_name)
    if args.limit_complexes > 0:
        pdb_ids = pdb_ids[: args.limit_complexes]

    for idx, pdb_id in enumerate(pdb_ids, start=1):
        if not candidate_store.has_candidates(pdb_id):
            continue

        data = dataset.get(dataset._pdb_to_idx[pdb_id])
        gt_pos = data["ligand_atom"].pos.clone()
        poses, meta = candidate_store.load_all_candidates(pdb_id)
        if args.limit_candidates_per_complex > 0:
            poses = poses[: args.limit_candidates_per_complex]
            if "candidate_scores" in meta:
                meta["candidate_scores"] = meta["candidate_scores"][: args.limit_candidates_per_complex]

        if poses.numel() == 0:
            continue

        logger.info(f"[{idx}/{len(pdb_ids)}] Evaluating {pdb_id} with {poses.size(0)} candidates")
        candidate_scores_raw = meta.get("candidate_scores")
        raw_scores_available = False
        if candidate_scores_raw is not None and len(candidate_scores_raw) >= poses.size(0):
            raw_score_array = np.asarray(candidate_scores_raw[: poses.size(0)], dtype=float)
            raw_scores_available = bool(np.isfinite(raw_score_array).all())
        if raw_scores_available:
            raw_rank_eligible += 1
            raw_rank_idx = np.argsort(raw_score_array)
        else:
            raw_rank_idx = None

        rmsd_before_list: list[float] = []
        rmsd_after_list: list[float] = []
        refined_affinity_scores: list[float] = []

        for cand_idx in range(int(poses.size(0))):
            candidate_pos = poses[cand_idx].to(dtype=gt_pos.dtype)
            rmsd_before = torch.sqrt(((candidate_pos - gt_pos).pow(2).sum(dim=-1)).mean()).item()
            rmsd_before_list.append(float(rmsd_before))
            all_candidate_rmsd_before.append(float(rmsd_before))

            score_t = torch.ones(1, device=device, dtype=gt_pos.dtype)
            if raw_rank_idx is not None:
                raw_batch = collator.collate([data.clone()]).to(device)
                raw_batch["ligand_atom"].pos = candidate_pos.to(device)
                raw_out = model(raw_batch, score_t)
                raw_clash = raw_out.get("steric_clash_batch")
                if raw_clash is not None:
                    clash_before_flags.append(float(raw_clash.view(-1)[0].item() > 0.0))
            else:
                raw_batch = None
                raw_out = None

            infer_batch = collator.collate([data.clone()]).to(device)
            infer_batch["ligand_atom"].pos = candidate_pos.to(device)
            final_pos, _ = matcher.ode_solve(
                model=model,
                data=infer_batch,
                steps=args.refine_steps,
                method="euler",
                store_trajectory=False,
            )
            rmsd_after = torch.sqrt(((final_pos.detach().cpu() - gt_pos).pow(2).sum(dim=-1)).mean()).item()
            rmsd_after_list.append(float(rmsd_after))
            all_candidate_rmsd_after.append(float(rmsd_after))

            score_batch = infer_batch.clone()
            score_batch["ligand_atom"].pos = final_pos
            score_out = model(score_batch, score_t)
            refined_affinity_scores.append(float(score_out["binding_affinity"].view(-1)[0].item()))
            refined_clash = score_out.get("steric_clash_batch")
            if refined_clash is not None:
                clash_after_flags.append(float(refined_clash.view(-1)[0].item() > 0.0))

            del raw_batch, raw_out, infer_batch, score_batch, score_out, score_t, final_pos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oracle_best_before.append(float(np.min(rmsd_before_list)))
        oracle_best_after.append(float(np.min(rmsd_after_list)))

        refined_rank_idx = np.argsort(np.asarray(refined_affinity_scores, dtype=float))[::-1]
        refined_top1_rmsd.append(float(rmsd_after_list[int(refined_rank_idx[0])]))

        for k in topk_values:
            refined_topk_best[k].append(float(np.min(np.asarray(rmsd_after_list)[refined_rank_idx[:k]])))
            if raw_rank_idx is not None:
                raw_topk_best[k].append(float(np.min(np.asarray(rmsd_before_list)[raw_rank_idx[:k]])))

        if raw_rank_idx is not None:
            raw_top1_rmsd.append(float(rmsd_before_list[int(raw_rank_idx[0])]))

        evaluated_complexes += 1

    summary: dict[str, Any] = {
        "num_complexes": int(evaluated_complexes),
        "num_complexes_with_raw_scores": int(raw_rank_eligible),
        "candidate_pool_has_2a": topk_metrics(oracle_best_before, 2.0),
        "candidate_pool_has_5a": topk_metrics(oracle_best_before, 5.0),
        "oracle_after_refine_2a": topk_metrics(oracle_best_after, 2.0),
        "oracle_after_refine_5a": topk_metrics(oracle_best_after, 5.0),
        "mean_best_rmsd_in_pool_before": float(np.mean(oracle_best_before)) if oracle_best_before else float("inf"),
        "mean_best_rmsd_in_pool_after": float(np.mean(oracle_best_after)) if oracle_best_after else float("inf"),
        "mean_candidate_rmsd_before": float(np.mean(all_candidate_rmsd_before)) if all_candidate_rmsd_before else float("inf"),
        "mean_candidate_rmsd_after": float(np.mean(all_candidate_rmsd_after)) if all_candidate_rmsd_after else float("inf"),
        "mean_rmsd_improvement_after_refine": (
            float(np.mean(all_candidate_rmsd_before) - np.mean(all_candidate_rmsd_after))
            if all_candidate_rmsd_before and all_candidate_rmsd_after
            else 0.0
        ),
        "clash_rate_before": float(np.mean(clash_before_flags) * 100.0) if clash_before_flags else 0.0,
        "clash_rate_after": float(np.mean(clash_after_flags) * 100.0) if clash_after_flags else 0.0,
    }

    if raw_top1_rmsd:
        summary.update(
            {
                "top1_from_raw_candidates_2a": topk_metrics(raw_top1_rmsd, 2.0),
                "top1_from_raw_candidates_5a": topk_metrics(raw_top1_rmsd, 5.0),
                "top1_from_raw_candidates_mean_rmsd": float(np.mean(raw_top1_rmsd)),
            }
        )
        for k, values in raw_topk_best.items():
            if values:
                summary[f"top{k}_from_raw_candidates_2a"] = topk_metrics(values, 2.0)
                summary[f"top{k}_from_raw_candidates_5a"] = topk_metrics(values, 5.0)

    if refined_top1_rmsd:
        summary.update(
            {
                "top1_from_refined_affinity_2a": topk_metrics(refined_top1_rmsd, 2.0),
                "top1_from_refined_affinity_5a": topk_metrics(refined_top1_rmsd, 5.0),
                "top1_from_refined_affinity_mean_rmsd": float(np.mean(refined_top1_rmsd)),
            }
        )
        for k, values in refined_topk_best.items():
            if values:
                summary[f"top{k}_from_refined_affinity_2a"] = topk_metrics(values, 2.0)
                summary[f"top{k}_from_refined_affinity_5a"] = topk_metrics(values, 5.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_refinement_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    logger.info(f"Saved candidate refinement summary to {report_path}")
    logger.info(f"[Candidate Refinement Summary] {summary}")


if __name__ == "__main__":
    main()