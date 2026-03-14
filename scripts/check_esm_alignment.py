#!/usr/bin/env python3
"""
检查 ESM embedding 与蛋白 residue 的分段和一一对应关系。

用途：
1. 验证当前 PDB 会被切成哪些连续链段
2. 验证 ESM cache / 现算 embedding 是否覆盖全部 protein residues
3. 验证每个 residue.ix 是否恰好对应一个 embedding
4. 可选导出逐残基对齐明细 CSV
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.core.groups import Residue as MDAResidue

from ehfnet.encoders.chemistry import resolve_esm_residue_type
from ehfnet.encoders.esm_embedding import load_or_compute_esm_embeddings
from ehfnet.encoders.protein_segments import (
    _residue_chain_tags,
    continuity_break_reason,
    segment_residues_by_continuity,
)
from ehfnet.datasets.prepare import get_esm_model


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("check_esm_alignment")


def missing_backbone_atoms(res: MDAResidue) -> list[str]:
    atom_names = {str(atom.name).strip().upper() for atom in res.atoms}
    return [name for name in ("N", "CA", "C") if name not in atom_names]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check ESM-residue alignment on a protein PDB.")
    parser.add_argument("--protein", type=str, required=True, help="Protein PDB path.")
    parser.add_argument("--cache", type=str, default=None, help="Path to ESM npz cache.")
    parser.add_argument(
        "--mode",
        type=str,
        default="cache",
        choices=["cache", "compute", "auto"],
        help="cache: 只读 cache；compute: 强制现算；auto: 优先 cache，不存在则现算。",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esmc_300m",
        help="ESM model name when compute/auto needs inference.",
    )
    parser.add_argument(
        "--dump-csv",
        type=str,
        default=None,
        help="Optional path to dump per-residue alignment table.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many residue rows to print in the terminal preview.",
    )
    parser.add_argument(
        "--show-breaks",
        type=int,
        default=20,
        help="How many segment-break rows to print in the terminal preview.",
    )
    parser.add_argument(
        "--allow-unknown-residues",
        action="store_true",
        help="Do not fail when unresolved residues still map to UNK / X.",
    )
    parser.add_argument(
        "--allow-missing-backbone",
        action="store_true",
        help="Do not fail when residues are missing N / CA / C backbone atoms.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    protein_path = Path(args.protein)

    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")

    cache_path = Path(args.cache) if args.cache else None
    if args.mode == "cache" and cache_path is None:
        raise ValueError("--cache is required when --mode=cache")

    universe = mda.Universe(str(protein_path))
    protein_residues = list(universe.select_atoms("protein").residues)
    segments = segment_residues_by_continuity(protein_residues)
    sorted_residues = sorted(protein_residues, key=lambda res: int(res.ix))
    expected_ixs = [int(res.ix) for seg in segments for res in seg.residues]

    if len(set(expected_ixs)) != len(expected_ixs):
        raise RuntimeError("Duplicate residue.ix detected inside protein residues.")

    segment_breaks: list[dict[str, object]] = []
    for prev_res, next_res in zip(sorted_residues, sorted_residues[1:]):
        break_reason = continuity_break_reason(prev_res, next_res)
        if break_reason is None:
            continue
        prev_segid, prev_chain = _residue_chain_tags(prev_res)
        next_segid, next_chain = _residue_chain_tags(next_res)
        segment_breaks.append(
            {
                "prev_ix": int(prev_res.ix),
                "prev_resid": int(getattr(prev_res, "resid", -1)),
                "prev_resname": str(getattr(prev_res, "resname", "")),
                "prev_chain": prev_chain or prev_segid or "-",
                "next_ix": int(next_res.ix),
                "next_resid": int(getattr(next_res, "resid", -1)),
                "next_resname": str(getattr(next_res, "resname", "")),
                "next_chain": next_chain or next_segid or "-",
                "reason": break_reason,
            }
        )

    if args.mode == "cache":
        esm_model = None
        force_recompute = False
        effective_cache = cache_path
    elif args.mode == "compute":
        esm_model = get_esm_model(model_name=args.esm_model)
        force_recompute = True
        effective_cache = cache_path or protein_path.with_suffix(".esm_check_tmp.npz")
    else:
        esm_model = None if (cache_path and cache_path.exists()) else get_esm_model(model_name=args.esm_model)
        force_recompute = False
        effective_cache = cache_path or protein_path.with_suffix(".esm_check_tmp.npz")

    embeddings = load_or_compute_esm_embeddings(
        universe=universe,
        esm_model=esm_model,
        cache_path=effective_cache,
        force_recompute=force_recompute,
    )

    found_ixs = sorted(int(k) for k in embeddings.keys())
    expected_ix_set = set(expected_ixs)
    found_ix_set = set(found_ixs)

    missing_ixs = sorted(expected_ix_set - found_ix_set)
    extra_ixs = sorted(found_ix_set - expected_ix_set)

    logger.info("Protein residues: %d", len(expected_ixs))
    logger.info("Continuous segments: %d", len(segments))
    logger.info("Embedding count: %d", len(found_ixs))

    for seg in segments:
        first = seg.residues[0]
        last = seg.residues[-1]
        segid, chain_id = _residue_chain_tags(first)
        logger.info(
            "Segment %s | len=%d | segid=%s | chain=%s | ix=[%d..%d] | resid=[%s..%s]",
            seg.key,
            len(seg.residues),
            segid or "-",
            chain_id or "-",
            int(first.ix),
            int(last.ix),
            getattr(first, "resid", "?"),
            getattr(last, "resid", "?"),
        )

    if missing_ixs:
        logger.error("Missing embedding ix count: %d", len(missing_ixs))
        logger.error("First missing ix values: %s", missing_ixs[:20])
    else:
        logger.info("No missing residue embeddings.")

    if extra_ixs:
        logger.error("Extra embedding ix count: %d", len(extra_ixs))
        logger.error("First extra ix values: %s", extra_ixs[:20])
    else:
        logger.info("No extra embedding indices.")

    rows: list[dict[str, object]] = []
    unknown_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []
    missing_backbone_rows: list[dict[str, object]] = []
    for seg in segments:
        for local_idx, res in enumerate(seg.residues):
            res_ix = int(res.ix)
            segid, chain_id = _residue_chain_tags(res)
            emb = embeddings.get(res_ix)
            resolution = resolve_esm_residue_type(res.resname)
            missing_backbone = missing_backbone_atoms(res)
            rows.append(
                {
                    "segment_key": seg.key,
                    "segment_local_index": local_idx,
                    "res_ix": res_ix,
                    "segid": segid,
                    "chain_id": chain_id,
                    "resid": int(getattr(res, "resid", -1)),
                    "resname": str(getattr(res, "resname", "")),
                    "esm_resname": resolution.normalized_resname,
                    "esm_resolution": resolution.source,
                    "esm_letter": resolution.residue_type.one_letter,
                    "missing_backbone": ",".join(missing_backbone),
                    "has_embedding": int(emb is not None),
                    "embedding_dim": int(emb.shape[0]) if emb is not None else 0,
                }
            )
            if resolution.source == "unknown":
                unknown_rows.append(rows[-1])
            elif resolution.source == "alias":
                alias_rows.append(rows[-1])
            if missing_backbone:
                missing_backbone_rows.append(rows[-1])

    logger.info("Segment breaks detected: %d", len(segment_breaks))
    break_preview = segment_breaks[: max(0, args.show_breaks)]
    for break_row in break_preview:
        logger.info(
            "  break ix=%s/%s (%s %s) -> ix=%s/%s (%s %s): %s",
            break_row["prev_ix"],
            break_row["prev_resid"],
            break_row["prev_resname"],
            break_row["prev_chain"],
            break_row["next_ix"],
            break_row["next_resid"],
            break_row["next_resname"],
            break_row["next_chain"],
            break_row["reason"],
        )

    if alias_rows:
        logger.warning("Canonicalized non-standard residues count: %d", len(alias_rows))
        for row in alias_rows[:20]:
            logger.warning(
                "  ix=%s resid=%s %s -> %s (%s)",
                row["res_ix"],
                row["resid"],
                row["resname"],
                row["esm_resname"],
                row["segment_key"],
            )
    else:
        logger.info("No canonicalized non-standard residues.")

    if unknown_rows:
        logger.error("Unresolved residues mapped to UNK/X count: %d", len(unknown_rows))
        for row in unknown_rows[:20]:
            logger.error(
                "  ix=%s resid=%s resname=%s segment=%s",
                row["res_ix"],
                row["resid"],
                row["resname"],
                row["segment_key"],
            )
    else:
        logger.info("No unresolved residues mapped to UNK/X.")

    if missing_backbone_rows:
        logger.error("Residues missing backbone atoms (N/CA/C): %d", len(missing_backbone_rows))
        for row in missing_backbone_rows[:20]:
            logger.error(
                "  ix=%s resid=%s %s missing=%s",
                row["res_ix"],
                row["resid"],
                row["resname"],
                row["missing_backbone"],
            )
    else:
        logger.info("All residues contain N/CA/C backbone atoms.")

    if args.dump_csv:
        dump_path = Path(args.dump_csv)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote alignment table to %s", dump_path)

    preview = rows[: max(0, args.show)]
    if preview:
        logger.info("Residue preview:")
        for row in preview:
            logger.info(
                "  %s idx=%s ix=%s resid=%s %s embed=%s dim=%s",
                row["segment_key"],
                row["segment_local_index"],
                row["res_ix"],
                row["resid"],
                row["resname"],
                row["has_embedding"],
                row["embedding_dim"],
            )

    has_unknown_residue_error = bool(unknown_rows) and not args.allow_unknown_residues
    has_missing_backbone_error = bool(missing_backbone_rows) and not args.allow_missing_backbone

    if missing_ixs or extra_ixs or has_unknown_residue_error or has_missing_backbone_error:
        raise SystemExit(2)

    logger.info("ESM alignment check passed.")


if __name__ == "__main__":
    main()
