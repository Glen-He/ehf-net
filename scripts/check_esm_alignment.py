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

from ehfnet.encoders.esm_embedding import load_or_compute_esm_embeddings
from ehfnet.encoders.protein_segments import (
    _residue_chain_tags,
    segment_residues_by_continuity,
)
from ehfnet.datasets.prepare import get_esm_model


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("check_esm_alignment")


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
    expected_ixs = [int(res.ix) for seg in segments for res in seg.residues]

    if len(set(expected_ixs)) != len(expected_ixs):
        raise RuntimeError("Duplicate residue.ix detected inside protein residues.")

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
    for seg in segments:
        for local_idx, res in enumerate(seg.residues):
            res_ix = int(res.ix)
            segid, chain_id = _residue_chain_tags(res)
            emb = embeddings.get(res_ix)
            rows.append(
                {
                    "segment_key": seg.key,
                    "segment_local_index": local_idx,
                    "res_ix": res_ix,
                    "segid": segid,
                    "chain_id": chain_id,
                    "resid": int(getattr(res, "resid", -1)),
                    "resname": str(getattr(res, "resname", "")),
                    "has_embedding": int(emb is not None),
                    "embedding_dim": int(emb.shape[0]) if emb is not None else 0,
                }
            )

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

    if missing_ixs or extra_ixs:
        raise SystemExit(2)

    logger.info("ESM alignment check passed.")


if __name__ == "__main__":
    main()
