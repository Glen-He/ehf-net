"""
构建项目可直接读取的 candidate cache。

支持两种来源：
1. external: 从外部 docking 输出整理候选构象；
2. random_perturb: 基于 GT pose 随机扰动生成候选构象。

输出目录约定：
data_root/candidates/<source>/<pdb_id>/
    poses.pt
    meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from ehfnet.datasets.candidate_labels import compute_candidate_rmsd
from ehfnet.datasets.pdbbind import PDBBindDataset, load_index
from ehfnet.datasets.prepare import load_ligand
from ehfnet.graph import GraphCollator
from ehfnet.training.flow_matcher import ConditionalFlowMatcher


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate cache for refinement training/evaluation")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing cleaned/")
    parser.add_argument("--index_file", type=str, required=True, help="Index CSV with pdb_id column")
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="external",
        choices=["external", "random_perturb"],
        help="Candidate generation mode",
    )
    parser.add_argument("--candidate_input_root", type=str, default=None, help="Root of raw candidate pose files")
    parser.add_argument("--candidate_source", type=str, default="vina", help="Candidate source name")
    parser.add_argument("--output_root", type=str, default=None, help="Output candidate cache root, default=data_root/candidates")
    parser.add_argument("--max_candidates", type=int, default=0, help="Max candidates per complex, 0 means all")
    parser.add_argument("--num_random_candidates", type=int, default=10, help="Number of random-perturb candidates per complex")
    parser.add_argument("--warmup_epoch", type=int, default=20, help="Curriculum stage used for random perturb generation")
    parser.add_argument("--pocket_radius", type=float, default=12.0, help="Dataset pocket radius when loading processed graphs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached complexes")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def find_candidate_pose_file(candidate_dir: Path, pdb_id: str) -> Path | None:
    candidates = [
        candidate_dir / "poses.sdf",
        candidate_dir / "poses.mol2",
        candidate_dir / f"{pdb_id}_poses.sdf",
        candidate_dir / f"{pdb_id}_poses.mol2",
        candidate_dir / f"{pdb_id}_decoys.mol2",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def iter_mol2_blocks(mol2_path: Path) -> Iterable[str]:
    text = mol2_path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split("@<TRIPOS>MOLECULE")
    for chunk in chunks[1:]:
        yield "@<TRIPOS>MOLECULE" + chunk


def load_candidate_mols(candidate_path: Path) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    if candidate_path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(candidate_path), sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol.UpdatePropertyCache(strict=False)
            mol = Chem.RemoveHs(mol)
            if mol.GetNumConformers() > 0:
                mols.append(mol)
    elif candidate_path.suffix.lower() == ".mol2":
        for block in iter_mol2_blocks(candidate_path):
            mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol.UpdatePropertyCache(strict=False)
            mol = Chem.RemoveHs(mol)
            if mol.GetNumConformers() > 0:
                mols.append(mol)
    else:
        raise ValueError(f"Unsupported candidate file format: {candidate_path}")
    return mols


def find_atom_mapping(gt_mol: Chem.Mol, candidate_mol: Chem.Mol) -> list[int] | None:
    if gt_mol.GetNumAtoms() != candidate_mol.GetNumAtoms():
        return None

    match = candidate_mol.GetSubstructMatch(gt_mol, useChirality=False)
    if len(match) == gt_mol.GetNumAtoms():
        return list(match)

    reverse_match = gt_mol.GetSubstructMatch(candidate_mol, useChirality=False)
    if len(reverse_match) == gt_mol.GetNumAtoms():
        inverse = [0] * gt_mol.GetNumAtoms()
        for cand_idx, gt_idx in enumerate(reverse_match):
            inverse[gt_idx] = cand_idx
        return inverse

    return None


def extract_candidate_positions(candidate_mol: Chem.Mol, gt_mol: Chem.Mol) -> torch.Tensor | None:
    mapping = find_atom_mapping(gt_mol, candidate_mol)
    if mapping is None:
        return None

    conf = candidate_mol.GetConformer()
    positions = []
    for candidate_atom_idx in mapping:
        pos = conf.GetAtomPosition(int(candidate_atom_idx))
        positions.append([pos.x, pos.y, pos.z])
    return torch.tensor(positions, dtype=torch.float32)


def extract_candidate_score(candidate_mol: Chem.Mol) -> float | None:
    score_keys = [
        "vina_score",
        "minimizedAffinity",
        "docking_score",
        "score",
        "CNNscore",
        "Affinity",
    ]
    for key in score_keys:
        if candidate_mol.HasProp(key):
            try:
                return float(candidate_mol.GetProp(key))
            except Exception:
                continue
    return None


def build_random_candidates(
    *,
    dataset: PDBBindDataset,
    collator: GraphCollator,
    matcher: ConditionalFlowMatcher,
    pdb_id: str,
    num_random_candidates: int,
    warmup_epoch: int,
) -> tuple[torch.Tensor, list[float | None]]:
    data = dataset.get(dataset._pdb_to_idx[pdb_id])
    batch = collator.collate([data])
    gt_pos = batch["ligand_atom"].pos.clone().to(dtype=torch.float32)
    poses: list[torch.Tensor] = []

    torsion_indices = getattr(batch, "torsion_indices", None)
    torsion_moving_mask = getattr(batch, "torsion_moving_mask", None)
    if torsion_indices is None:
        torsion_indices = torch.empty((0, 4), dtype=torch.long, device=gt_pos.device)
    if torsion_moving_mask is None:
        torsion_moving_mask = torch.empty((0, gt_pos.size(0)), dtype=torch.bool, device=gt_pos.device)

    lig_batch = getattr(batch["ligand_atom"], "batch", None)
    if lig_batch is None:
        lig_batch = torch.zeros(gt_pos.size(0), dtype=torch.long, device=gt_pos.device)

    masses = batch["ligand_atom"].masses
    B = int(lig_batch.max().item()) + 1 if lig_batch.numel() > 0 else 1

    for _ in range(num_random_candidates):
        pose = matcher._generate_random_pose(
            x_ref=gt_pos,
            batch=lig_batch,
            B=B,
            masses=masses,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
            epoch=warmup_epoch,
        )
        poses.append(pose.cpu())

    return torch.stack(poses, dim=0), [None] * len(poses)


def main() -> None:
    args = parse_args()
    setup_logging()

    data_root = Path(args.data_root).resolve()
    candidate_input_root = Path(args.candidate_input_root).resolve() if args.candidate_input_root else None
    output_root = Path(args.output_root).resolve() if args.output_root else data_root / "candidates"
    source_output_root = output_root / args.candidate_source
    source_output_root.mkdir(parents=True, exist_ok=True)

    index_df = load_index(args.index_file)
    dataset = None
    collator = None
    matcher = None
    if args.generation_mode == "random_perturb":
        dataset = PDBBindDataset(
            root=str(data_root),
            index_file=args.index_file,
            esm_root=None,
            esm="auto",
            esm_dim=960,
            pocket_radius=args.pocket_radius,
            interaction_profile="full",
        )
        collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
        matcher = ConditionalFlowMatcher(warmup_epochs=max(1, args.warmup_epoch))

    success = 0
    skipped = 0

    for _, row in index_df.iterrows():
        pdb_id = str(row["pdb_id"]).lower()
        gt_dir = data_root / "cleaned" / pdb_id
        if not gt_dir.exists():
            skipped += 1
            continue

        candidate_dir = candidate_input_root / pdb_id if candidate_input_root is not None else None
        if args.generation_mode == "external" and candidate_dir is not None and not candidate_dir.exists():
            skipped += 1
            continue

        output_dir = source_output_root / pdb_id
        poses_out = output_dir / "poses.pt"
        meta_out = output_dir / "meta.json"
        if not args.overwrite and poses_out.exists() and meta_out.exists():
            skipped += 1
            continue

        gt_ligand_path = gt_dir / f"{pdb_id}_ligand.sdf"
        if not gt_ligand_path.exists():
            gt_ligand_path = gt_dir / f"{pdb_id}_ligand.mol2"
        if not gt_ligand_path.exists():
            logger.warning(f"Skipping {pdb_id}: missing GT ligand file")
            skipped += 1
            continue

        try:
            gt_mol = load_ligand(str(gt_ligand_path))
            gt_conf = gt_mol.GetConformer()
            gt_pos = torch.tensor(gt_conf.GetPositions(), dtype=torch.float32)

            if args.generation_mode == "external":
                candidate_pose_path = find_candidate_pose_file(candidate_dir, pdb_id) if candidate_dir is not None else None
                if candidate_pose_path is None:
                    logger.warning(f"Skipping {pdb_id}: no candidate pose file found")
                    skipped += 1
                    continue

                candidate_mols = load_candidate_mols(candidate_pose_path)
                if args.max_candidates > 0:
                    candidate_mols = candidate_mols[: args.max_candidates]

                candidate_positions: list[torch.Tensor] = []
                candidate_scores: list[float | None] = []

                for candidate_mol in candidate_mols:
                    pos = extract_candidate_positions(candidate_mol, gt_mol)
                    if pos is None:
                        continue
                    if pos.shape != gt_pos.shape:
                        continue
                    candidate_positions.append(pos)
                    candidate_scores.append(extract_candidate_score(candidate_mol))
                input_pose_file = str(candidate_pose_path)
            else:
                assert dataset is not None and collator is not None and matcher is not None
                num_random_candidates = args.max_candidates if args.max_candidates > 0 else args.num_random_candidates
                poses_tensor, candidate_scores = build_random_candidates(
                    dataset=dataset,
                    collator=collator,
                    matcher=matcher,
                    pdb_id=pdb_id,
                    num_random_candidates=num_random_candidates,
                    warmup_epoch=args.warmup_epoch,
                )
                candidate_positions = [poses_tensor[i] for i in range(poses_tensor.size(0))]
                input_pose_file = "generated:random_perturb"

            if not candidate_positions:
                logger.warning(f"Skipping {pdb_id}: no valid candidates after atom alignment")
                skipped += 1
                continue

            poses_tensor = torch.stack(candidate_positions, dim=0)
            rmsd = compute_candidate_rmsd(poses_tensor, gt_pos)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(poses_tensor, poses_out)

            meta = {
                "pdb_id": pdb_id,
                "candidate_source": args.candidate_source,
                "num_candidates": int(poses_tensor.size(0)),
                "candidate_scores": [None if s is None else float(s) for s in candidate_scores],
                "candidate_rmsd": [float(x) for x in rmsd.tolist()],
                "has_near_native_candidate": bool((rmsd < 2.0).any().item()),
                "has_5a_candidate": bool((rmsd < 5.0).any().item()),
                "input_pose_file": input_pose_file,
                "generation_mode": args.generation_mode,
            }
            with meta_out.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)

            success += 1
        except Exception as exc:
            logger.warning(f"Skipping {pdb_id}: {exc}")
            skipped += 1

    logger.info(
        f"Candidate cache build finished: success={success}, skipped={skipped}, output_root={source_output_root}"
    )


if __name__ == "__main__":
    main()