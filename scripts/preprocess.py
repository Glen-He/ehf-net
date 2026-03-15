"""
统一预处理管理器

提供 build、clean、stats 子命令，用于图缓存构建、缓存清理、数据集统计。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 与 ehfnet.datasets.protein_ligand 中版本标签一致
ESM_CACHE_VERSION_TAG = "esm_chainseg"
GRAPH_CACHE_DIRNAME = "cache"


def _geometry_pre_filter(data):
    """几何合理性检查：配体原子间最小距离 < 0.5 Å 则过滤。"""
    if "ligand_atom" not in data or not hasattr(data["ligand_atom"], "pos"):
        return True
    lig_pos = data["ligand_atom"].pos
    if lig_pos.shape[0] <= 1:
        return True
    import torch
    dist_mat = torch.cdist(lig_pos, lig_pos, p=2)
    dist_mat = dist_mat + torch.eye(dist_mat.shape[0], device=dist_mat.device) * 1000.0
    min_dist = dist_mat.min().item()
    if min_dist < 0.5:
        logger.warning(
            "Filtering sample with unreasonable geometry: min atom distance = %.3f Å",
            min_dist,
        )
        return False
    return True


def cmd_build(args: argparse.Namespace) -> None:
    """构建图缓存 + ESM + 几何检查。"""
    from ehfnet.datasets.protein_ligand import ProteinLigandDataset

    data_root = Path(args.data_root)
    index_file = str(data_root / "index.csv")
    if not data_root.exists():
        logger.error("Data root not found: %s", data_root)
        return

    if args.force_rebuild:
        logger.info("Force rebuild: cleaning graph and ESM cache first...")
        _clean_target(data_root, target="graph", dry_run=False)
        _clean_target(data_root, target="esm", dry_run=False)

    dataset = ProteinLigandDataset(
        root=str(data_root),
        index_file=index_file,
        esm_root=args.esm_root or None,
        esm="auto",
        force_reprocess=args.force_rebuild,
        pre_filter=_geometry_pre_filter,
    )
    dataset.process()
    logger.info("Build complete.")


def _clean_target(data_root: Path, target: str, dry_run: bool) -> tuple[int, float]:
    """清理指定类型的缓存。返回 (删除文件数, 释放空间 MB)。"""
    root = Path(data_root)
    cleaned_dir = root / "cleaned"
    count = 0
    size_bytes = 0

    if target == "graph":
        for name in (GRAPH_CACHE_DIRNAME,):
            d = root / name
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        size_bytes += f.stat().st_size
                        count += 1
                if not dry_run:
                    shutil.rmtree(d)
                    logger.info("Deleted %s", d)
                else:
                    logger.info("[DRY RUN] Would delete %s", d)

        # normalization_stats
        stats_file = root / "normalization_stats.pt"
        if stats_file.exists():
            size_bytes += stats_file.stat().st_size
            count += 1
            if not dry_run:
                stats_file.unlink()
                logger.info("Deleted %s", stats_file)
            else:
                logger.info("[DRY RUN] Would delete %s", stats_file)

    elif target == "esm":
        for npz in cleaned_dir.rglob("*.npz"):
            if npz.name.endswith(f"_{ESM_CACHE_VERSION_TAG}.npz"):
                size_bytes += npz.stat().st_size
                count += 1
                if not dry_run:
                    npz.unlink()
                else:
                    logger.info("[DRY RUN] Would delete %s", npz)
        if not cleaned_dir.exists():
            logger.warning("Cleaned dir not found: %s", cleaned_dir)

    size_mb = size_bytes / (1024 * 1024)
    return count, size_mb


def cmd_clean(args: argparse.Namespace) -> None:
    """清理指定类型的缓存。"""
    data_root = Path(args.data_root)
    target = args.target
    dry_run = args.dry_run

    if target == "all":
        for t in ("graph", "esm"):
            c, m = _clean_target(data_root, t, dry_run)
            logger.info("Target %s: %d items, %.2f MB", t, c, m)
    else:
        c, m = _clean_target(data_root, target, dry_run)
        logger.info("Cleaned %d items, %.2f MB", c, m)


def cmd_stats(args: argparse.Namespace) -> None:
    """计算数据集统计（特征归一化、边距离分布）。"""
    import torch
    from torch_geometric.data import HeteroData

    from ehfnet.datasets.protein_ligand import ProteinLigandDataset

    data_root = Path(args.data_root)
    index_file = str(data_root / "index.csv")
    output_file = args.output or str(data_root / "dataset_profile.json")

    dataset = ProteinLigandDataset(
        root=str(data_root),
        index_file=index_file,
        esm_root=None,
        esm="off",
    )
    processed_dir = Path(dataset.processed_dir)
    if not processed_dir.exists():
        logger.error("Processed dir not found: %s", processed_dir)
        return

    # 特征统计（类似 compute_dataset_stats）
    node_keys = ["ligand_atom", "protein_atom", "protein_residue", "ligand_molecule"]
    stats: dict[str, dict] = {
        k: {"sum": None, "sum_sq": None, "count": 0} for k in node_keys
    }
    stats["affinity"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

    # 边距离统计（用于数据驱动 cutoff）
    edge_distances: dict[str, list[float]] = {}

    dataset._build_valid_index()
    valid_ids = dataset._valid_pdb_ids
    max_samples = args.max_samples if args.max_samples > 0 else len(valid_ids)
    to_process = valid_ids[:max_samples]

    for pdb_id in tqdm(to_process, desc="Computing stats"):
        fp = processed_dir / f"data_{pdb_id}.pt"
        if not fp.exists():
            continue
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            data = data  # type: HeteroData
        except Exception as e:
            logger.warning("Skip %s: %s", pdb_id, e)
            continue

        for k in node_keys:
            if k in data and hasattr(data[k], "x_cont"):
                x = data[k].x_cont
                if stats[k]["sum"] is None:
                    stats[k]["sum"] = x.sum(dim=0).double()
                    stats[k]["sum_sq"] = (x ** 2).sum(dim=0).double()
                else:
                    stats[k]["sum"] = stats[k]["sum"] + x.sum(dim=0).double()
                    stats[k]["sum_sq"] = stats[k]["sum_sq"] + (x ** 2).sum(dim=0).double()
                stats[k]["count"] += x.shape[0]

        if hasattr(data, "y_energy") and data.y_energy is not None:
            y = data.y_energy
            stats["affinity"]["sum"] += float(y.sum().item())
            stats["affinity"]["sum_sq"] += float((y ** 2).sum().item())
            stats["affinity"]["count"] += y.numel()

        # 边距离：统计 ligand_atom <-> protein_atom 等
        if "ligand_atom" in data and "protein_atom" in data:
            lig_pos = data["ligand_atom"].pos
            pro_pos = data["protein_atom"].pos
            if lig_pos.numel() > 0 and pro_pos.numel() > 0:
                dist = torch.cdist(lig_pos, pro_pos, p=2)
                key = "ligand_atom-protein_atom"
                if key not in edge_distances:
                    edge_distances[key] = []
                edge_distances[key].extend(dist.flatten().tolist())

    # 汇总
    profile: dict = {"feature_stats": {}, "edge_distance_p95": {}, "affinity": {}}
    for k in node_keys:
        if stats[k]["count"] > 0 and stats[k]["sum"] is not None:
            mean = (stats[k]["sum"] / stats[k]["count"]).float()
            var = (stats[k]["sum_sq"] / stats[k]["count"]) - (stats[k]["sum"] / stats[k]["count"]) ** 2
            std = torch.sqrt(torch.clamp(var.float(), min=1e-6))
            profile["feature_stats"][k] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
    if stats["affinity"]["count"] > 0:
        mean = stats["affinity"]["sum"] / stats["affinity"]["count"]
        var = stats["affinity"]["sum_sq"] / stats["affinity"]["count"] - mean ** 2
        std = max(0.0, var) ** 0.5
        profile["affinity"] = {"mean": float(mean), "std": float(std)}

    for key, dists in edge_distances.items():
        if dists:
            import numpy as np
            p95 = float(np.percentile(dists, 95))
            profile["edge_distance_p95"][key] = p95
            # 建议 cutoff: 1.2 * p95，限制在合理范围
            suggested = min(14.0, max(6.0, 1.2 * p95))
            profile.setdefault("suggested_cutoffs", {})[key] = suggested

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info("Stats saved to %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EHFNet unified preprocess manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build graph cache + ESM + geometry check")
    p_build.add_argument("--data-root", type=str, required=True, help="数据根目录，须含 index.csv，如 data/processed/hiqbind")
    p_build.add_argument("--esm-root", type=str, default=None)
    p_build.add_argument("--force-rebuild", action="store_true")
    p_build.set_defaults(func=cmd_build)

    # clean
    p_clean = subparsers.add_parser("clean", help="Clean cache")
    p_clean.add_argument("--data-root", type=str, required=True, help="e.g. data/processed/hiqbind")
    p_clean.add_argument(
        "--target",
        choices=["graph", "esm", "all"],
        required=True,
        help="graph=图缓存目录, esm=ESM npz 缓存, all=两者",
    )
    p_clean.add_argument("--dry-run", action="store_true")
    p_clean.set_defaults(func=cmd_clean)

    # stats
    p_stats = subparsers.add_parser("stats", help="Compute dataset statistics")
    p_stats.add_argument("--data-root", type=str, required=True, help="数据根目录，须含 index.csv")
    p_stats.add_argument("--output", "-o", type=str, default=None)
    p_stats.add_argument("--max-samples", type=int, default=0, help="0=all")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
