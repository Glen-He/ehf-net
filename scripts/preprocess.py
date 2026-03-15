"""
预处理命令入口。

负责构建图缓存、清理缓存与统计数据集信息，
并复用训练配置中的共享运行参数。
"""


import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import cast

from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from ehfnet.contracts import ESM_CACHE_VERSION_TAG, GRAPH_CACHE_DIRNAME
from ehfnet.runtime import (
    configure_text_logging,
    get_configured_device,
    get_configured_smoke,
    load_train_defaults,
    resolve_interaction_profile,
)

logger = logging.getLogger(__name__)


def _geometry_pre_filter(data, *, min_atom_distance: float) -> bool:
    """
    几何合理性检查：配体原子间最小距离低于阈值则过滤。

    Args:
        data: 当前待检查的图样本对象。
        min_atom_distance: 允许保留样本的最小配体原子间距阈值（Å）。

    Returns:
        bool: 返回样本是否通过几何合理性筛选。
    """
    if "ligand_atom" not in data or not hasattr(data["ligand_atom"], "pos"):
        return True
    lig_pos = data["ligand_atom"].pos
    if lig_pos.shape[0] <= 1:
        return True
    import torch
    dist_mat = torch.cdist(lig_pos, lig_pos, p=2)
    dist_mat = dist_mat + torch.eye(dist_mat.shape[0], device=dist_mat.device) * 1000.0
    min_dist = dist_mat.min().item()
    if min_dist < min_atom_distance:
        logger.warning(
            "Filtering sample with unreasonable geometry: min atom distance = %.3f Å",
            min_dist,
        )
        return False
    return True


def _suggest_distance_cutoff(
    *,
    percentile_value: float,
    scale: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """
    根据距离分布统计生成建议 cutoff。

    Args:
        percentile_value: 距离统计分位数对应的原始数值。
        scale: 将分位数放大的比例系数。
        lower_bound: 建议 cutoff 的最小截断值。
        upper_bound: 建议 cutoff 的最大截断值。

    Returns:
        float: 返回裁剪到给定上下界后的建议 cutoff。
    """
    return min(upper_bound, max(lower_bound, scale * percentile_value))


def _configure_logging(command: str, *, smoke: bool) -> str:
    """
    配置预处理脚本的终端与文件日志。

    Args:
        command: 当前执行的预处理子命令名称。
        smoke: 是否使用 smoke 运行分组写入日志。

    Returns:
        str: 返回当前预处理命令对应的日志文件路径。
    """

    log_file, _ = configure_text_logging(
        category="preprocess",
        file_stem=f"preprocess_{command}",
        smoke=smoke,
    )
    logger.info("Logging to %s", log_file)
    logger.info("Smoke log grouping: %s", smoke)
    return str(log_file)


def _build_dataset_kwargs(config_defaults: dict[str, object]) -> dict[str, object]:
    """
    从合并后的默认配置中提取图构建参数。

    Args:
        config_defaults: 已从配置文件加载并展平的默认参数字典。

    Returns:
        dict[str, object]: 返回传给数据集构建流程的共享图参数字典。
    """

    return {
        "esm": str(config_defaults["esm"]),
        "esm_model_name": str(config_defaults["esm_model_name"]),
        "esm_dim": int(config_defaults["esm_dim"]),
        "r_cutoff_intra": float(config_defaults["r_cutoff_intra"]),
        "max_neighbors_intra": int(config_defaults["max_neighbors_intra"]),
        "atom_neighbor_cap": int(config_defaults["atom_neighbor_cap"]),
        "residue_neighbor_cap": int(config_defaults["residue_neighbor_cap"]),
        "residue_radius_scale": float(config_defaults["residue_radius_scale"]),
        "residue_radius_bias": float(config_defaults["residue_radius_bias"]),
        "ligand_atom_fallback_k": int(config_defaults["ligand_atom_fallback_k"]),
        "protein_atom_fallback_k": int(config_defaults["protein_atom_fallback_k"]),
        "protein_residue_fallback_k": int(
            config_defaults["protein_residue_fallback_k"]
        ),
        "interaction_profile": resolve_interaction_profile(
            ablation_mode=str(config_defaults["ablation_mode"])
        ),
    }


def cmd_build(args: argparse.Namespace) -> None:
    """
    构建图缓存与 ESM 缓存。

    负责遍历数据集执行图样本预处理、ESM 计算和几何校验，
    是训练前准备缓存数据的主要命令实现。

    Args:
        args: 命令行解析后的参数对象，包含当前命令所需的配置。
    """
    from ehfnet.data.preprocess import configure_hf_cache_env, resolve_esm_device
    from ehfnet.runtime import build_dataset

    data_root = Path(args.data_root)
    index_file = str(data_root / "index.csv")
    if not data_root.exists():
        logger.error("Data root not found: %s", data_root)
        return

    if args.force_rebuild:
        logger.info("Force rebuild: cleaning graph and ESM cache first...")
        _clean_target(data_root, target="graph", dry_run=False)
        _clean_target(data_root, target="esm", dry_run=False)

    _, hub_cache, cache_source = configure_hf_cache_env(project_root=PROJECT_ROOT)
    resolved_device = resolve_esm_device(args.device)
    logger.info("Using ESM device: %s", resolved_device)
    logger.info("Using HuggingFace hub cache: %s (%s)", hub_cache, cache_source)

    dataset = build_dataset(
        root=str(data_root),
        index_file=index_file,
        esm_root=args.esm_root or None,
        esm_device=str(resolved_device),
        **args.dataset_build_kwargs,
        force_reprocess=args.force_rebuild,
        pre_filter=lambda data: _geometry_pre_filter(
            data,
            min_atom_distance=float(args.geometry_min_atom_distance),
        ),
    )
    dataset.process()
    logger.info("Build complete.")


def _clean_target(
    data_root: Path,
    *,
    target: str,
    dry_run: bool,
) -> tuple[int, float]:
    """
    清理指定类型的缓存。返回 (删除文件数, 释放空间 MB)。

    Args:
        data_root: 数据集根目录。
        target: 待清理的缓存类型，只支持 `graph`、`esm` 或 `logs`。
        dry_run: 若为 `True`，仅统计并打印待删除内容，不实际删除。

    Returns:
        tuple[int, float]: 返回删除文件数量与预计释放的存储空间（MB）。
    """
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

    elif target == "esm":
        for npz in cleaned_dir.rglob("*.npz"):
            if ESM_CACHE_VERSION_TAG in npz.stem:
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
    """
    清理预处理缓存。

    按目标类型删除图缓存、ESM 缓存或两者，
    用于在配置或预处理逻辑变化后强制重建缓存。

    Args:
        args: 命令行解析后的参数对象，包含当前命令所需的配置。
    """
    data_root = Path(args.data_root)
    target = args.target
    dry_run = args.dry_run

    if target == "all":
        for t in ("graph", "esm"):
            c, m = _clean_target(data_root, target=t, dry_run=dry_run)
            logger.info("Target %s: %d items, %.2f MB", t, c, m)
    else:
        c, m = _clean_target(data_root, target=target, dry_run=dry_run)
        logger.info("Cleaned %d items, %.2f MB", c, m)


def cmd_stats(args: argparse.Namespace) -> None:
    """
    统计数据集特征信息。

    负责遍历缓存样本并计算归一化和边距离等统计量，
    供训练阶段的标准化和数据检查流程使用。

    Args:
        args: 命令行解析后的参数对象，包含当前命令所需的配置。
    """
    import numpy as np
    import torch

    from ehfnet.runtime import build_dataset

    data_root = Path(args.data_root)
    index_file = str(data_root / "index.csv")
    output_file = args.output or str(data_root / "dataset_profile.json")

    dataset = build_dataset(
        root=str(data_root),
        index_file=index_file,
        esm_root=None,
        esm="off",
        **args.dataset_build_kwargs,
    )
    processed_dir = Path(dataset.processed_dir)
    if not processed_dir.exists():
        logger.error("Processed dir not found: %s", processed_dir)
        return

    # 统计连续特征的均值与标准差。
    node_keys = ["ligand_atom", "protein_atom", "protein_residue", "ligand_molecule"]
    stats: dict[str, dict] = {
        k: {"sum": None, "sum_sq": None, "count": 0} for k in node_keys
    }
    stats["affinity"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

    # 统计关键跨图距离分布，用于生成建议截断半径。
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
            data = cast(object, torch.load(fp, map_location="cpu", weights_only=False))
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

        # 统计 `ligand_atom` 到 `protein_atom` 的全局距离分布。
        if "ligand_atom" in data and "protein_atom" in data:
            lig_pos = data["ligand_atom"].pos
            pro_pos = data["protein_atom"].pos
            if lig_pos.numel() > 0 and pro_pos.numel() > 0:
                dist = torch.cdist(lig_pos, pro_pos, p=2)
                key = "ligand_atom-protein_atom"
                if key not in edge_distances:
                    edge_distances[key] = []
                edge_distances[key].extend(dist.flatten().tolist())

    # 汇总统计结果并生成可写入 JSON 的配置建议。
    profile: dict = {
        "feature_stats": {},
        "edge_distance_reference": {},
        "edge_distance_percentile": float(args.stats_cutoff_percentile),
        "affinity": {},
    }
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
            reference_distance = float(
                np.percentile(dists, float(args.stats_cutoff_percentile))
            )
            profile["edge_distance_reference"][key] = reference_distance
            suggested = _suggest_distance_cutoff(
                percentile_value=reference_distance,
                scale=float(args.stats_cutoff_scale),
                lower_bound=float(args.stats_cutoff_min),
                upper_bound=float(args.stats_cutoff_max),
            )
            profile.setdefault("suggested_cutoffs", {})[key] = suggested

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info("Stats saved to %s", output_file)


def main() -> None:
    """
    预处理入口函数。

    负责组装子命令参数、读取共享配置并初始化日志，
    随后分派到缓存构建、缓存清理或数据统计流程。
    """
    default_config_path = PROJECT_ROOT / "configs" / "train.toml"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="Path to the training config file used to read the shared device default",
    )
    pre_args, _ = pre_parser.parse_known_args()
    configured_device = get_configured_device(
        config_path=pre_args.config,
        project_root=PROJECT_ROOT,
    )
    configured_smoke = get_configured_smoke(
        config_path=pre_args.config,
        project_root=PROJECT_ROOT,
    )
    config_defaults = load_train_defaults(
        config_path=pre_args.config,
        project_root=PROJECT_ROOT,
    )

    parser = argparse.ArgumentParser(
        description="EHFNet unified preprocess manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=configured_smoke,
        help="Write text logs under logs/smoke/... for easier smoke-run cleanup.",
    )
    parser.add_argument(
        "--no-smoke",
        dest="smoke",
        action="store_false",
        help="Disable smoke log grouping and use the default logs/... directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 构建缓存。
    p_build = subparsers.add_parser("build", help="Build graph cache + ESM + geometry check")
    p_build.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Processed data root containing index.csv, e.g. data/processed/hiqbind",
    )
    p_build.add_argument("--esm-root", type=str, default=None)
    p_build.add_argument(
        "--device",
        type=str,
        default=configured_device,
        help="ESM device, e.g. cuda:0 or cpu",
    )
    p_build.add_argument("--force-rebuild", action="store_true")
    p_build.set_defaults(func=cmd_build)

    # 清理缓存。
    p_clean = subparsers.add_parser("clean", help="Clean cache")
    p_clean.add_argument("--data-root", type=str, required=True, help="e.g. data/processed/hiqbind")
    p_clean.add_argument(
        "--target",
        choices=["graph", "esm", "all"],
        required=True,
        help="graph=graph cache directory, esm=ESM npz cache, all=both",
    )
    p_clean.add_argument("--dry-run", action="store_true")
    p_clean.set_defaults(func=cmd_clean)

    # 统计数据集。
    p_stats = subparsers.add_parser("stats", help="Compute dataset statistics")
    p_stats.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Processed data root containing index.csv",
    )
    p_stats.add_argument("--output", "-o", type=str, default=None)
    p_stats.add_argument("--max-samples", type=int, default=0, help="0=all")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.dataset_build_kwargs = _build_dataset_kwargs(config_defaults)
    args.geometry_min_atom_distance = float(config_defaults["geometry_min_atom_distance"])
    args.stats_cutoff_percentile = float(config_defaults["stats_cutoff_percentile"])
    args.stats_cutoff_scale = float(config_defaults["stats_cutoff_scale"])
    args.stats_cutoff_min = float(config_defaults["stats_cutoff_min"])
    args.stats_cutoff_max = float(config_defaults["stats_cutoff_max"])
    _configure_logging(args.command, smoke=bool(args.smoke))
    args.func(args)


if __name__ == "__main__":
    main()
