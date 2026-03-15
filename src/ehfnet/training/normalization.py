"""
归一化统计工具。

负责计算并缓存训练集统计量，
为输入特征标准化提供数据支持。
"""


import hashlib
import logging
import os
from pathlib import Path
from typing import cast

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from ehfnet.data.datasets import ProteinLigandDataset

logger = logging.getLogger(__name__)


def _normalization_cache_path(
    *,
    split_cache_file: str,
    processed_dir: str,
    train_indices: list[int],
) -> Path:
    digest_src = ",".join(str(int(i)) for i in sorted(train_indices))
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
    processed_tag = Path(processed_dir).name
    split_path = Path(split_cache_file)
    return split_path.with_name(
        f"{split_path.stem}_{processed_tag}_{digest}_train_norm.pt"
    )


def _empty_feature_stat(dim: int) -> dict[str, torch.Tensor]:
    return {
        "sum": torch.zeros(dim, dtype=torch.float64),
        "sum_sq": torch.zeros(dim, dtype=torch.float64),
        "count": torch.zeros(dim, dtype=torch.float64),
    }


def _accumulate_feature_block(
    stat: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    missing_mask: torch.Tensor | None = None,
    masked_feature_start: int | None = None,
) -> None:
    if x.numel() == 0:
        return

    x_cpu = x.detach().to(dtype=torch.float64, device="cpu")
    if stat["sum"].numel() != x_cpu.size(1):
        raise ValueError(
            f"Feature dimension mismatch while accumulating stats: expected {stat['sum'].numel()}, got {x_cpu.size(1)}."
        )

    if (
        missing_mask is not None
        and masked_feature_start is not None
        and 0 < int(masked_feature_start) < x_cpu.size(1)
    ):
        mask_cpu = missing_mask.detach().to(device="cpu", dtype=torch.bool)
        split = int(masked_feature_start)
        torsion = x_cpu[:, :split]
        esm = x_cpu[:, split:]

        stat["sum"][:split] += torsion.sum(dim=0)
        stat["sum_sq"][:split] += torsion.pow(2).sum(dim=0)
        stat["count"][:split] += float(x_cpu.size(0))

        valid_mask = ~mask_cpu
        if bool(valid_mask.any()):
            valid_esm = esm[valid_mask]
            stat["sum"][split:] += valid_esm.sum(dim=0)
            stat["sum_sq"][split:] += valid_esm.pow(2).sum(dim=0)
            stat["count"][split:] += float(valid_esm.size(0))
        return

    stat["sum"] += x_cpu.sum(dim=0)
    stat["sum_sq"] += x_cpu.pow(2).sum(dim=0)
    stat["count"] += float(x_cpu.size(0))


def _finalize_feature_stats(
    stat: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    count = stat["count"].clamp_min(1.0)
    mean = stat["sum"] / count
    mean_sq = stat["sum_sq"] / count
    var = (mean_sq - mean.pow(2)).clamp(min=1e-6)

    zero_mask = stat["count"] <= 0
    if bool(zero_mask.any()):
        mean[zero_mask] = 0.0
        var[zero_mask] = 1.0

    return {
        "mean": mean.to(dtype=torch.float32),
        "std": torch.sqrt(var).to(dtype=torch.float32),
    }


def compute_train_split_normalization_stats(
    dataset: ProteinLigandDataset,
    train_indices: list[int],
    *,
    split_cache_file: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    """
    计算训练集归一化统计并缓存到磁盘。

    Args:
        dataset: 参与处理或划分的数据集对象。
        train_indices: 训练集样本索引列表。
        split_cache_file: 数据划分缓存文件路径。

    Returns:
        tuple[dict[str, dict[str, Tensor]], dict[str, float]]: 各节点类型的归一化统计（mean/std）与亲和力统计。
    """
    cache_path = _normalization_cache_path(
        split_cache_file=split_cache_file,
        processed_dir=dataset.processed_dir,
        train_indices=train_indices,
    )
    cache_meta = {
        "processed_dir": os.path.abspath(dataset.processed_dir),
        "index_file": os.path.abspath(dataset.index_file),
        "train_size": int(len(train_indices)),
    }

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and cached.get("metadata") == cache_meta:
            cached_stats = cached.get("stats")
            cached_affinity = cached.get("affinity")
            if isinstance(cached_stats, dict) and isinstance(cached_affinity, dict):
                logger.info("Loaded train-only normalization stats from %s", cache_path)
                return cast(dict[str, dict[str, torch.Tensor]], cached_stats), cast(dict[str, float], cached_affinity)

    sample_dim = cast(HeteroData, torch.load(
        os.path.join(dataset.processed_dir, f"data_{dataset._valid_pdb_ids[train_indices[0]]}.pt"),
        map_location="cpu",
        weights_only=False,
    ))
    feature_stats: dict[str, dict[str, torch.Tensor]] = {
        "ligand_atom": _empty_feature_stat(int(sample_dim["ligand_atom"].x_cont.size(1))),
        "protein_atom": _empty_feature_stat(int(sample_dim["protein_atom"].x_cont.size(1))),
        "ligand_molecule": _empty_feature_stat(int(sample_dim["ligand_molecule"].x_cont.size(1))),
    }

    for dataset_idx in tqdm(train_indices, desc="Computing train normalization stats", leave=False):
        pdb_id = dataset._valid_pdb_ids[int(dataset_idx)]
        file_path = os.path.join(dataset.processed_dir, f"data_{pdb_id}.pt")
        data = cast(HeteroData, torch.load(file_path, map_location="cpu", weights_only=False))

        _accumulate_feature_block(feature_stats["ligand_atom"], data["ligand_atom"].x_cont)
        _accumulate_feature_block(feature_stats["protein_atom"], data["protein_atom"].x_cont)
        _accumulate_feature_block(feature_stats["ligand_molecule"], data["ligand_molecule"].x_cont)

    final_stats = {
        key: _finalize_feature_stats(stat)
        for key, stat in feature_stats.items()
    }
    affinity_stats = dataset.compute_affinity_stats(train_indices)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": cache_meta,
            "stats": final_stats,
            "affinity": affinity_stats,
        },
        cache_path,
    )
    logger.info("Saved train-only normalization stats to %s", cache_path)
    return final_stats, affinity_stats
