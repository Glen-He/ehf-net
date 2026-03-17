"""
成本预算探测脚本。

在不启动完整训练的前提下，统计不同阶段的样本成本分布、
不可分单样本超预算比例，以及当前预算下的 batch 填充率，
帮助为训练/验证/Top-N 选择更合适的 cost budget。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from ehfnet.data.datasets import ScaffoldSplitter  # noqa: E402
from ehfnet.graph import estimate_graph_cost_units  # noqa: E402
from ehfnet.runtime import build_dataset, load_train_defaults, resolve_interaction_profile  # noqa: E402
from ehfnet.training.adaptive_batching import (  # noqa: E402
    AdaptiveCostBatchSampler,
    resolve_subset_root_indices,
)


@dataclass(frozen=True)
class PhaseSpec:
    """
    阶段探测配置。

    Attributes:
        name: 阶段名称。
        subset_name: 对应的数据划分名称。
        phase_multiplier: 该阶段的成本倍率。
        default_budget: 当前配置文件中的默认预算。
    """

    name: str
    subset_name: str
    phase_multiplier: float
    default_budget: int


def _parse_budget_list(text: str | None, *, fallback: int) -> list[int]:
    """
    解析预算列表。

    Args:
        text: 逗号分隔的预算字符串。
        fallback: 未显式提供时使用的基准预算。

    Returns:
        list[int]: 去重且升序排列的预算列表。
    """
    if text is None or not text.strip():
        seeds = [
            max(1, int(fallback * 0.5)),
            max(1, int(fallback * 0.75)),
            max(1, int(fallback)),
            max(1, int(fallback * 1.25)),
            max(1, int(fallback * 1.5)),
        ]
        return sorted(set(seeds))
    values = [max(1, int(item.strip())) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Budget list must not be empty.")
    return sorted(set(values))


def _resolve_split_indices(
    *,
    splitter: ScaffoldSplitter,
    dataset: Any,
    split_cache_file: str,
    split_train_frac: float,
    split_val_frac: float,
    split_test_frac: float,
    force_resplit: bool,
) -> dict[str, list[int]]:
    """
    获取数据划分索引。

    Args:
        splitter: Scaffold 划分器。
        dataset: 当前数据集对象。
        split_cache_file: 划分缓存路径。
        split_train_frac: 训练集比例。
        split_val_frac: 验证集比例。
        split_test_frac: 测试集比例。
        force_resplit: 是否强制重建划分。

    Returns:
        dict[str, list[int]]: 划分索引字典。
    """
    if os.path.exists(split_cache_file) and not force_resplit:
        split_indices, _ = ScaffoldSplitter.load_split(split_cache_file)
        return split_indices

    split_indices = splitter.split_indices(
        dataset,
        frac_train=split_train_frac,
        frac_val=split_val_frac,
        frac_test=split_test_frac,
    )
    metadata = {
        "seed": splitter.seed,
        "include_chirality": splitter.include_chirality,
        "fractions": {
            "train": split_train_frac,
            "val": split_val_frac,
            "test": split_test_frac,
        },
        "dataset_size": len(dataset),
        "index_file": os.path.abspath(str(dataset.index_file)),
    }
    ScaffoldSplitter.save_split(split_cache_file, split_indices, metadata=metadata)
    return split_indices


def _quantile(costs: list[int], q: float) -> float:
    """
    计算分位数。

    Args:
        costs: 成本列表。
        q: 分位数取值，范围为 `[0, 1]`。

    Returns:
        float: 对应分位点的成本。
    """
    if not costs:
        return 0.0
    return float(np.quantile(np.asarray(costs, dtype=np.float64), q))


def _simulate_fill_ratios(sample_costs: list[int], *, budget: int) -> list[float]:
    """
    模拟当前预算下的 batch 填充率。

    Args:
        sample_costs: 样本成本列表。
        budget: 单 batch 成本预算。

    Returns:
        list[float]: 每个 batch 的填充率。
    """
    sampler = AdaptiveCostBatchSampler(
        sample_costs=sample_costs,
        max_cost=budget,
        shuffle=False,
        seed=42,
    )
    fill_ratios: list[float] = []
    for batch_indices in sampler:
        batch_cost = sum(min(sample_costs[idx], budget) for idx in batch_indices)
        fill_ratios.append(float(batch_cost) / float(budget))
    return fill_ratios


def _subset_pdb_ids(dataset: Any, root_indices: list[int]) -> list[str]:
    """
    解析样本对应的 PDB ID。

    Args:
        dataset: 当前数据集对象。
        root_indices: 根数据集索引列表。

    Returns:
        list[str]: 与索引对齐的 PDB ID 列表。
    """
    index_df = dataset.index_df
    return [str(index_df.iloc[root_idx]["pdb_id"]) for root_idx in root_indices]


def _print_phase_report(
    *,
    phase: PhaseSpec,
    costs: list[int],
    pdb_ids: list[str],
    budgets: list[int],
) -> None:
    """
    打印单个阶段的预算分析报告。

    Args:
        phase: 阶段配置。
        costs: 样本成本列表。
        pdb_ids: 与成本对齐的 PDB ID 列表。
        budgets: 需要分析的预算列表。
    """
    if not costs:
        print(f"[{phase.name}] Empty subset, skipped.")
        return

    print(
        f"[{phase.name}] samples={len(costs)} "
        f"mean={np.mean(costs):.1f} p90={_quantile(costs, 0.90):.1f} "
        f"p95={_quantile(costs, 0.95):.1f} p99={_quantile(costs, 0.99):.1f} "
        f"max={max(costs)} default_budget={phase.default_budget}"
    )

    largest = sorted(zip(costs, pdb_ids), key=lambda item: item[0], reverse=True)[:5]
    for rank, (cost, pdb_id) in enumerate(largest, start=1):
        print(f"  top{rank}: pdb={pdb_id} cost={cost}")

    for budget in budgets:
        oversize_positions = [idx for idx, cost in enumerate(costs) if cost > budget]
        oversize_rate = 100.0 * len(oversize_positions) / max(1, len(costs))
        fill_ratios = _simulate_fill_ratios(costs, budget=budget)
        avg_fill = 100.0 * float(np.mean(fill_ratios)) if fill_ratios else 0.0
        p10_fill = 100.0 * _quantile([int(r * 10000) for r in fill_ratios], 0.10) / 10000.0
        p50_fill = 100.0 * _quantile([int(r * 10000) for r in fill_ratios], 0.50) / 10000.0
        print(
            f"  budget={budget:<8d} "
            f"oversize_samples={len(oversize_positions):<4d} ({oversize_rate:5.1f}%) "
            f"avg_fill={avg_fill:5.1f}% p10_fill={p10_fill:5.1f}% p50_fill={p50_fill:5.1f}% "
            f"batches={len(fill_ratios)}"
        )

    print()


def main() -> None:
    """
    探测入口。

    Raises:
        ValueError: 当输入参数不合法时抛出。
    """
    parser = argparse.ArgumentParser(
        description="Probe dataset cost budgets without launching a full training run."
    )
    parser.add_argument("--config", type=str, default="configs/train.toml", help="Path to train TOML config")
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--index_file", type=str, default=None, help="Override index CSV path")
    parser.add_argument("--split_cache_file", type=str, default=None, help="Override split cache path")
    parser.add_argument("--force_resplit", action="store_true", help="Regenerate scaffold split before probing")
    parser.add_argument("--train_budgets", type=str, default=None, help="Comma-separated train budgets")
    parser.add_argument("--val_partial_budgets", type=str, default=None, help="Comma-separated lightweight validation budgets")
    parser.add_argument("--val_full_budgets", type=str, default=None, help="Comma-separated full validation budgets")
    parser.add_argument("--blind_pool_budgets", type=str, default=None, help="Comma-separated blind-pool refresh budgets")
    parser.add_argument("--final_topn_budgets", type=str, default=None, help="Comma-separated final Top-N budgets")
    args = parser.parse_args()

    defaults = load_train_defaults(
        config_path=args.config,
        project_root=PROJECT_ROOT,
    )
    if args.data_root is not None:
        defaults["data_root"] = args.data_root
    if args.index_file is not None:
        defaults["index_file"] = args.index_file
    elif "index_file" not in defaults:
        defaults["index_file"] = str(Path(str(defaults["data_root"])) / "index.csv")
    if args.split_cache_file is not None:
        defaults["split_cache_file"] = args.split_cache_file

    required_keys = [
        "data_root",
        "index_file",
        "split_cache_file",
        "split_train_frac",
        "split_val_frac",
        "split_test_frac",
        "split_seed",
        "esm",
        "esm_model_name",
        "esm_dim",
        "r_cutoff_intra",
        "max_neighbors_intra",
        "atom_neighbor_cap",
        "residue_neighbor_cap",
        "residue_radius_scale",
        "residue_radius_bias",
        "ligand_atom_fallback_k",
        "protein_atom_fallback_k",
        "protein_residue_fallback_k",
        "ablation_mode",
        "num_gnn_blocks",
        "dynamic_inter_knn_k",
        "dynamic_residue_knn_k",
        "crop_candidate_topk",
        "train_cost_budget",
        "val_cost_budget",
        "blind_pool_cost_budget",
        "final_topn_cost_budget",
    ]
    missing_keys = [key for key in required_keys if key not in defaults]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    interaction_profile = resolve_interaction_profile(
        ablation_mode=str(defaults["ablation_mode"])
    )
    dataset = build_dataset(
        root=str(defaults["data_root"]),
        index_file=str(defaults["index_file"]),
        esm_root=None,
        esm=str(defaults["esm"]),
        esm_model_name=str(defaults["esm_model_name"]),
        esm_device=str(defaults.get("device", "cpu")),
        esm_dim=int(defaults["esm_dim"]),
        r_cutoff_intra=float(defaults["r_cutoff_intra"]),
        max_neighbors_intra=int(defaults["max_neighbors_intra"]),
        atom_neighbor_cap=int(defaults["atom_neighbor_cap"]),
        residue_neighbor_cap=int(defaults["residue_neighbor_cap"]),
        residue_radius_scale=float(defaults["residue_radius_scale"]),
        residue_radius_bias=float(defaults["residue_radius_bias"]),
        ligand_atom_fallback_k=int(defaults["ligand_atom_fallback_k"]),
        protein_atom_fallback_k=int(defaults["protein_atom_fallback_k"]),
        protein_residue_fallback_k=int(defaults["protein_residue_fallback_k"]),
        interaction_profile=interaction_profile,
    )

    splitter = ScaffoldSplitter(include_chirality=False, seed=int(defaults["split_seed"]))
    split_indices = _resolve_split_indices(
        splitter=splitter,
        dataset=dataset,
        split_cache_file=str(defaults["split_cache_file"]),
        split_train_frac=float(defaults["split_train_frac"]),
        split_val_frac=float(defaults["split_val_frac"]),
        split_test_frac=float(defaults["split_test_frac"]),
        force_resplit=bool(args.force_resplit),
    )
    train_set, val_set, test_set = ScaffoldSplitter.subsets_from_indices(dataset, split_indices)
    subset_map = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    phases = [
        PhaseSpec(
            name="train",
            subset_name="train",
            phase_multiplier=1.0,
            default_budget=int(defaults["train_cost_budget"]),
        ),
        PhaseSpec(
            name="val_partial",
            subset_name="val",
            phase_multiplier=1.35,
            default_budget=int(defaults["val_cost_budget"]),
        ),
        PhaseSpec(
            name="val_full",
            subset_name="val",
            phase_multiplier=1.75,
            default_budget=int(defaults["val_cost_budget"]),
        ),
        PhaseSpec(
            name="blind_pool",
            subset_name="train",
            phase_multiplier=2.25,
            default_budget=int(defaults["blind_pool_cost_budget"]),
        ),
        PhaseSpec(
            name="final_topn",
            subset_name="test",
            phase_multiplier=2.25,
            default_budget=int(defaults["final_topn_cost_budget"]),
        ),
    ]
    phase_budget_text = {
        "train": args.train_budgets,
        "val_partial": args.val_partial_budgets,
        "val_full": args.val_full_budgets,
        "blind_pool": args.blind_pool_budgets,
        "final_topn": args.final_topn_budgets,
    }

    print(f"Dataset root: {defaults['data_root']}")
    print(
        "Split sizes: "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)}"
    )
    device_text = str(defaults.get("device", "cpu"))
    print(f"Configured device: {device_text}")
    if str(device_text).startswith("cuda"):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device_text))
            print(
                "Current CUDA memory: "
                f"free={free_bytes / 1024**3:.2f} GiB total={total_bytes / 1024**3:.2f} GiB"
            )
        except Exception:
            print("Current CUDA memory: unavailable")
    print()

    for phase in phases:
        subset = subset_map[phase.subset_name]
        root_indices = resolve_subset_root_indices(subset)
        pdb_ids = _subset_pdb_ids(dataset, root_indices)
        costs = [
            estimate_graph_cost_units(
                dataset.get_graph_cost_profile(root_idx),
                num_gnn_blocks=int(defaults["num_gnn_blocks"]),
                dynamic_inter_max_neighbors=int(defaults["dynamic_inter_knn_k"]),
                dynamic_residue_max_neighbors=int(defaults["dynamic_residue_knn_k"]),
                dynamic_residue_candidate_topk=int(defaults["crop_candidate_topk"]),
                phase_multiplier=phase.phase_multiplier,
            )
            for root_idx in root_indices
        ]
        budgets = _parse_budget_list(
            phase_budget_text[phase.name],
            fallback=phase.default_budget,
        )
        _print_phase_report(
            phase=phase,
            costs=costs,
            pdb_ids=pdb_ids,
            budgets=budgets,
        )


if __name__ == "__main__":
    main()
