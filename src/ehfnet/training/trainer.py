"""
训练循环

提供 EHFNet 模型的训练和验证功能。
"""

import os
import math
import json
import hashlib
import traceback
import torch
import logging
import gc
import numpy as np
from typing import Any, cast
from pathlib import Path
from scipy import stats as scipy_stats
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_geometric.data import HeteroData

from torch_scatter import scatter_mean

from ehfnet.models import EHFNet
from ehfnet.graph import GraphCollator, crop_graph_to_center
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.datasets.splitter import ScaffoldSplitter
from ehfnet.encoders.feature_specs import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.training.losses import FlowMatchingLoss
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.blind_pool import (
    refresh_blind_candidate_pool,
    save_blind_pool,
    load_blind_pool,
    build_blind_pool_compatibility,
    BlindCandidateReplayDataset,
    replay_and_compute_losses,
    should_refresh_pool,
    get_pool_stats,
)
from ehfnet.training.candidate_generation import (
    generate_blind_candidates,
    generate_candidates_from_loader,
)
from ehfnet.training.checkpoint_schema import (
    build_feature_signature,
    build_model_config,
)


logger = logging.getLogger(__name__)


DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "pose_weight": 1.0,
    "center_weight": 0.35,
    "aff_weight": 0.0,
    "clash_weight": 0.0,
    "bias": 0.0,
}


def compute_pose_quality_target(current_pos: torch.Tensor, target_pos: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    sq_diff = ((current_pos - target_pos) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(scatter_mean(sq_diff, batch_idx, dim=0) + 1e-8)
    return torch.sigmoid((4.0 - rmsd) / 0.75).unsqueeze(-1)


def select_pose_ranking_logit(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    rank_logit = predictions.get("pose_rank_score")
    if rank_logit is not None:
        return rank_logit

    pose_quality = predictions.get("pose_quality")
    if pose_quality is None:
        raise KeyError("Predictions must contain either 'pose_rank_score' or 'pose_quality'.")
    return pose_quality


def apply_loss_context(
    batch_obj: Any,
    *,
    current_epoch: int,
    total_epochs_count: int,
    warmup_epochs_count: int,
    training: bool,
) -> None:
    progress = 1.0 if total_epochs_count <= 1 else current_epoch / max(1, total_epochs_count - 1)
    warmup_end = min(1.0, warmup_epochs_count / max(1, total_epochs_count))
    batch_obj.loss_progress = float(max(0.0, min(1.0, progress)))
    batch_obj.loss_warmup_end = float(max(0.0, min(1.0, warmup_end)))
    batch_obj.loss_is_training = bool(training)


def build_local_batch_from_centers(
    batch_obj: Any,
    *,
    centers: torch.Tensor,
    crop_radius: float,
    graph_builder: Any,
    collator: GraphCollator,
) -> Any:
    if centers.ndim != 2 or centers.size(1) != 3:
        raise ValueError("centers must have shape [B, 3].")

    samples = batch_obj.to_data_list() if hasattr(batch_obj, "to_data_list") else [batch_obj]
    if len(samples) != int(centers.size(0)):
        raise ValueError(
            f"Mismatch between samples ({len(samples)}) and centers ({int(centers.size(0))})."
        )

    cropped_samples = [
        crop_graph_to_center(
            sample,
            center=centers[i].detach().cpu(),
            radius=crop_radius,
            graph_builder=graph_builder,
        )
        for i, sample in enumerate(samples)
    ]
    return collator.collate(cropped_samples)


def compute_center_proposal_target(
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    ligand_centers: torch.Tensor,
    *,
    positive_radius: float = 4.0,
    soft_sigma: float = 3.0,
) -> torch.Tensor:
    center_ref = ligand_centers[residue_batch]
    dist = torch.norm(residue_pos - center_ref, dim=-1)
    soft_target = torch.exp(-0.5 * (dist / soft_sigma) ** 2)
    soft_target = torch.where(dist <= positive_radius, torch.ones_like(soft_target), soft_target)
    return soft_target.unsqueeze(-1)


def compute_residue_proposal_priors(
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    knn: int = 16,
) -> torch.Tensor:
    priors = residue_pos.new_zeros((residue_pos.size(0), 4))
    if residue_pos.numel() == 0:
        return priors

    num_graphs = int(residue_batch.max().item()) + 1 if residue_batch.numel() > 0 else 0
    for graph_idx in range(num_graphs):
        mask = residue_batch == graph_idx
        pos = residue_pos[mask]
        if pos.size(0) == 0:
            continue
        if pos.size(0) == 1:
            priors[mask] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=residue_pos.device, dtype=residue_pos.dtype)
            continue

        dist = torch.cdist(pos, pos)
        dist.fill_diagonal_(float("inf"))
        k = min(knn, max(1, pos.size(0) - 1))
        knn_dist = torch.topk(dist, k=k, largest=False, dim=-1).values
        mean_knn = knn_dist.mean(dim=-1)

        protein_center = pos.mean(dim=0, keepdim=True)
        radial = torch.norm(pos - protein_center, dim=-1)
        radial_norm = radial / radial.max().clamp_min(1e-6)
        depth = 1.0 - radial_norm

        density = torch.exp(-mean_knn / 4.0)
        exposure = torch.sigmoid((mean_knn - mean_knn.mean()) / mean_knn.std(unbiased=False).clamp_min(1e-6))
        cavity = density * depth * (1.0 - exposure)

        priors[mask] = torch.stack([density, exposure, depth, cavity], dim=-1)

    return priors.clamp(0.0, 1.0)


def predict_center_proposal_logits(
    model: torch.nn.Module,
    batch_obj: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = resolve_ehfnet_model(model)
    residue_store = batch_obj["protein_residue"]
    lig_store = batch_obj["ligand_molecule"]
    residue_batch = getattr(
        residue_store,
        "batch",
        torch.zeros(residue_store.pos.size(0), dtype=torch.long),
    )
    esm_missing_mask = getattr(residue_store, "esm_missing_mask", None)
    residue_prior_feat = compute_residue_proposal_priors(
        residue_store.pos.to(device),
        residue_batch.to(device),
    )
    logits = base_model.predict_center_logits(
        residue_x_cat=residue_store.x_cat.to(device),
        residue_x_cont=residue_store.x_cont.to(device),
        residue_pos=residue_store.pos.to(device),
        residue_batch=residue_batch.to(device),
        lig_mol_x_cont=lig_store.x_cont.to(device),
        residue_esm_missing_mask=esm_missing_mask.to(device) if esm_missing_mask is not None else None,
        residue_prior_feat=residue_prior_feat,
    )
    return logits, residue_store.pos.to(device), residue_batch.to(device), residue_prior_feat


def resolve_ehfnet_model(model: torch.nn.Module) -> EHFNet:
    base_model = getattr(model, "module", model)
    if not isinstance(base_model, EHFNet):
        raise TypeError(f"Expected EHFNet-compatible model, got {type(base_model)!r}")
    return base_model


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


def _compute_train_split_normalization_stats(
    dataset: PDBBindDataset,
    train_indices: list[int],
    *,
    split_cache_file: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
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


def compute_proposal_loss(
    model: torch.nn.Module,
    batch_obj: Any,
    *,
    device: torch.device,
    positive_radius: float = 4.0,
) -> torch.Tensor:
    residue_store = batch_obj["protein_residue"]
    lig_store = batch_obj["ligand_molecule"]

    logits, _, residue_batch, _ = predict_center_proposal_logits(
        model,
        batch_obj,
        device=device,
    )
    ligand_centers = scatter_mean(
        batch_obj["ligand_atom"].pos,
        batch_obj["ligand_atom"].batch,
        dim=0,
        dim_size=lig_store.x_cont.size(0),
    )
    target = compute_center_proposal_target(
        residue_store.pos.to(device),
        residue_batch,
        ligand_centers.to(device),
        positive_radius=positive_radius,
    )
    weight = 1.0 + 3.0 * target
    per_residue = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="none")
    per_graph = scatter_mean(per_residue.view(-1), residue_batch, dim=0)
    return per_graph.mean()


def select_diverse_center_indices(
    logits: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int,
    min_distance: float,
) -> torch.Tensor:
    order = torch.argsort(logits.view(-1), descending=True)
    selected: list[int] = []

    for idx in order.tolist():
        if len(selected) >= topk:
            break
        if not selected:
            selected.append(idx)
            continue
        pos = positions[idx]
        if all(torch.norm(pos - positions[j]).item() >= min_distance for j in selected):
            selected.append(idx)

    if len(selected) < min(topk, positions.size(0)):
        for idx in order.tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min(topk, positions.size(0)):
                break

    return torch.tensor(selected, dtype=torch.long, device=positions.device)


def combine_center_pose_score(
    center_logit: torch.Tensor,
    pose_logit: torch.Tensor,
    *,
    aff_logit: torch.Tensor | None = None,
    clash_value: torch.Tensor | None = None,
    fusion_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    fusion = dict(DEFAULT_FUSION_WEIGHTS)
    if fusion_weights is not None:
        fusion.update(fusion_weights)
    center_score = torch.sigmoid(center_logit.view(-1))
    pose_score = torch.sigmoid(pose_logit.view(-1))
    result = (
        fusion["pose_weight"] * pose_score
        + fusion["center_weight"] * center_score
        + fusion["bias"]
    )
    if aff_logit is not None and fusion.get("aff_weight", 0.0) != 0.0:
        aff_score = torch.sigmoid(aff_logit.view(-1))
        result = result + fusion["aff_weight"] * aff_score
    if clash_value is not None and fusion.get("clash_weight", 0.0) != 0.0:
        clash_penalty = torch.exp(-clash_value.view(-1) / 10.0)
        result = result + fusion["clash_weight"] * clash_penalty
    return result


def select_training_crop_centers(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    progress: float,
    positive_radius: float,
    bucket_topk: int = 8,
    weighted_sampling: bool = True,
    disable_jitter: bool = False,
    disable_hard_negative: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    progress = float(max(0.0, min(1.0, progress)))
    stage_weights = {
        "gt": max(0.10, 0.50 - 0.40 * progress),
        "jitter": max(0.15, 0.30 - 0.10 * progress),
        "proposal_pos": 0.10 + 0.20 * progress,
        "near_miss": 0.05 + 0.15 * progress,
        "hard_neg": max(0.0, -0.05 + 0.30 * progress),
    }
    if disable_jitter:
        stage_weights["jitter"] = 0.0
    if disable_hard_negative:
        stage_weights["hard_neg"] = 0.0
    jitter_sigma = 2.0 + 6.0 * progress
    chosen_centers: list[torch.Tensor] = []
    chosen_modes: list[str] = []
    num_graphs = int(ligand_centers.size(0))

    def _sample_from_bucket(bucket_pos: torch.Tensor, bucket_logits: torch.Tensor) -> torch.Tensor:
        if bucket_pos.size(0) == 1:
            return bucket_pos[0]
        k = min(max(1, bucket_topk), bucket_pos.size(0))
        pool_pos = bucket_pos[:k]
        pool_logits = bucket_logits[:k]
        if not weighted_sampling:
            choice_idx = torch.randint(k, (1,), device=pool_pos.device).item()
            return pool_pos[choice_idx]
        weight = torch.softmax(pool_logits, dim=0)
        choice_idx = int(torch.multinomial(weight, 1).item())
        return pool_pos[choice_idx]

    for graph_idx in range(num_graphs):
        gt_center = ligand_centers[graph_idx]
        mask = residue_batch == graph_idx
        graph_pos = residue_pos[mask]
        graph_logits = proposal_logits[mask].view(-1)
        if graph_pos.numel() == 0:
            chosen_centers.append(gt_center)
            chosen_modes.append("gt_fallback")
            continue

        order = torch.argsort(graph_logits, descending=True)
        ordered_pos = graph_pos[order]
        ordered_logits = graph_logits[order]
        ordered_dist = torch.norm(ordered_pos - gt_center.unsqueeze(0), dim=-1)
        positive_mask = ordered_dist <= positive_radius
        near_mask = (ordered_dist > positive_radius) & (ordered_dist <= positive_radius * 2.0)
        hard_mask = ordered_dist > positive_radius * 2.0

        bucket_to_center: dict[str, torch.Tensor] = {
            "gt": gt_center,
        }
        if not disable_jitter:
            bucket_to_center["jitter"] = gt_center + torch.randn_like(gt_center) * jitter_sigma
        if positive_mask.any():
            pos_pool = ordered_pos[positive_mask]
            pos_logits = ordered_logits[positive_mask]
            bucket_to_center["proposal_pos"] = _sample_from_bucket(pos_pool, pos_logits)
        if near_mask.any():
            near_pool = ordered_pos[near_mask]
            near_logits = ordered_logits[near_mask]
            bucket_to_center["near_miss"] = _sample_from_bucket(near_pool, near_logits)
        if hard_mask.any() and not disable_hard_negative:
            hard_pool = ordered_pos[hard_mask]
            hard_logits = ordered_logits[hard_mask]
            bucket_to_center["hard_neg"] = _sample_from_bucket(hard_pool, hard_logits)

        available_modes = list(bucket_to_center.keys())
        weight_tensor = torch.tensor(
            [stage_weights.get(mode, 0.0) for mode in available_modes],
            dtype=ligand_centers.dtype,
            device=ligand_centers.device,
        )
        if float(weight_tensor.sum().item()) <= 0.0:
            chosen_mode = "gt"
        else:
            chosen_mode = available_modes[int(torch.multinomial(weight_tensor / weight_tensor.sum(), 1).item())]

        chosen_centers.append(bucket_to_center[chosen_mode])
        chosen_modes.append(chosen_mode)

    return torch.stack(chosen_centers, dim=0), chosen_modes


def compute_pairwise_pose_ranking_loss(
    pose_logit_a: torch.Tensor,
    pose_target_a: torch.Tensor,
    pose_logit_b: torch.Tensor,
    pose_target_b: torch.Tensor,
    *,
    margin: float,
    min_delta: float = 0.05,
    extra_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    qa = pose_target_a.view(-1)
    qb = pose_target_b.view(-1)
    sa = pose_logit_a.view(-1)
    sb = pose_logit_b.view(-1)
    delta = qa - qb
    valid = delta.abs() >= min_delta
    if extra_mask is not None:
        valid = valid & extra_mask.view(-1).to(device=valid.device, dtype=torch.bool)
    if not bool(valid.any()):
        return sa.new_zeros(()), 0
    direction = torch.sign(delta[valid])
    return F.relu(margin - direction * (sa[valid] - sb[valid])).mean(), int(valid.sum().item())


def select_wrong_center_candidates(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    positive_radius: float,
    bucket_topk: int = 8,
    weighted_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_graphs = int(ligand_centers.size(0))
    wrong_centers: list[torch.Tensor] = []
    wrong_center_scores: list[torch.Tensor] = []
    valid_mask: list[bool] = []

    def _sample_bucket(bucket_pos: torch.Tensor, bucket_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(max(1, bucket_topk), bucket_pos.size(0))
        pool_pos = bucket_pos[:k]
        pool_logits = bucket_logits[:k]
        if pool_pos.size(0) == 1:
            return pool_pos[0], pool_logits[0]
        if not weighted_sampling:
            idx = int(torch.randint(pool_pos.size(0), (1,), device=pool_pos.device).item())
            return pool_pos[idx], pool_logits[idx]
        prob = torch.softmax(pool_logits, dim=0)
        idx = int(torch.multinomial(prob, 1).item())
        return pool_pos[idx], pool_logits[idx]

    for graph_idx in range(num_graphs):
        gt_center = ligand_centers[graph_idx]
        mask = residue_batch == graph_idx
        graph_pos = residue_pos[mask]
        graph_logits = proposal_logits[mask].view(-1)
        if graph_pos.numel() == 0:
            wrong_centers.append(gt_center)
            wrong_center_scores.append(gt_center.new_zeros(()))
            valid_mask.append(False)
            continue
        order = torch.argsort(graph_logits, descending=True)
        ordered_pos = graph_pos[order]
        ordered_logits = graph_logits[order]
        dist = torch.norm(ordered_pos - gt_center.unsqueeze(0), dim=-1)
        near_mask = (dist > positive_radius) & (dist <= positive_radius * 2.0)
        hard_mask = dist > positive_radius * 2.0
        if near_mask.any():
            center, score = _sample_bucket(ordered_pos[near_mask], ordered_logits[near_mask])
            valid = True
        elif hard_mask.any():
            center, score = _sample_bucket(ordered_pos[hard_mask], ordered_logits[hard_mask])
            valid = True
        else:
            center, score = gt_center, gt_center.new_zeros(())
            valid = False
        wrong_centers.append(center)
        wrong_center_scores.append(score)
        valid_mask.append(valid)

    return (
        torch.stack(wrong_centers, dim=0),
        torch.stack([score.view(1) for score in wrong_center_scores], dim=0).view(-1),
        torch.as_tensor(valid_mask, dtype=torch.bool, device=ligand_centers.device),
    )


def sample_hard_ranking_time_and_centers(
    t_anchor: torch.Tensor,
    crop_centers: torch.Tensor,
    wrong_centers: torch.Tensor,
    wrong_center_valid_mask: torch.Tensor,
    *,
    progress: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample hard ranking variants.

    Returns:
      t_hard: time steps for hard samples
      hard_centers: candidate centers
      strategy_id: 0 same-center-bad-pose, 1 wrong-center
    """
    B = t_anchor.size(0)
    device = t_anchor.device

    strategy = torch.rand(B, device=device)
    t_hard = t_anchor.clone()
    hard_centers = crop_centers.clone()
    strategy_id = torch.zeros(B, device=device, dtype=torch.long)

    mask_worse_time = strategy < 0.45
    if mask_worse_time.any():
        n = int(mask_worse_time.sum().item())
        scale = 0.25 + 0.45 * torch.rand(n, device=device)
        t_hard[mask_worse_time] = t_anchor[mask_worse_time] * scale

    mask_offset = (~mask_worse_time) & wrong_center_valid_mask.to(device=device)
    if mask_offset.any():
        hard_centers[mask_offset] = wrong_centers[mask_offset]
        strategy_id[mask_offset] = 1
        scale = 0.65 + 0.25 * torch.rand(int(mask_offset.sum().item()), device=device)
        t_hard[mask_offset] = torch.clamp(t_anchor[mask_offset] * scale, max=0.85)

    sigma = 1e-3
    t_hard = t_hard.clamp(min=sigma, max=1.0 - sigma)
    return t_hard, hard_centers, strategy_id


def clone_shares_tensor_storage(batch_obj: Any, node_type: str = "ligand_atom", attr: str = "pos") -> bool:
    cloned = batch_obj.clone()
    original_tensor = batch_obj[node_type][attr]
    cloned_tensor = cloned[node_type][attr]
    return bool(
        torch.is_tensor(original_tensor)
        and torch.is_tensor(cloned_tensor)
        and original_tensor.data_ptr() == cloned_tensor.data_ptr()
    )


def summarize_blind_candidate_records(
    candidate_records: list[dict[str, Any]],
    *,
    topk_values: tuple[int, ...],
    fusion_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not candidate_records:
        return {"topn_total_graphs": 0.0}

    topk_unique = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    metrics: dict[str, float] = {
        "topn_total_graphs": float(len(candidate_records)),
    }
    center_recall_hits = {1: 0.0, 3: 0.0, 8: 0.0, 16: 0.0}
    oracle_top1_rmsd: list[float] = []
    oracle_top5_rmsd: list[float] = []
    reranked_top1_rmsd: list[float] = []
    reranked_top5_rmsd: list[float] = []
    reranked_topk_rmsd: dict[int, list[float]] = {k: [] for k in topk_unique}
    proposal_failures = 0.0
    local_failures = 0.0
    ranking_failures = 0.0
    successes = 0.0

    for record in candidate_records:
        center_hits = list(record.get("center_hits", []))
        candidates = list(record.get("candidates", []))
        if not candidates:
            proposal_failures += 1.0
            continue

        for k in center_recall_hits:
            if any(center_hits[: min(k, len(center_hits))]):
                center_recall_hits[k] += 1.0

        oracle_all = min(float(item["rmsd"]) for item in candidates)
        oracle_first5_pool = [float(item["rmsd"]) for item in candidates if int(item.get("proposal_rank", 999)) <= 5]
        if not oracle_first5_pool:
            oracle_first5_pool = [float(item["rmsd"]) for item in candidates]
        oracle_top1_rmsd.append(oracle_all)
        oracle_top5_rmsd.append(min(oracle_first5_pool))

        reranked = sorted(
            candidates,
            key=lambda item: float(
                combine_center_pose_score(
                    torch.tensor([item["center_logit"]], dtype=torch.float32),
                    torch.tensor([item.get("ranking_logit", item["pose_logit"])], dtype=torch.float32),
                    aff_logit=torch.tensor([item.get("aff_logit", 0.0)], dtype=torch.float32),
                    clash_value=torch.tensor([item.get("clash_value", 0.0)], dtype=torch.float32),
                    fusion_weights=fusion_weights,
                )[0].item()
            ),
            reverse=True,
        )
        reranked_top1 = float(reranked[0]["rmsd"])
        reranked_top5 = min(float(item["rmsd"]) for item in reranked[: min(5, len(reranked))])
        reranked_top1_rmsd.append(reranked_top1)
        reranked_top5_rmsd.append(reranked_top5)
        for k in topk_unique:
            reranked_topk_rmsd[k].append(min(float(item["rmsd"]) for item in reranked[: min(k, len(reranked))]))

        has_center_hit = any(center_hits)
        if not has_center_hit:
            proposal_failures += 1.0
        elif oracle_all >= 2.0:
            local_failures += 1.0
        elif reranked_top1 >= 2.0:
            ranking_failures += 1.0
        else:
            successes += 1.0

    total = max(1.0, float(len(candidate_records)))
    for k, hit_count in center_recall_hits.items():
        metrics[f"center_recall@{k}"] = (hit_count / total) * 100.0

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    def _success(values: list[float], threshold: float) -> float:
        return 100.0 * float(sum(v < threshold for v in values)) / max(1, len(values))

    metrics["oracle_top1_success_2a"] = _success(oracle_top1_rmsd, 2.0)
    metrics["oracle_top1_success_5a"] = _success(oracle_top1_rmsd, 5.0)
    metrics["oracle_top5_success_2a"] = _success(oracle_top5_rmsd, 2.0)
    metrics["oracle_top5_success_5a"] = _success(oracle_top5_rmsd, 5.0)
    metrics["oracle_top1_mean_best_rmsd"] = _mean(oracle_top1_rmsd)
    metrics["oracle_top5_mean_best_rmsd"] = _mean(oracle_top5_rmsd)
    metrics["reranked_top1_success_2a"] = _success(reranked_top1_rmsd, 2.0)
    metrics["reranked_top1_success_5a"] = _success(reranked_top1_rmsd, 5.0)
    metrics["reranked_top5_success_2a"] = _success(reranked_top5_rmsd, 2.0)
    metrics["reranked_top5_success_5a"] = _success(reranked_top5_rmsd, 5.0)
    metrics["reranked_top1_mean_best_rmsd"] = _mean(reranked_top1_rmsd)
    metrics["reranked_top5_mean_best_rmsd"] = _mean(reranked_top5_rmsd)
    metrics["proposal_gap"] = (proposal_failures / total) * 100.0
    metrics["local_gap"] = (local_failures / total) * 100.0
    metrics["ranking_gap"] = (ranking_failures / total) * 100.0
    metrics["pipeline_success"] = (successes / total) * 100.0

    for k in topk_unique:
        source = reranked_topk_rmsd[k]
        metrics[f"top{k}_success_2a"] = _success(source, 2.0)
        metrics[f"top{k}_success_5a"] = _success(source, 5.0)
        metrics[f"top{k}_mean_best_rmsd"] = _mean(source)

    return metrics


def calibrate_linear_fusion_weights(
    candidate_records: list[dict[str, Any]],
    *,
    topk_values: tuple[int, ...],
    search_center_weights: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.5, 0.65),
    search_aff_weights: tuple[float, ...] = (0.0,),
    search_clash_weights: tuple[float, ...] = (0.0,),
) -> dict[str, float]:
    best_weights = dict(DEFAULT_FUSION_WEIGHTS)
    best_metrics = summarize_blind_candidate_records(
        candidate_records,
        topk_values=topk_values,
        fusion_weights=best_weights,
    )

    def _is_better(trial: dict, ref: dict) -> bool:
        t1 = trial.get("reranked_top1_success_2a", 0.0)
        r1 = ref.get("reranked_top1_success_2a", 0.0)
        if t1 > r1 + 1e-6:
            return True
        if t1 < r1 - 1e-6:
            return False
        t5 = trial.get("reranked_top5_success_2a", 0.0)
        r5 = ref.get("reranked_top5_success_2a", 0.0)
        if t5 > r5 + 1e-6:
            return True
        if t5 < r5 - 1e-6:
            return False
        return trial.get("reranked_top1_mean_best_rmsd", float("inf")) < ref.get("reranked_top1_mean_best_rmsd", float("inf")) - 1e-6

    for cw in search_center_weights:
        for aw in search_aff_weights:
            for clw in search_clash_weights:
                trial_weights = {
                    "pose_weight": 1.0,
                    "center_weight": float(cw),
                    "aff_weight": float(aw),
                    "clash_weight": float(clw),
                    "bias": 0.0,
                }
                trial_metrics = summarize_blind_candidate_records(
                    candidate_records,
                    topk_values=topk_values,
                    fusion_weights=trial_weights,
                )
                if _is_better(trial_metrics, best_metrics):
                    best_weights = trial_weights
                    best_metrics = trial_metrics

    return best_weights


def should_run_bootstrap(
    *,
    epoch: int,
    batch_idx: int,
    total_epochs: int,
    frequency: int,
    start_ratio: float,
) -> bool:
    if frequency <= 0:
        return False
    progress = 1.0 if total_epochs <= 1 else epoch / max(1, total_epochs - 1)
    return progress >= start_ratio and batch_idx % frequency == 0


def select_bootstrap_blind_centers(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    positive_radius: float,
    bucket_topk: int = 8,
) -> torch.Tensor:
    wrong_centers, _, wrong_valid_mask = select_wrong_center_candidates(
        ligand_centers,
        proposal_logits,
        residue_pos,
        residue_batch,
        positive_radius=positive_radius,
        bucket_topk=bucket_topk,
        weighted_sampling=True,
    )
    bootstrap_centers = ligand_centers.clone()
    if wrong_valid_mask.any():
        mix_mask = (torch.rand_like(wrong_valid_mask.float()) < 0.7) & wrong_valid_mask
        bootstrap_centers[mix_mask] = wrong_centers[mix_mask]
    return bootstrap_centers


def compute_bootstrap_pose_quality_loss(
    *,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    source_batch: Any,
    placement_centers: torch.Tensor,
    epoch: int,
    ode_steps: int,
    graph_builder: Any,
    collator: GraphCollator,
    crop_radius: float,
) -> torch.Tensor:
    blind_local_batch = build_local_batch_from_centers(
        source_batch,
        centers=placement_centers.detach().cpu(),
        crop_radius=crop_radius,
        graph_builder=graph_builder,
        collator=collator,
    )
    device = next(student_model.parameters()).device
    blind_local_batch = blind_local_batch.to(device)
    x_ref = blind_local_batch["ligand_atom"].pos
    lig_batch = blind_local_batch["ligand_atom"].batch
    masses = blind_local_batch["ligand_atom"].masses
    B = int(lig_batch.max().item()) + 1
    with torch.no_grad():
        teacher_batch = blind_local_batch.clone()
        x0 = matcher._generate_random_pose(
            x_ref=x_ref,
            batch=lig_batch,
            B=B,
            masses=masses,
            torsion_indices=getattr(teacher_batch, "torsion_indices", None),
            torsion_moving_mask=getattr(teacher_batch, "torsion_moving_mask", None),
            seed_pos=teacher_batch["ligand_atom"].get("start_pos", None),
            protein_pos=teacher_batch["protein_atom"].pos,
            protein_batch=getattr(teacher_batch["protein_atom"], "batch", None),
            placement_centers=placement_centers,
            epoch=epoch,
        )
        teacher_batch["ligand_atom"].pos = x0
        final_pos, _ = matcher.ode_solve(
            model=teacher_model,
            data=teacher_batch,
            steps=ode_steps,
            method="euler",
            store_trajectory=False,
        )
        teacher_target = compute_pose_quality_target(final_pos, x_ref, lig_batch)

    student_batch = blind_local_batch.clone()
    student_batch["ligand_atom"].pos = final_pos.detach()
    student_batch.t = torch.ones(B, device=x_ref.device, dtype=x_ref.dtype)
    student_pred = student_model(student_batch, student_batch.t)
    pred_pose_quality = student_pred["pose_quality"].view(-1)
    target_pose_quality = teacher_target.view(-1).to(device=pred_pose_quality.device, dtype=pred_pose_quality.dtype)
    weight = 1.0 + 2.0 * target_pose_quality
    return F.binary_cross_entropy_with_logits(pred_pose_quality, target_pose_quality, weight=weight)


def train(
    *,
    data_root: str,
    index_file: str,
    save_dir: str = "./checkpoints",
    esm_path: str | None = None,
    epochs: int = 100,

    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    clip_grad: float = 10.0,
    hidden_dim: int = 128,
    num_gnn_blocks: int = 6,
    lig_atom_cont_count: int = len(LIGAND_ATOM_CONT_SCHEMA),
    lig_mol_cont_count: int = len(LIGAND_MOLECULE_CONT_SCHEMA),
    pro_atom_cont_count: int = len(PROTEIN_ATOM_CONT_SCHEMA),
    pro_res_cont_count: int = len(PROTEIN_RESIDUE_CONT_SCHEMA) + 960,
    esm_dim: int = 960,
    device: str | torch.device = "auto",
    pocket_radius: float | None = 10.0,
    protein_context_mode: str = "full",
    normalization_stats: dict | None = None,
    warmup_epochs: int = 20,
    rmsd_check_ratio: float = 0.2,
    accumulation_steps: int = 1,
    max_nodes_per_batch: int = 10000,
    val_max_nodes_per_batch: int | None = None,
    test_max_nodes_per_batch: int | None = None,
    topn_max_nodes_per_batch: int | None = None,
    ema_decay: float = 0.999,
    dataloader_num_workers: int = 4,
    dataloader_pin_memory: bool = True,
    dataloader_persistent_workers: bool = True,
    split_train_frac: float = 0.7,
    split_val_frac: float = 0.1,
    split_test_frac: float = 0.2,
    split_seed: int = 42,
    split_cache_file: str | None = None,
    force_resplit: bool = False,
    ablation_mode: str = "none",
    run_test_after_training: bool = True,
    test_topk_values: tuple[int, ...] = (1, 5, 10),
    test_pose_samples: int = 10,
    enable_oom_adaptive_batch: bool = True,
    oom_reduce_threshold: int = 3,
    oom_reduce_factor: float = 0.85,
    min_max_nodes_per_batch: int = 12000,
    enable_val_oom_adaptive_batch: bool = True,
    val_oom_reduce_threshold: int = 3,
    val_oom_reduce_factor: float = 0.85,
    min_val_max_nodes_per_batch: int | None = None,
    oom_recover_epochs: int = 3,
    oom_recover_factor: float = 1.1,
    center_proposal_weight: float = 0.15,
    center_positive_radius: float = 4.0,
    center_proposal_topk: int = 8,
    center_refine_topk: int = 3,
    center_nms_radius: float = 6.0,
    stage1_pose_samples: int = 2,
    stage2_pose_samples: int = 4,
    crop_candidate_topk: int = 8,
    disable_jitter_crop: bool = False,
    disable_hard_negative_crop: bool = False,
    pose_ranking_pair_weight: float = 0.2,
    pose_ranking_margin: float = 0.5,
    pose_bootstrap_weight: float = 0.05,
    pose_bootstrap_frequency: int = 25,
    pose_bootstrap_ode_steps: int = 10,
    enable_fusion_calibration: bool = True,
    val_ode_steps: int = 50,
    checkpoint_selection_mode: str = "composite",
    fusion_search_center_weights: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.5, 0.65),
    fusion_search_aff_weights: tuple[float, ...] = (0.0,),
    fusion_search_clash_weights: tuple[float, ...] = (0.0,),
    blind_pool_refresh_every: int = 5,
    blind_pool_start_epoch: int = 10,
    blind_pool_max_complexes: int = 500,
    blind_pool_cache_bce_weight: float = 0.5,
    blind_pool_cache_rank_weight: float = 1.0,
    blind_pool_pairs_per_complex: int = 4,
    run_name: str | None = None,
    run_log_file: str | None = None,
):
    """
    训练 EHFNet 模型

    Args:
        data_root: PDBBind 数据根目录
        index_file: 索引 CSV 文件路径
        save_dir: 模型保存目录
        esm_path: 预计算的 ESM 嵌入路径
        epochs: 训练轮数

        lr: 学习率
        weight_decay: 权重衰减
        clip_grad: 梯度裁剪阈值
        hidden_dim: 隐藏层维度
        num_gnn_blocks: GNN 块数量
        lig_atom_cont_count: 配体原子连续特征数量
        lig_mol_cont_count: 配体分子连续特征数量
        pro_atom_cont_count: 蛋白原子连续特征数量
        pro_res_cont_count: 蛋白残基连续特征数量
        esm_dim: ESM embedding 维度
        device: 训练设备 ("cpu", "cuda", "cuda:0", "cuda:1" 等)，默认为 "auto" (自动检测)
        pocket_radius: 运行时局部 docking 半径 (Å)
        protein_context_mode: 蛋白上下文缓存模式，full 表示缓存全蛋白并在运行时裁剪
        normalization_stats: 保留兼容参数；运行时会统一改用 train split 统计
        warmup_epochs: 空间课程学习预热轮数
        rmsd_check_ratio: 验证集中计算 RMSD 的样本比例 (0.0 ~ 1.0)
                          例如 0.1 表示随机抽取 10% 的 batch 进行耗时的 RMSD 推演
        accumulation_steps: 梯度累积步数。当显存较小时，可设为 2/4 模拟更大 batch_size
        max_nodes_per_batch: 训练集 DynamicBatchSampler 节点预算
        val_max_nodes_per_batch: 验证集节点预算（None 时使用 min(train_budget, 6000)）
        test_max_nodes_per_batch: 测试集节点预算（None 时沿用验证预算）
        topn_max_nodes_per_batch: Top-N 评估节点预算（None 时沿用测试预算）
        ema_decay: EMA 衰减率，默认 0.999；小规模试跑可设为 0.99 加快吸收
        dataloader_num_workers: DataLoader worker 数
        dataloader_pin_memory: 是否启用 pin_memory
        dataloader_persistent_workers: 是否启用 persistent_workers（仅 num_workers>0 时有效）
        split_train_frac: 训练集比例（建议 0.7）
        split_val_frac: 验证集比例（建议 0.1）
        split_test_frac: 测试集比例（建议 0.2）
        split_seed: Scaffold 划分随机种子
        split_cache_file: 划分缓存 JSON 路径；为 None 时自动生成默认路径
        force_resplit: 是否忽略缓存并强制重新划分
        ablation_mode: 消融模式（"none" 或 "inter_multiscale_off"）
        run_test_after_training: 训练结束后是否自动在 test 集上评估并输出报告
        test_topk_values: Top-N 成功率统计的 N 列表，如 (1,5,10)
        test_pose_samples: 每个复合物采样的候选 pose 数（应 >= max(test_topk_values)）
        enable_oom_adaptive_batch: 是否启用 OOM 触发的自动降批保护
        oom_reduce_threshold: 单个 epoch 触发多少次 OOM 后，下个 epoch 自动降低 batch 节点预算
        oom_reduce_factor: 自动降批系数，(0,1) 内有效（级联熔断时自动使用 min(factor, 0.7) 更激进的缩减）
        min_max_nodes_per_batch: 自动降批的下限，避免降得过小影响训练质量
        enable_val_oom_adaptive_batch: 是否启用验证阶段 OOM 触发的独立降批
        val_oom_reduce_threshold: 单个 epoch 验证 OOM 达到该阈值后，下轮降低验证预算
        val_oom_reduce_factor: 验证预算缩减系数，(0,1) 内有效
        min_val_max_nodes_per_batch: 验证自动降批下限，None 时默认与训练下限一致
        oom_recover_epochs: 连续多少个无 OOM epoch 后尝试回升 batch 预算
        oom_recover_factor: 回升系数 (>1)，如 1.1 表示每次回升 10%
    """

    # 1. 准备环境
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device(device)
        
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Using device: {device}")
    if run_name is not None:
        logger.info(f"Run name: {run_name}")
    if run_log_file is not None:
        logger.info(f"Run log file: {run_log_file}")

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)

    except Exception:
        pass

    # 2. 准备数据
    logger.info("Initializing Dataset...")
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
    if protein_context_mode not in {"full", "pocket"}:
        raise ValueError("protein_context_mode must be 'full' or 'pocket'.")

    interaction_profile = "atom_only" if ablation_mode == "inter_multiscale_off" else "full"

    dataset = PDBBindDataset(
        root=data_root,
        index_file=index_file,
        esm_root=esm_path,
        esm="auto",
        esm_dim=esm_dim,
        pocket_radius=None if protein_context_mode == "full" else pocket_radius,
        interaction_profile=interaction_profile,
    )
    graph_builder = dataset.graph_builder

    if not math.isclose(split_train_frac + split_val_frac + split_test_frac, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Split fractions must sum to 1.0, got "
            f"{split_train_frac + split_val_frac + split_test_frac:.6f}."
        )

    if split_cache_file is None:
        split_dir = Path(data_root) / "splits"
        split_name = (
            f"scaffold_{int(split_train_frac*100)}_"
            f"{int(split_val_frac*100)}_{int(split_test_frac*100)}_seed{split_seed}.json"
        )
        split_cache_file = str(split_dir / split_name)

    logger.info("Splitting dataset by Scaffold with persisted indices...")
    splitter = ScaffoldSplitter(include_chirality=False, seed=split_seed)

    split_indices: dict[str, list[int]]
    split_metadata: dict[str, Any] = {}
    if os.path.exists(split_cache_file) and not force_resplit:
        split_indices, split_metadata = ScaffoldSplitter.load_split(split_cache_file)
        max_idx = max(
            split_indices.get("train", [0])
            + split_indices.get("val", [0])
            + split_indices.get("test", [0])
        )
        cached_dataset_size = split_metadata.get("dataset_size")
        cached_index_file = split_metadata.get("index_file")
        cached_fractions = split_metadata.get("fractions", {})
        current_index_file = os.path.abspath(index_file)
        cached_index_file_abs = os.path.abspath(str(cached_index_file)) if cached_index_file is not None else None

        split_cache_mismatch = any([
            max_idx >= len(dataset),
            cached_dataset_size != len(dataset),
            cached_index_file_abs != current_index_file,
            split_metadata.get("seed") != split_seed,
            bool(split_metadata) and cached_fractions.get("train") != split_train_frac,
            bool(split_metadata) and cached_fractions.get("val") != split_val_frac,
            bool(split_metadata) and cached_fractions.get("test") != split_test_frac,
        ])

        if split_cache_mismatch:
            logger.warning(
                f"Cached split file {split_cache_file} is incompatible with current dataset configuration; regenerating. "
                f"(cached_dataset_size={cached_dataset_size}, current_dataset_size={len(dataset)}, "
                f"cached_index_file={cached_index_file}, current_index_file={current_index_file})"
            )
            split_indices = splitter.split_indices(
                dataset,
                frac_train=split_train_frac,
                frac_val=split_val_frac,
                frac_test=split_test_frac,
            )
            split_metadata = {
                "seed": split_seed,
                "include_chirality": False,
                "fractions": {
                    "train": split_train_frac,
                    "val": split_val_frac,
                    "test": split_test_frac,
                },
                "dataset_size": len(dataset),
                "index_file": current_index_file,
            }
            ScaffoldSplitter.save_split(split_cache_file, split_indices, metadata=split_metadata)
        else:
            logger.info(f"Loaded split indices from {split_cache_file}")
            logger.info(f"Split metadata: {split_metadata}")
    else:
        split_indices = splitter.split_indices(
            dataset,
            frac_train=split_train_frac,
            frac_val=split_val_frac,
            frac_test=split_test_frac,
        )
        split_metadata = {
            "seed": split_seed,
            "include_chirality": False,
            "fractions": {
                "train": split_train_frac,
                "val": split_val_frac,
                "test": split_test_frac,
            },
            "dataset_size": len(dataset),
            "index_file": os.path.abspath(index_file),
        }
        ScaffoldSplitter.save_split(split_cache_file, split_indices, metadata=split_metadata)
        logger.info(f"Saved split indices to {split_cache_file}")

    train_set, val_set, test_set = ScaffoldSplitter.subsets_from_indices(dataset, split_indices)
    train_indices = [int(i) for i in split_indices.get("train", [])]
    if not train_indices:
        raise ValueError("Train split is empty; cannot compute train-only normalization stats.")

    if normalization_stats:
        logger.warning("Ignoring externally supplied normalization_stats; using train-split-only statistics.")

    normalization_stats, train_affinity_stats = _compute_train_split_normalization_stats(
        dataset,
        train_indices,
        split_cache_file=split_cache_file,
    )
    normalization_stats["affinity"] = {
        "mean": torch.tensor(train_affinity_stats["mean"], dtype=torch.float32),
        "std": torch.tensor(train_affinity_stats["std"], dtype=torch.float32),
    }
    dataset.set_affinity_stats(train_affinity_stats)
    logger.info(
        "Using train-only affinity stats: mean=%.4f std=%.4f",
        train_affinity_stats["mean"],
        train_affinity_stats["std"],
    )

    logger.info(
        f"Final Dataset Sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}"
    )

    from torch_geometric.loader import DynamicBatchSampler

    if accumulation_steps < 1:
        logger.warning(f"Invalid accumulation_steps={accumulation_steps}, fallback to 1.")
        accumulation_steps = 1

    if not (0.0 < oom_reduce_factor < 1.0):
        logger.warning(f"Invalid oom_reduce_factor={oom_reduce_factor}, fallback to 0.8.")
        oom_reduce_factor = 0.8

    configured_train_max_nodes_per_batch = max(1, int(max_nodes_per_batch))
    configured_val_max_nodes_per_batch = (
        max(1, int(val_max_nodes_per_batch))
        if val_max_nodes_per_batch is not None
        else min(configured_train_max_nodes_per_batch, 6000)
    )
    configured_test_max_nodes_per_batch = (
        max(1, int(test_max_nodes_per_batch))
        if test_max_nodes_per_batch is not None
        else configured_val_max_nodes_per_batch
    )
    configured_topn_max_nodes_per_batch = (
        max(1, int(topn_max_nodes_per_batch))
        if topn_max_nodes_per_batch is not None
        else configured_test_max_nodes_per_batch
    )
    # [修复] 降低边预算因子
    # 旧值 60 + protein_atom k=128 bug 导致 batch 静态边已接近显存上限，
    # 而编码器 forward 中还会通过 radius() 动态创建跨图边（对 Sampler 不可见），
    # 使实际 GPU 边数远超预算 → OOM。
    # 修复 k→32 后静态边/图大幅下降，Sampler 会装入更多图，
    # 必须同步下调因子为动态边预留 ~40-50% 显存 headroom。
    # 注意：batch 变小不影响精度——accumulation_steps=8 保证了有效梯度规模。
    train_edge_budget_factor = 40
    eval_edge_guard_headroom = 1.5

    def _safe_metric(value: Any, default: float, *, higher_is_better: bool = True) -> float:
        try:
            metric = float(value)
        except Exception:
            return default

        if math.isnan(metric) or math.isinf(metric):
            return default

        return metric

    def _build_selection_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        success_2a = _safe_metric(metrics.get("reranked_top1_success_2a", metrics.get("success_2a")), 0.0)
        success_5a = _safe_metric(metrics.get("reranked_top5_success_2a", metrics.get("success_5a")), 0.0)
        oracle_top5_success_2a = _safe_metric(metrics.get("oracle_top5_success_2a"), 0.0)
        mean_rmsd = _safe_metric(
            metrics.get("reranked_top1_mean_best_rmsd", metrics.get("mean_rmsd_final")),
            1e9,
            higher_is_better=False,
        )
        val_loss = _safe_metric(metrics.get("local_val_loss", metrics.get("val_loss")), 1e9, higher_is_better=False)
        center_recall = _safe_metric(metrics.get("center_recall@8"), 0.0)
        proposal_gap = _safe_metric(metrics.get("proposal_gap"), 100.0, higher_is_better=False)
        ranking_gap = _safe_metric(metrics.get("ranking_gap"), 100.0, higher_is_better=False)
        composite_score = (
            1.75 * success_2a
            + 0.35 * success_5a
            + 0.20 * oracle_top5_success_2a
            + 0.10 * center_recall
            - 0.75 * mean_rmsd
            - 0.15 * proposal_gap
            - 0.20 * ranking_gap
        )
        blind_combo_score = 1.0 * success_2a + 0.35 * oracle_top5_success_2a - 0.10 * ranking_gap

        return {
            "composite_score": composite_score,
            "blind_combo_score": blind_combo_score,
            "success_2a": success_2a,
            "success_5a": success_5a,
            "oracle_top5_success_2a": oracle_top5_success_2a,
            "mean_rmsd": mean_rmsd,
            "val_loss": val_loss,
            "center_recall@8": center_recall,
            "proposal_gap": proposal_gap,
            "ranking_gap": ranking_gap,
        }

    def _resolve_selection_rule() -> tuple[str, bool, str]:
        mapping = {
            "composite": ("composite_score", True, "Composite"),
            "reranked_top1_success_2a": ("success_2a", True, "Rerank@1<2A"),
            "reranked_top5_success_2a": ("success_5a", True, "Rerank@5<2A"),
            "reranked_top1_plus_oracle_top5": ("blind_combo_score", True, "Rerank@1 + Oracle@5"),
        }
        if checkpoint_selection_mode not in mapping:
            raise ValueError(
                "checkpoint_selection_mode must be one of "
                f"{tuple(mapping.keys())}, got {checkpoint_selection_mode!r}"
            )
        return mapping[checkpoint_selection_mode]

    def _is_better_checkpoint(
        candidate: dict[str, float],
        incumbent: dict[str, float] | None,
        *,
        primary_key: str,
        primary_higher_is_better: bool,
        tol: float = 1e-6,
    ) -> bool:
        if incumbent is None:
            return True

        candidate_primary = candidate[primary_key]
        incumbent_primary = incumbent[primary_key]

        if primary_higher_is_better:
            if candidate_primary > incumbent_primary + tol:
                return True
            if candidate_primary < incumbent_primary - tol:
                return False
        else:
            if candidate_primary < incumbent_primary - tol:
                return True
            if candidate_primary > incumbent_primary + tol:
                return False

        if candidate["success_2a"] > incumbent["success_2a"] + tol:
            return True
        if candidate["success_2a"] < incumbent["success_2a"] - tol:
            return False

        if candidate["success_5a"] > incumbent["success_5a"] + tol:
            return True
        if candidate["success_5a"] < incumbent["success_5a"] - tol:
            return False

        if candidate["mean_rmsd"] < incumbent["mean_rmsd"] - tol:
            return True
        if candidate["mean_rmsd"] > incumbent["mean_rmsd"] + tol:
            return False

        if candidate["val_loss"] < incumbent["val_loss"] - tol:
            return True
        if candidate["val_loss"] > incumbent["val_loss"] + tol:
            return False

        return False

    def _annotate_loss_context(batch_obj: Any, *, current_epoch: int, total_epochs_count: int, warmup_epochs_count: int, training: bool) -> None:
        apply_loss_context(
            batch_obj,
            current_epoch=current_epoch,
            total_epochs_count=total_epochs_count,
            warmup_epochs_count=warmup_epochs_count,
            training=training,
        )

    def _compose_checkpoint(
        *,
        epoch_idx: int,
        avg_train_loss_value: float,
        val_metrics_obj: dict[str, Any],
        selection_metrics: dict[str, float],
    ) -> dict[str, Any]:
        model_config = build_model_config(
            hidden_dim=hidden_dim,
            time_dim=hidden_dim,
            num_gnn_blocks=num_gnn_blocks,
            lig_atom_cont_count=lig_atom_cont_count,
            lig_mol_cont_count=lig_mol_cont_count,
            pro_atom_cont_count=pro_atom_cont_count,
            pro_res_cont_count=pro_res_cont_count,
            esm_dim=esm_dim,
            interaction_profile=interaction_profile,
        )
        return {
            "epoch": epoch_idx,
            "run_name": run_name,
            "run_log_file": run_log_file,
            "model_config": model_config,
            "feature_signature": build_feature_signature(esm_dim=esm_dim),
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.module.state_dict() if ema_model is not None else model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_rmsd": best_rmsd,
            "avg_train_loss": avg_train_loss_value,
            "val_metrics": dict(val_metrics_obj),
            "selection_metrics": dict(selection_metrics),
            "fusion_weights": dict(current_fusion_weights),
        }

    effective_min_train_nodes_per_batch = max(1, int(min_max_nodes_per_batch))
    effective_min_val_nodes_per_batch = max(
        1,
        int(min_val_max_nodes_per_batch)
        if min_val_max_nodes_per_batch is not None
        else int(min_max_nodes_per_batch),
    )

    if effective_min_train_nodes_per_batch > configured_train_max_nodes_per_batch:
        logger.warning(
            f"min_max_nodes_per_batch ({effective_min_train_nodes_per_batch}) is greater than "
            f"max_nodes_per_batch ({configured_train_max_nodes_per_batch}); clamping min to max."
        )
        effective_min_train_nodes_per_batch = configured_train_max_nodes_per_batch

    if effective_min_val_nodes_per_batch > configured_val_max_nodes_per_batch:
        logger.warning(
            f"min_val_max_nodes_per_batch ({effective_min_val_nodes_per_batch}) is greater than "
            f"val_max_nodes_per_batch ({configured_val_max_nodes_per_batch}); clamping min to val max."
        )
        effective_min_val_nodes_per_batch = configured_val_max_nodes_per_batch

    if not (0.0 < val_oom_reduce_factor < 1.0):
        logger.warning(f"Invalid val_oom_reduce_factor={val_oom_reduce_factor}, fallback to 0.8.")
        val_oom_reduce_factor = 0.8

    current_train_max_nodes_per_batch = configured_train_max_nodes_per_batch
    current_val_max_nodes_per_batch = configured_val_max_nodes_per_batch
    persistent_workers = bool(dataloader_persistent_workers and dataloader_num_workers > 0)

    # 用于追踪旧 loader 引用，确保重建时显式销毁 persistent_workers
    _prev_loaders: list[DataLoader] = []

    def _build_loaders(train_max_nodes: int, val_max_nodes: int) -> tuple[DataLoader, DataLoader]:
        # 显式销毁旧 loader，释放 persistent_workers 进程
        for old_loader in _prev_loaders:
            del old_loader
        _prev_loaders.clear()
        gc.collect()

        train_edge_budget = max(1, int(train_max_nodes * train_edge_budget_factor))
        val_edge_budget = max(1, int(val_max_nodes * train_edge_budget_factor))
        logger.info(
            f"Using DynamicBatchSampler budgets: train_max_num={train_edge_budget} (mode=edge), "
            f"val_max_num={val_edge_budget} (mode=edge)."
        )

        train_sampler = DynamicBatchSampler(
            cast(Any, train_set),
            max_num=train_edge_budget,
            mode="edge",
            shuffle=True,
        )
        train_loader_local = DataLoader(
            train_set,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=train_sampler,
        )

        # [修复] 验证集也使用 edge 模式进行批处理
        # 旧逻辑使用 mode="node"，导致节点数受控但边数完全不可控，
        # 几乎 100% 的验证 batch 都因超出 edge_guard_limit 而被 preflight skip
        val_sampler = DynamicBatchSampler(
            cast(Any, val_set),
            max_num=val_edge_budget,
            mode="edge",
            shuffle=False,
        )
        val_loader_local = DataLoader(
            val_set,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=val_sampler,
        )

        _prev_loaders.extend([train_loader_local, val_loader_local])
        return train_loader_local, val_loader_local

    def _build_eval_loader(subset: Any, max_nodes: int) -> DataLoader:
        eval_edge_budget = max(1, int(max_nodes * train_edge_budget_factor))
        eval_sampler = DynamicBatchSampler(
            cast(Any, subset),
            max_num=eval_edge_budget,
            mode="edge",
            shuffle=False,
        )
        return DataLoader(
            subset,
            collate_fn=collator.collate,
            num_workers=dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=dataloader_pin_memory,
            batch_sampler=eval_sampler,
        )

    train_loader, val_loader = _build_loaders(
        current_train_max_nodes_per_batch,
        current_val_max_nodes_per_batch,
    )

    # [新增逻辑] 动态计算 Batch 数量 (DynamicBatchSampler 没有固定的 len)
    try:
        total_val_batches = len(val_loader)

    except ValueError:
        total_val_batches = max(1, len(val_set) // 4)
        
    rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)
    
    # 确保至少检查 1 个 batch (如果 ratio > 0)
    if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
        rmsd_check_batches = 1
    
    logger.info(f"Validation Sampling: Check RMSD for {rmsd_check_batches}/{total_val_batches} batches ({rmsd_check_ratio*100:.1f}%)")
    logger.info(
        "Evaluation budgets: "
        f"test_nodes={configured_test_max_nodes_per_batch}, "
        f"test_edges={max(1, int(configured_test_max_nodes_per_batch * train_edge_budget_factor))}, "
        f"topn_nodes={configured_topn_max_nodes_per_batch}, "
        f"topn_edges={max(1, int(configured_topn_max_nodes_per_batch * train_edge_budget_factor))}."
    )

    # 3. 准备模型组件
    logger.info("Initializing Model & Flow Components...")

    model = EHFNet(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        interaction_profile=interaction_profile,
        normalization_stats=normalization_stats,
    ).to(device)

    matcher = ConditionalFlowMatcher(
        sigma_min=1e-3,
        warmup_epochs=warmup_epochs,
    )
    criterion = FlowMatchingLoss().to(device)
    # 速度分解由 matcher 内部完成，trainer 不持有分解器

    # 4. 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Warmup + 余弦退火（Step 级），防止初期梯度冲击 + 中后期平滑收敛
    try:
        total_train_batches = len(train_loader)
    except ValueError:
        total_train_batches = max(1, len(train_set) // 4)
    updates_per_epoch = math.ceil(total_train_batches / accumulation_steps)
    total_steps = epochs * updates_per_epoch
    warmup_steps = max(1, warmup_epochs) * updates_per_epoch  # LR warmup 与 Curriculum 同步
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps],
    )
    logger.info(
        f"LR scheduler initialized: total_steps={total_steps}, warmup_steps={warmup_steps}."
    )

    ema_model: AveragedModel | None = None

    # 5. 训练循环
    best_val_loss = float("inf")
    best_rmsd = float("inf")
    best_composite_metrics: dict[str, float] | None = None
    best_success2a_metrics: dict[str, float] | None = None
    best_rmsd_metrics: dict[str, float] | None = None
    best_selected_metrics: dict[str, float] | None = None
    current_fusion_weights = dict(DEFAULT_FUSION_WEIGHTS)
    total_oom_batches = 0
    consecutive_clean_epochs = 0  # 连续无 OOM 的 epoch 计数，用于回升决策
    consecutive_clean_val_epochs = 0
    oom_blacklisted_pdb_ids: set[str] = set()
    oom_counts_by_pdb: dict[str, int] = {}
    OOM_BLACKLIST_THRESHOLD = 2
    clone_safety_checked = False
    selected_primary_key, selected_higher_is_better, selected_metric_label = _resolve_selection_rule()
    blind_pool_cache_dir = os.path.join(save_dir, "blind_pool_cache")
    os.makedirs(blind_pool_cache_dir, exist_ok=True)
    blind_pool_compatibility = build_blind_pool_compatibility(
        esm_dim=esm_dim,
        processed_dir=dataset.processed_dir,
        index_file=dataset.index_file,
        interaction_profile=interaction_profile,
    )
    cached_blind_pool: list[dict[str, Any]] = load_blind_pool(
        blind_pool_cache_dir,
        expected_compatibility=blind_pool_compatibility,
    )
    if cached_blind_pool:
        logger.info("Loaded existing blind pool: %d complexes.", len(cached_blind_pool))
    best_selected_updated_this_epoch = False

    def _extract_batch_pdb_ids(batch_obj: Any) -> list[str]:
        pdb_attr = getattr(batch_obj, "pdb_id", None)

        if pdb_attr is None:
            return []

        if isinstance(pdb_attr, str):
            return [pdb_attr]

        if isinstance(pdb_attr, (list, tuple)):
            return [str(pid) for pid in pdb_attr]

        try:
            return [str(pid) for pid in list(pdb_attr)]
        except Exception:
            return [str(pdb_attr)]

    def _estimate_batch_total_edges(batch_obj: Any) -> int:
        total_edges = 0

        edge_types = getattr(batch_obj, "edge_types", None)
        if not edge_types:
            return 0

        for edge_type in edge_types:
            edge_store = batch_obj[edge_type]
            edge_index = getattr(edge_store, "edge_index", None)
            if edge_index is not None and edge_index.ndim == 2:
                total_edges += int(edge_index.size(1))

        return total_edges

    for epoch in range(epochs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        criterion.train()

        train_loss_meter = 0.0
        pbar = tqdm(total=len(train_set), desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="graphs")
        
        actual_batches = 0
        epoch_oom_batches = 0
        epoch_edge_guard_skips = 0
        accumulated_graphs = 0  # 当前累积周期内的总图数
        consecutive_oom = 0     # 连续 OOM 计数，用于级联熔断
        CIRCUIT_BREAKER_LIMIT = 10  # 连续 OOM 达到此值则熔断当前 epoch
        ENERGY_NAN_FAILFAST_LIMIT = 8
        epoch_fused = False     # 本 epoch 是否被熔断
        consecutive_energy_nan_skips = 0
        optimizer.zero_grad()   # 在循环外初始化梯度清零
        epoch_proposal_losses: list[float] = []
        epoch_local_losses: list[float] = []
        epoch_proposal_residues: list[float] = []
        epoch_local_residues: list[float] = []
        epoch_rank_pair_counts = {
            "same_center": 0,
            "wrong_center_low_clash": 0,
            "misleading_center": 0,
            "misleading_affinity": 0,
        }
        epoch_rank_oom_skips = 0
        epoch_rank_peak_mem_mb = 0.0
        epoch_energy_nan_skips = 0

        for batch_idx, batch in enumerate(train_loader):
            num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
            pbar.update(num_graphs)
            batch_pdb_ids = _extract_batch_pdb_ids(batch)

            if oom_blacklisted_pdb_ids and batch_pdb_ids:
                if any(pid in oom_blacklisted_pdb_ids for pid in batch_pdb_ids):
                    blacklisted_in_batch = [pid for pid in batch_pdb_ids if pid in oom_blacklisted_pdb_ids]
                    logger.warning(
                        f"Batch {batch_idx}: skipping batch containing OOM-blacklisted samples "
                        f"({len(blacklisted_in_batch)}/{len(batch_pdb_ids)} in batch)."
                    )
                    consecutive_oom = 0
                    continue

            total_edges_cpu = _estimate_batch_total_edges(batch)
            edge_guard_limit = max(1, int(current_train_max_nodes_per_batch * train_edge_budget_factor))
            if total_edges_cpu > edge_guard_limit:
                epoch_edge_guard_skips += 1
                logger.warning(
                    f"Batch {batch_idx}: preflight skip due to edge-heavy batch "
                    f"(total_edges={total_edges_cpu} > limit={edge_guard_limit})."
                )
                consecutive_oom = 0
                continue
            
            # 【全覆盖 OOM 安全网】
            # 包裹 batch.to(device) → 采样 → 前向 → 损失 → 反向 的完整流程
            # OOM 可能发生在任何 GPU 操作点：数据迁移、EGNN 消息传递、梯度计算
            try:
                source_batch = batch
                if not clone_safety_checked:
                    clone_safety_checked = True
                    logger.info(
                        "HeteroData.clone() tensor storage check | shared_ligand_pos=%s",
                        clone_shares_tensor_storage(source_batch),
                    )
                ligand_centers = scatter_mean(
                    source_batch["ligand_atom"].pos,
                    source_batch["ligand_atom"].batch,
                    dim=0,
                    dim_size=num_graphs,
                )
                proposal_loss = compute_proposal_loss(
                    model,
                    source_batch,
                    device=device,
                    positive_radius=center_positive_radius,
                )
                proposal_logits, residue_pos_for_crop, residue_batch_for_crop, _ = predict_center_proposal_logits(
                    model,
                    source_batch,
                    device=device,
                )
                train_progress = 1.0 if epochs <= 1 else epoch / max(1, epochs - 1)
                proposal_logits_cpu = proposal_logits.detach().cpu().view(-1)
                residue_pos_cpu = residue_pos_for_crop.detach().cpu()
                residue_batch_cpu = residue_batch_for_crop.detach().cpu()
                wrong_centers_cpu, wrong_center_scores_cpu, wrong_center_valid_cpu = select_wrong_center_candidates(
                    ligand_centers.detach().cpu(),
                    proposal_logits_cpu,
                    residue_pos_cpu,
                    residue_batch_cpu,
                    positive_radius=center_positive_radius,
                    bucket_topk=crop_candidate_topk,
                    weighted_sampling=True,
                )
                bootstrap_centers_cpu = select_bootstrap_blind_centers(
                    ligand_centers.detach().cpu(),
                    proposal_logits_cpu,
                    residue_pos_cpu,
                    residue_batch_cpu,
                    positive_radius=center_positive_radius,
                    bucket_topk=crop_candidate_topk,
                )
                proposal_top_scores_cpu = torch.full((num_graphs,), -1e9, dtype=proposal_logits_cpu.dtype)
                for graph_idx in range(num_graphs):
                    graph_mask = residue_batch_cpu == graph_idx
                    graph_logits = proposal_logits_cpu[graph_mask]
                    if graph_logits.numel() > 0:
                        proposal_top_scores_cpu[graph_idx] = graph_logits.max()
                if ablation_mode == "gt_only_crop":
                    crop_centers_cpu = ligand_centers.detach().cpu()
                    crop_modes = ["gt_forced"] * num_graphs
                else:
                    crop_centers_cpu, crop_modes = select_training_crop_centers(
                        ligand_centers.detach().cpu(),
                        proposal_logits_cpu,
                        residue_pos_cpu,
                        residue_batch_cpu,
                        progress=train_progress,
                        positive_radius=center_positive_radius,
                        bucket_topk=crop_candidate_topk,
                        weighted_sampling=True,
                        disable_jitter=disable_jitter_crop,
                        disable_hard_negative=disable_hard_negative_crop,
                    )
                local_batch = build_local_batch_from_centers(
                    source_batch,
                    centers=crop_centers_cpu,
                    crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                    graph_builder=graph_builder,
                    collator=collator,
                )
                batch = local_batch.to(device)
                crop_centers = crop_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                wrong_center_valid = wrong_center_valid_cpu.to(device=device)
                wrong_center_scores = wrong_center_scores_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                wrong_centers = wrong_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                bootstrap_centers = bootstrap_centers_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                proposal_top_scores = proposal_top_scores_cpu.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
                epoch_proposal_losses.append(float(proposal_loss.detach().item()))
                epoch_proposal_residues.append(
                    float(source_batch["protein_residue"].pos.size(0)) / max(1, num_graphs)
                )
                epoch_local_residues.append(
                    float(local_batch["protein_residue"].pos.size(0)) / max(1, num_graphs)
                )
                _annotate_loss_context(
                    batch,
                    current_epoch=epoch,
                    total_epochs_count=epochs,
                    warmup_epochs_count=warmup_epochs,
                    training=True,
                )

                # 流匹配训练步骤
                # 生成训练目标不需要梯度
                with torch.no_grad():
                    x_1 = batch["ligand_atom"].pos
                    t, x_t, targets = matcher.sample_location_and_target(
                        x_1=x_1,
                        data=batch,
                        current_epoch=epoch,
                        total_epochs=epochs,
                        placement_centers=crop_centers,
                    )

                batch["ligand_atom"].pos = x_t
                batch.t = t

                # FP32 前向传播
                predictions = model(batch, t)

                # 补充结合能 target
                targets["binding_affinity_target"] = batch.get("y_energy", None)
                targets["pose_quality_target"] = compute_pose_quality_target(x_t, x_1, batch["ligand_atom"].batch)

                loss_dict = criterion(predictions, targets, batch)
                epoch_local_losses.append(float(loss_dict["total"].detach().item()))
                energy_nan_this_batch = int(loss_dict.get("energy_nan_skipped", torch.tensor(0.0)).item())
                epoch_energy_nan_skips += energy_nan_this_batch
                if energy_nan_this_batch > 0:
                    consecutive_energy_nan_skips += 1
                    if consecutive_energy_nan_skips >= ENERGY_NAN_FAILFAST_LIMIT:
                        raise RuntimeError(
                            f"Energy head produced non-finite affinity values on "
                            f"{consecutive_energy_nan_skips} consecutive batches "
                            f"(epoch={epoch+1}, batch={batch_idx})."
                        )
                else:
                    consecutive_energy_nan_skips = 0
                loss_pose_rank = torch.tensor(0.0, device=device)
                if pose_ranking_pair_weight > 0.0:
                    try:
                        rank_terms: list[torch.Tensor] = []
                        current_rank_logit = select_pose_ranking_logit(predictions)
                        # same-center 好/坏 pose pair
                        same_center_batch = batch.clone()
                        with torch.no_grad():
                            t_same = torch.clamp(
                                t * (0.25 + 0.45 * torch.rand_like(t)),
                                min=1e-3,
                                max=1.0 - 1e-3,
                            )
                            _, x_t_same, _ = matcher.sample_location_and_target(
                                x_1=x_1,
                                data=same_center_batch,
                                current_epoch=epoch,
                                total_epochs=epochs,
                                placement_centers=crop_centers,
                                t_override=t_same,
                            )
                        same_center_batch["ligand_atom"].pos = x_t_same
                        same_center_batch.t = t_same
                        same_center_pred = model(same_center_batch, t_same)
                        pose_quality_same = compute_pose_quality_target(x_t_same, x_1, batch["ligand_atom"].batch)
                        same_center_rank_logit = select_pose_ranking_logit(same_center_pred)
                        loss_same, count_same = compute_pairwise_pose_ranking_loss(
                            current_rank_logit,
                            targets["pose_quality_target"],
                            same_center_rank_logit,
                            pose_quality_same,
                            margin=pose_ranking_margin,
                        )
                        if count_same > 0:
                            rank_terms.append(loss_same)
                            epoch_rank_pair_counts["same_center"] += count_same

                        # wrong-center but clash small / center-high / affinity-not-bad
                        if bool(wrong_center_valid.any()):
                            wrong_local_batch = build_local_batch_from_centers(
                                source_batch,
                                centers=wrong_centers_cpu,
                                crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                                graph_builder=graph_builder,
                                collator=collator,
                            ).to(device)
                            x_1_wrong = wrong_local_batch["ligand_atom"].pos
                            with torch.no_grad():
                                t_wrong = torch.clamp(
                                    t * (0.65 + 0.25 * torch.rand_like(t)),
                                    min=1e-3,
                                    max=0.9,
                                )
                                _, x_t_wrong, _ = matcher.sample_location_and_target(
                                    x_1=x_1_wrong,
                                    data=wrong_local_batch,
                                    current_epoch=epoch,
                                    total_epochs=epochs,
                                    placement_centers=wrong_centers,
                                    t_override=t_wrong,
                                )
                            wrong_local_batch["ligand_atom"].pos = x_t_wrong
                            wrong_local_batch.t = t_wrong
                            wrong_pred = model(wrong_local_batch, t_wrong)
                            wrong_rank_logit = select_pose_ranking_logit(wrong_pred)
                            pose_quality_wrong = compute_pose_quality_target(
                                x_t_wrong,
                                x_1_wrong,
                                wrong_local_batch["ligand_atom"].batch,
                            )
                            anchor_clash = predictions.get("steric_clash_batch")
                            wrong_clash = wrong_pred.get("steric_clash_batch")
                            if anchor_clash is None:
                                anchor_clash = torch.zeros_like(t)
                            if wrong_clash is None:
                                wrong_clash = torch.zeros_like(t)
                            low_clash_mask = wrong_center_valid & (
                                wrong_clash.view(-1) <= (anchor_clash.view(-1) + 1.0)
                            )
                            loss_wrong, count_wrong = compute_pairwise_pose_ranking_loss(
                                current_rank_logit,
                                targets["pose_quality_target"],
                                wrong_rank_logit,
                                pose_quality_wrong,
                                margin=pose_ranking_margin,
                                extra_mask=low_clash_mask,
                            )
                            if count_wrong > 0:
                                rank_terms.append(loss_wrong)
                                epoch_rank_pair_counts["wrong_center_low_clash"] += count_wrong

                            misleading_center_mask = low_clash_mask & (
                                wrong_center_scores >= (proposal_top_scores - 0.25)
                            )
                            loss_center_hard, count_center_hard = compute_pairwise_pose_ranking_loss(
                                current_rank_logit,
                                targets["pose_quality_target"],
                                wrong_rank_logit,
                                pose_quality_wrong,
                                margin=pose_ranking_margin,
                                extra_mask=misleading_center_mask,
                            )
                            if count_center_hard > 0:
                                rank_terms.append(loss_center_hard)
                                epoch_rank_pair_counts["misleading_center"] += count_center_hard

                            anchor_aff = predictions.get("binding_affinity")
                            wrong_aff = wrong_pred.get("binding_affinity")
                            if anchor_aff is not None and wrong_aff is not None:
                                misleading_aff_mask = low_clash_mask & (
                                    wrong_aff.view(-1) >= (anchor_aff.view(-1) - 0.25)
                                )
                                loss_aff_hard, count_aff_hard = compute_pairwise_pose_ranking_loss(
                                    current_rank_logit,
                                    targets["pose_quality_target"],
                                    wrong_rank_logit,
                                    pose_quality_wrong,
                                    margin=pose_ranking_margin,
                                    extra_mask=misleading_aff_mask,
                                )
                                if count_aff_hard > 0:
                                    rank_terms.append(loss_aff_hard)
                                    epoch_rank_pair_counts["misleading_affinity"] += count_aff_hard

                            del wrong_local_batch, x_1_wrong, t_wrong, x_t_wrong, wrong_pred, pose_quality_wrong

                        if rank_terms:
                            loss_pose_rank = torch.stack(rank_terms).mean()
                        loss_dict["loss_pose_rank"] = loss_pose_rank.detach()
                        loss_dict["rank_pairs_same_center"] = torch.tensor(
                            epoch_rank_pair_counts["same_center"], device=device
                        )
                        loss_dict["rank_pairs_wrong_center"] = torch.tensor(
                            epoch_rank_pair_counts["wrong_center_low_clash"], device=device
                        )
                        if torch.cuda.is_available():
                            epoch_rank_peak_mem_mb = max(
                                epoch_rank_peak_mem_mb,
                                float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
                            )
                        del same_center_batch, same_center_pred, pose_quality_same, t_same, x_t_same
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"Batch {batch_idx}: ranking forward OOM, skipping pairwise loss.")
                        loss_pose_rank = torch.tensor(0.0, device=device)
                        epoch_rank_oom_skips += 1
                        gc.collect()
                        torch.cuda.empty_cache()

                loss_pose_bootstrap = torch.tensor(0.0, device=device)
                teacher_model = ema_model if ema_model is not None else model
                if pose_bootstrap_weight > 0.0 and should_run_bootstrap(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_epochs=epochs,
                    frequency=pose_bootstrap_frequency,
                    start_ratio=0.30,
                ):
                    loss_pose_bootstrap = compute_bootstrap_pose_quality_loss(
                        student_model=model,
                        teacher_model=teacher_model,
                        matcher=matcher,
                        source_batch=source_batch,
                        placement_centers=bootstrap_centers,
                        epoch=epoch,
                        ode_steps=pose_bootstrap_ode_steps,
                        graph_builder=graph_builder,
                        collator=collator,
                        crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                    )
                    loss_dict["loss_pose_bootstrap"] = loss_pose_bootstrap.detach()

                loss_dict["loss_center_proposal"] = proposal_loss.detach()
                loss_dict["weight_center_proposal"] = torch.tensor(center_proposal_weight, device=device)
                loss_dict["weight_pose_rank"] = torch.tensor(pose_ranking_pair_weight, device=device)
                loss_dict["weight_pose_bootstrap"] = torch.tensor(pose_bootstrap_weight, device=device)
                loss = (
                    loss_dict["total"]
                    + center_proposal_weight * proposal_loss
                    + pose_ranking_pair_weight * loss_pose_rank
                    + pose_bootstrap_weight * loss_pose_bootstrap
                )

                # 防御性检查
                if loss.grad_fn is None:
                    logger.warning(f"Batch {batch_idx}: loss has no grad_fn, skipping.")
                    continue

                if torch.isnan(loss) or loss > 200:
                    logger.warning(f"{'NaN' if torch.isnan(loss) else 'Huge'} Loss on batch {batch_idx}, skipping.")
                    for k, v in loss_dict.items():
                        logger.warning(f"  {k}: {v}")
                    continue

                # 样本级梯度累积
                loss_sum = loss * num_graphs
                loss_sum.backward()

            except torch.cuda.OutOfMemoryError:
                epoch_oom_batches += 1
                total_oom_batches += 1
                consecutive_oom += 1

                # 【分级 CUDA 恢复】
                # Level 1: 基础清理（首次 OOM）
                optimizer.zero_grad(set_to_none=True)
                accumulated_graphs = 0
                # 必须删除所有引用 GPU tensor 的局部变量，否则碎片无法回收
                batch = source_batch = local_batch = None  # type: ignore[assignment]
                predictions = loss_dict = loss = loss_sum = None  # type: ignore[assignment]
                targets = x_1 = x_t = t = None  # type: ignore[assignment]
                proposal_loss = ligand_centers = proposal_logits = None  # type: ignore[assignment]
                proposal_logits_cpu = proposal_top_scores = proposal_top_scores_cpu = None  # type: ignore[assignment]
                residue_pos_for_crop = residue_batch_for_crop = None  # type: ignore[assignment]
                residue_pos_cpu = residue_batch_cpu = None  # type: ignore[assignment]
                crop_centers = crop_centers_cpu = None  # type: ignore[assignment]
                wrong_centers = wrong_centers_cpu = None  # type: ignore[assignment]
                wrong_center_scores = wrong_center_scores_cpu = None  # type: ignore[assignment]
                wrong_center_valid = wrong_center_valid_cpu = None  # type: ignore[assignment]
                bootstrap_centers = bootstrap_centers_cpu = None  # type: ignore[assignment]
                gc.collect()
                torch.cuda.empty_cache()

                # Level 2: 深度恢复（连续 OOM >= 3）
                if consecutive_oom >= 3:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    torch.cuda.empty_cache()

                # Level 3: 模型参数 CPU 往返（连续 OOM >= 5）—— 彻底重组显存布局
                if consecutive_oom == 5:
                    logger.warning(
                        f"Batch {batch_idx}: {consecutive_oom} consecutive OOMs, "
                        f"skipping model CPU roundtrip to avoid secondary OOM during restore."
                    )

                newly_blacklisted: list[str] = []
                if batch_pdb_ids:
                    for pid in batch_pdb_ids:
                        count = oom_counts_by_pdb.get(pid, 0) + 1
                        oom_counts_by_pdb[pid] = count
                        if count >= OOM_BLACKLIST_THRESHOLD and pid not in oom_blacklisted_pdb_ids:
                            oom_blacklisted_pdb_ids.add(pid)
                            newly_blacklisted.append(pid)

                if newly_blacklisted:
                    consecutive_oom = 0
                    logger.warning(
                        f"Batch {batch_idx}: blacklisted {len(newly_blacklisted)} repeatedly OOM samples; "
                        f"total blacklisted={len(oom_blacklisted_pdb_ids)}."
                    )

                # 【级联熔断器】连续 OOM 达到阈值，立即退出当前 epoch
                if consecutive_oom >= CIRCUIT_BREAKER_LIMIT:
                    logger.error(
                        f"Epoch {epoch+1}: circuit breaker triggered after {consecutive_oom} "
                        f"consecutive OOMs at batch {batch_idx}. Breaking out of epoch."
                    )
                    epoch_fused = True
                    break

                if consecutive_oom <= 2:
                    logger.warning(
                        f"Batch {batch_idx}: CUDA OOM, skipping and clearing cache "
                        f"(batch_total_edges={total_edges_cpu}, edge_guard_limit={edge_guard_limit})."
                    )
                continue

            # OOM 恢复：成功处理一个 batch 后重置连续 OOM 计数
            consecutive_oom = 0
            actual_batches += 1
            accumulated_graphs += num_graphs

            # 仅在完成一个完整累积周期后才更新参数
            is_last_in_cycle = (batch_idx + 1) % accumulation_steps == 0

            if is_last_in_cycle and accumulated_graphs > 0:
                
                # 将累积梯度除以真实图总数，得到无偏的样本级平均梯度
                for param in model.parameters():

                    if param.grad is not None:
                        param.grad /= accumulated_graphs

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                # Inf/NaN 梯度直接跳过，防止权重被污染

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Batch {batch_idx}: grad_norm={grad_norm:.4g}, skipping optimizer step.")

                else:
                    optimizer.step()
                    # 惰性构建 EMA（首次 step 后 lazy 参数已全部初始化）
                    if ema_model is None:
                        ema_model = AveragedModel(
                            model,
                            avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                        )
                    ema_model.update_parameters(model)
                    scheduler.step()

                # 清零梯度和计数器，开始新的累积周期
                optimizer.zero_grad()
                accumulated_graphs = 0

            # 记录日志
            train_loss_meter += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L_tr": f"{loss_dict.get('loss_trans', torch.tensor(0)).item():.3f}",
                    "L_rot": f"{loss_dict.get('loss_rot', torch.tensor(0)).item():.3f}",
                    "L_tor": f"{loss_dict.get('loss_torsion', torch.tensor(0)).item():.3f}",
                    "L_ene": f"{loss_dict.get('loss_energy', torch.tensor(0)).item():.3f}",
                    "L_cls": f"{loss_dict.get('loss_clash', torch.tensor(0)).item():.3f}",
                    "L_rank": f"{loss_dict.get('loss_pose_rank', torch.tensor(0)).item():.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            del predictions, loss_dict, loss, loss_sum, targets, x_1, x_t, t, batch, source_batch
            del proposal_logits, proposal_logits_cpu, proposal_top_scores, proposal_top_scores_cpu
            del residue_pos_for_crop, residue_batch_for_crop, residue_pos_cpu, residue_batch_cpu
            del crop_centers, crop_centers_cpu, crop_modes
            del wrong_centers, wrong_centers_cpu, wrong_center_scores, wrong_center_scores_cpu
            del wrong_center_valid, wrong_center_valid_cpu, bootstrap_centers, bootstrap_centers_cpu

        # 循环结束，如果有剩余积累的梯度，进行最后一次 step
        if accumulated_graphs > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= accumulated_graphs

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()
                
                if ema_model is None:
                    ema_model = AveragedModel(
                        model,
                        avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                    )
                ema_model.update_parameters(model)
                scheduler.step()
            
            optimizer.zero_grad()

        pbar.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        avg_train_loss = train_loss_meter / max(1, actual_batches)
        if epoch_proposal_losses and epoch_local_losses:
            proposal_mean = float(np.mean(epoch_proposal_losses))
            proposal_std = float(np.std(epoch_proposal_losses))
            local_mean = float(np.mean(epoch_local_losses))
            local_std = float(np.std(epoch_local_losses))
            proposal_scale_ratio = proposal_mean / max(local_mean, 1e-8)
            logger.info(
                "Loss scale stats | proposal=%.4f±%.4f | local=%.4f±%.4f | ratio=%.4f | "
                "proposal_res/graph=%.1f | local_res/graph=%.1f",
                proposal_mean,
                proposal_std,
                local_mean,
                local_std,
                proposal_scale_ratio,
                float(np.mean(epoch_proposal_residues)) if epoch_proposal_residues else 0.0,
                float(np.mean(epoch_local_residues)) if epoch_local_residues else 0.0,
            )
        logger.info(
            "Ranking stats | same_center=%d | wrong_center_low_clash=%d | misleading_center=%d | "
            "misleading_affinity=%d | rank_oom_skips=%d | rank_peak_mem_mb=%.1f",
            epoch_rank_pair_counts["same_center"],
            epoch_rank_pair_counts["wrong_center_low_clash"],
            epoch_rank_pair_counts["misleading_center"],
            epoch_rank_pair_counts["misleading_affinity"],
            epoch_rank_oom_skips,
            epoch_rank_peak_mem_mb,
        )
        if epoch_energy_nan_skips > 0:
            logger.warning(
                "Energy loss skipped due to non-finite affinity values on %d training batches.",
                epoch_energy_nan_skips,
            )

        if epoch_oom_batches > 0:
            logger.warning(
                f"Epoch {epoch+1}: encountered {epoch_oom_batches} CUDA OOM batches "
                f"(total OOM batches={total_oom_batches})"
                + (" [circuit breaker triggered]" if epoch_fused else "")
                + "."
            )

        # 【自适应降批】熔断 epoch 或 OOM 超阈值时降低节点预算
        # 熔断意味着级联发生，必须立即响应
        should_reduce = (
            enable_oom_adaptive_batch
            and (epoch_fused or epoch_oom_batches >= oom_reduce_threshold)
            and current_train_max_nodes_per_batch > effective_min_train_nodes_per_batch
        )
        if should_reduce:
            # 熔断时使用更激进的缩减（0.7x），普通 OOM 使用标准缩减
            factor = min(oom_reduce_factor, 0.7) if epoch_fused else oom_reduce_factor
            reduced_max_nodes = max(
                int(current_train_max_nodes_per_batch * factor),
                int(effective_min_train_nodes_per_batch),
            )

            if reduced_max_nodes < current_train_max_nodes_per_batch:
                logger.warning(
                    f"Epoch {epoch+1}: OOM threshold reached ({epoch_oom_batches}/{oom_reduce_threshold}). "
                    f"Reducing train max_nodes_per_batch: {current_train_max_nodes_per_batch} -> {reduced_max_nodes}."
                )
                current_train_max_nodes_per_batch = reduced_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                logger.info("Rebuilt loaders with tighter node budget; keeping scheduler state continuous.")

                try:
                    total_val_batches = len(val_loader)

                except ValueError:
                    total_val_batches = max(1, len(val_set) // 4)
                rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)

                if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
                    rmsd_check_batches = 1
                logger.info(
                    f"Updated validation sampling: {rmsd_check_batches}/{total_val_batches} batches for RMSD."
                )
                consecutive_clean_epochs = 0  # 降批后重置回升计数

        # 【回升机制】连续无 OOM 时逐步恢复 batch 预算
        if epoch_oom_batches == 0:
            consecutive_clean_epochs += 1

        else:
            consecutive_clean_epochs = 0

        if (
            enable_oom_adaptive_batch
            and consecutive_clean_epochs >= oom_recover_epochs
            and current_train_max_nodes_per_batch < configured_train_max_nodes_per_batch
        ):
            recovered_max_nodes = min(
                int(current_train_max_nodes_per_batch * oom_recover_factor),
                int(configured_train_max_nodes_per_batch),
            )
            if recovered_max_nodes > current_train_max_nodes_per_batch:
                logger.info(
                    f"Epoch {epoch+1}: {consecutive_clean_epochs} consecutive clean epochs. "
                    f"Recovering train max_nodes_per_batch: {current_train_max_nodes_per_batch} -> {recovered_max_nodes}."
                )
                current_train_max_nodes_per_batch = recovered_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                logger.info("Rebuilt loaders with recovered budget; keeping scheduler state continuous.")

                try:
                    total_val_batches = len(val_loader)

                except ValueError:
                    total_val_batches = max(1, len(val_set) // 4)
                rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)

                if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
                    rmsd_check_batches = 1

                consecutive_clean_epochs = 0  # 回升后重置计数

        # 验证
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [新增] 训练结束，验证开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_metrics_raw = compute_validation_loss(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_rmsd_batches=rmsd_check_batches,
            dataset=dataset,
            warmup_epochs=warmup_epochs,
            graph_builder=graph_builder,
            collator=collator,
            crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
            center_proposal_weight=center_proposal_weight,
            center_positive_radius=center_positive_radius,
            # [修复] edge_guard 应基于实际的 val_edge_budget 并预留动态边余量
            # 旧算法用 val_max_nodes * 60 = 360K，但批内实际边数远超该值
            # 现在改为 val_edge_budget * 1.5，为前向传播动态边预留 50% headroom
            edge_guard_limit=max(1, int(
                current_val_max_nodes_per_batch * train_edge_budget_factor * 1.5
            )),
            ode_steps=val_ode_steps,
        )
        local_metrics = dict(local_metrics_raw) if isinstance(local_metrics_raw, dict) else {"val_loss": float(local_metrics_raw)}

        blind_eval = evaluate_topn_success(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            loader=val_loader,
            device=device,
            graph_builder=graph_builder,
            collator=collator,
            topk_values=test_topk_values,
            num_pose_samples=max(test_pose_samples, max(test_topk_values)),
            center_topk=center_proposal_topk,
            refine_topk=center_refine_topk,
            center_nms_radius=center_nms_radius,
            stage1_pose_samples=stage1_pose_samples,
            stage2_pose_samples=stage2_pose_samples,
            crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
            ode_steps=val_ode_steps,
            warmup_epochs=warmup_epochs,
            edge_guard_limit=max(1, int(
                current_val_max_nodes_per_batch * train_edge_budget_factor * 1.5
            )),
            center_hit_radius=center_positive_radius,
            fusion_weights=current_fusion_weights,
            return_candidate_records=True,
        )
        blind_candidate_records = cast(list[dict[str, Any]], blind_eval.get("candidate_records", []))
        if enable_fusion_calibration and blind_candidate_records:
            current_fusion_weights = calibrate_linear_fusion_weights(
                blind_candidate_records,
                topk_values=test_topk_values,
                search_center_weights=fusion_search_center_weights,
                search_aff_weights=fusion_search_aff_weights,
                search_clash_weights=fusion_search_clash_weights,
            )
        blind_metrics = summarize_blind_candidate_records(
            blind_candidate_records,
            topk_values=test_topk_values,
            fusion_weights=current_fusion_weights,
        )
        blind_metrics["topn_edge_guard_skips"] = float(blind_eval.get("topn_edge_guard_skips", 0.0))
        blind_metrics["topn_pose_samples"] = float(blind_eval.get("topn_pose_samples", 0.0))
        blind_metrics["fusion_pose_weight"] = float(current_fusion_weights["pose_weight"])
        blind_metrics["fusion_center_weight"] = float(current_fusion_weights["center_weight"])
        blind_metrics["fusion_aff_weight"] = float(current_fusion_weights.get("aff_weight", 0.0))
        blind_metrics["fusion_clash_weight"] = float(current_fusion_weights.get("clash_weight", 0.0))
        blind_metrics["fusion_bias"] = float(current_fusion_weights["bias"])
        
        # [新增] 验证结束，下一轮开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 提取指标
        avg_val_loss_scalar = float(local_metrics.get("val_loss", float("nan")))
        mean_rmsd = float(blind_metrics.get("reranked_top1_mean_best_rmsd", local_metrics.get("mean_rmsd_final", float("inf"))))
        val_metrics = {f"local_{k}": v for k, v in local_metrics.items()}
        val_metrics.update(blind_metrics)
        val_metrics["val_loss"] = avg_val_loss_scalar
        val_metrics["mean_rmsd_final"] = float(local_metrics.get("mean_rmsd_final", float("inf")))
        val_metrics["local_val_loss"] = avg_val_loss_scalar

        val_oom_batches = int(val_metrics.get("local_oom_batches", 0)) if isinstance(val_metrics, dict) else 0

        if (
            enable_val_oom_adaptive_batch
            and val_oom_batches >= val_oom_reduce_threshold
            and current_val_max_nodes_per_batch > effective_min_val_nodes_per_batch
        ):
            reduced_val_max_nodes = max(
                int(current_val_max_nodes_per_batch * val_oom_reduce_factor),
                int(effective_min_val_nodes_per_batch),
            )
            if reduced_val_max_nodes < current_val_max_nodes_per_batch:
                logger.warning(
                    f"Epoch {epoch+1}: validation OOM threshold reached ({val_oom_batches}/{val_oom_reduce_threshold}). "
                    f"Reducing val max_nodes_per_batch: {current_val_max_nodes_per_batch} -> {reduced_val_max_nodes}."
                )
                current_val_max_nodes_per_batch = reduced_val_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )

        if val_oom_batches == 0:
            consecutive_clean_val_epochs += 1
        else:
            consecutive_clean_val_epochs = 0

        if (
            enable_val_oom_adaptive_batch
            and consecutive_clean_val_epochs >= oom_recover_epochs
            and current_val_max_nodes_per_batch < configured_val_max_nodes_per_batch
        ):
            recovered_val_max_nodes = min(
                int(current_val_max_nodes_per_batch * oom_recover_factor),
                int(configured_val_max_nodes_per_batch),
            )
            if recovered_val_max_nodes > current_val_max_nodes_per_batch:
                logger.info(
                    f"Epoch {epoch+1}: {consecutive_clean_val_epochs} consecutive validation clean epochs. "
                    f"Recovering val max_nodes_per_batch: {current_val_max_nodes_per_batch} -> {recovered_val_max_nodes}."
                )
                current_val_max_nodes_per_batch = recovered_val_max_nodes
                train_loader, val_loader = _build_loaders(
                    current_train_max_nodes_per_batch,
                    current_val_max_nodes_per_batch,
                )
                consecutive_clean_val_epochs = 0

        if not (math.isnan(avg_val_loss_scalar) or math.isinf(avg_val_loss_scalar)):
            best_val_loss = min(best_val_loss, avg_val_loss_scalar)

        # ReduceLROnPlateau 已移除，scheduler 已在 Step 级自动推进

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val-Local Loss: {avg_val_loss_scalar:.4f} | "
            f"Val-Blind Top1 RMSD: {mean_rmsd:.4f} | "
            f"Oracle@1<2A: {val_metrics.get('oracle_top1_success_2a', 0.0):.2f} | "
            f"Rerank@1<2A: {val_metrics.get('reranked_top1_success_2a', 0.0):.2f} | "
            f"CenterRecall@8: {val_metrics.get('center_recall@8', 0.0):.2f} | "
            f"OOM batches: epoch={epoch_oom_batches}, total={total_oom_batches} | "
            f"Edge-guard skips: {epoch_edge_guard_skips} | "
            f"OOM-blacklisted samples: {len(oom_blacklisted_pdb_ids)}"
        )

        selection_metrics = _build_selection_metrics(val_metrics)
        logger.info(
            "Checkpoint selection metrics | "
            f"Composite: {selection_metrics['composite_score']:.4f} | "
            f"BlindCombo: {selection_metrics['blind_combo_score']:.4f} | "
            f"Rerank@1<2A: {selection_metrics['success_2a']:.2f} | "
            f"Rerank@5<2A: {selection_metrics['success_5a']:.2f} | "
            f"Oracle@5<2A: {selection_metrics['oracle_top5_success_2a']:.2f} | "
            f"CenterRecall@8: {selection_metrics['center_recall@8']:.2f} | "
            f"ProposalGap: {selection_metrics['proposal_gap']:.2f} | "
            f"RankingGap: {selection_metrics['ranking_gap']:.2f} | "
            f"Mean RMSD: {selection_metrics['mean_rmsd']:.4f} | "
            f"Val Loss: {selection_metrics['val_loss']:.4f}"
        )
        logger.info(
            "Checkpoint selection mode | mode=%s | primary=%s | value=%.4f",
            checkpoint_selection_mode,
            selected_metric_label,
            selection_metrics[selected_primary_key],
        )

        checkpoint = _compose_checkpoint(
            epoch_idx=epoch,
            avg_train_loss_value=avg_train_loss,
            val_metrics_obj=val_metrics,
            selection_metrics=selection_metrics,
        )

        # 1. 始终保存最新模型（作为保底）
        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pt"))

        # 2. Best model：Curriculum 结束后按多指标分别保存
        is_warmup = epoch < warmup_epochs
        if not is_warmup:
            if _is_better_checkpoint(
                selection_metrics,
                best_selected_metrics,
                primary_key=selected_primary_key,
                primary_higher_is_better=selected_higher_is_better,
            ):
                best_selected_metrics = dict(selection_metrics)
                best_selected_updated_this_epoch = True
                torch.save(checkpoint, os.path.join(save_dir, "best_selected_model.pt"))
                torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                logger.info(
                    "Saved best selected model | mode=%s | %s=%.4f | Rerank@1<2A=%.2f | Oracle@5<2A=%.2f",
                    checkpoint_selection_mode,
                    selected_metric_label,
                    selection_metrics[selected_primary_key],
                    selection_metrics["success_2a"],
                    selection_metrics["oracle_top5_success_2a"],
                )
            if _is_better_checkpoint(
                selection_metrics,
                best_composite_metrics,
                primary_key="composite_score",
                primary_higher_is_better=True,
            ):
                best_composite_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_composite_model.pt"))
                logger.info(
                    "Saved best composite model | "
                    f"Composite={selection_metrics['composite_score']:.4f}, "
                    f"Rerank@1<2A={selection_metrics['success_2a']:.2f}, "
                    f"Rerank@5<2A={selection_metrics['success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if _is_better_checkpoint(
                selection_metrics,
                best_success2a_metrics,
                primary_key="success_2a",
                primary_higher_is_better=True,
            ):
                best_success2a_metrics = dict(selection_metrics)
                torch.save(checkpoint, os.path.join(save_dir, "best_success2a_model.pt"))
                logger.info(
                    "Saved best Success@2A model | "
                    f"Rerank@1<2A={selection_metrics['success_2a']:.2f}, "
                    f"Rerank@5<2A={selection_metrics['success_5a']:.2f}, "
                    f"Mean RMSD={selection_metrics['mean_rmsd']:.4f}."
                )

            if _is_better_checkpoint(
                selection_metrics,
                best_rmsd_metrics,
                primary_key="mean_rmsd",
                primary_higher_is_better=False,
            ):
                best_rmsd_metrics = dict(selection_metrics)
                best_rmsd = selection_metrics["mean_rmsd"]
                checkpoint["best_rmsd"] = best_rmsd
                torch.save(checkpoint, os.path.join(save_dir, "best_rmsd_model.pt"))
                logger.info(f"Saved best Mean RMSD model: {best_rmsd:.4f}")

        # 3. 每 10 轮保存一个永久备份
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

        # 4. 半离线 Blind Candidate Pool 刷新
        if should_refresh_pool(
            epoch,
            refresh_every=blind_pool_refresh_every,
            min_start_epoch=blind_pool_start_epoch,
            best_updated_this_epoch=best_selected_updated_this_epoch,
        ):
            logger.info("Refreshing blind candidate pool at epoch %d ...", epoch + 1)
            pool_model = ema_model if ema_model is not None else model
            pool_loader = _build_eval_loader(train_set, configured_topn_max_nodes_per_batch)
            try:
                new_pool = refresh_blind_candidate_pool(
                    model=pool_model,
                    matcher=matcher,
                    loader=pool_loader,
                    device=device,
                    graph_builder=graph_builder,
                    collator=collator,
                    center_topk=center_proposal_topk,
                    refine_topk=center_refine_topk,
                    center_nms_radius=center_nms_radius,
                    stage1_pose_samples=stage1_pose_samples,
                    stage2_pose_samples=stage2_pose_samples,
                    crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                    ode_steps=val_ode_steps,
                    warmup_epochs=warmup_epochs,
                    center_hit_radius=center_positive_radius,
                    max_complexes=blind_pool_max_complexes,
                    fusion_weights=current_fusion_weights,
                    pool_epoch=epoch,
                    generator_ckpt_id=f"epoch_{epoch}",
                )
                if new_pool:
                    cached_blind_pool = new_pool
                    save_blind_pool(
                        new_pool, blind_pool_cache_dir, epoch=epoch,
                        meta={
                            "compatibility": blind_pool_compatibility,
                            "center_proposal_topk": center_proposal_topk,
                            "center_refine_topk": center_refine_topk,
                            "stage1_pose_samples": stage1_pose_samples,
                            "stage2_pose_samples": stage2_pose_samples,
                            "ode_steps": val_ode_steps,
                            "crop_radius": float(pocket_radius if pocket_radius is not None else 10.0),
                        },
                    )
                    pool_stats = get_pool_stats(cached_blind_pool)
                    logger.info("Blind pool stats: %s", pool_stats)

            except Exception as e:
                logger.warning("Blind pool refresh failed: %s", e)
            finally:
                del pool_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 5. Replay-based reranker training: 当前模型重新打分缓存候选
        if cached_blind_pool and blind_pool_cache_rank_weight > 0:
            try:
                import random as _rng
                replay_dataset = BlindCandidateReplayDataset(
                    cached_blind_pool,
                    candidates_per_complex=max(4, blind_pool_pairs_per_complex * 2),
                )
                if len(replay_dataset) > 0:
                    model.train()
                    replay_sample_size = min(16, len(replay_dataset))
                    replay_indices = _rng.sample(range(len(replay_dataset)), replay_sample_size)
                    replay_items = [replay_dataset[i] for i in replay_indices]

                    replay_losses = replay_and_compute_losses(
                        model=model,
                        replay_items=replay_items,
                        train_set=train_set,
                        graph_builder=graph_builder,
                        collator=collator,
                        device=device,
                        crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                        margin=pose_ranking_margin,
                        lambda_bce=blind_pool_cache_bce_weight,
                        lambda_pair=blind_pool_cache_rank_weight,
                        lambda_list=0.5,
                        lambda_center_value=0.3,
                        use_pose_rank_head=True,
                    )

                    replay_total = replay_losses["rerank_total"]
                    center_val_loss = replay_losses.get("center_value_loss", torch.tensor(0.0, device=device))
                    combined_replay_loss = replay_total + 0.3 * center_val_loss

                    if combined_replay_loss.requires_grad and combined_replay_loss.item() > 0:
                        optimizer.zero_grad()
                        combined_replay_loss.backward()
                        replay_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        if not (torch.isnan(replay_grad_norm) or torch.isinf(replay_grad_norm)):
                            optimizer.step()
                            if ema_model is not None:
                                ema_model.update_parameters(model)
                        optimizer.zero_grad()

                    logger.info(
                        "Replay reranker | bce=%.4f | pairwise=%.4f | listwise=%.4f | "
                        "center_value=%.4f | n_pairs=%d | total=%.4f",
                        replay_losses["rerank_bce"].item(),
                        replay_losses["rerank_pairwise"].item(),
                        replay_losses["rerank_listwise"].item(),
                        center_val_loss.item(),
                        int(replay_losses["rerank_n_pairs"].item()),
                        combined_replay_loss.item(),
                    )
                    model.eval()

            except Exception as e:
                logger.warning("Replay reranker training failed: %s\n%s", e, traceback.format_exc())

        best_selected_updated_this_epoch = False

    # 6. 训练完成后的独立测试集评估（用于最终报告/专利材料）
    if run_test_after_training:
        if len(test_set) == 0:
            logger.warning("Test set is empty; skipping final test evaluation.")
        else:
            preferred_ckpt_paths = [
                os.path.join(save_dir, "best_selected_model.pt"),
                os.path.join(save_dir, "best_composite_model.pt"),
                os.path.join(save_dir, "best_model.pt"),
                os.path.join(save_dir, "best_rmsd_model.pt"),
            ]
            for best_ckpt_path in preferred_ckpt_paths:
                if os.path.exists(best_ckpt_path):
                    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    best_state = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
                    if best_state is not None:
                        model.load_state_dict(best_state)
                        logger.info(f"Loaded best checkpoint for final test evaluation: {best_ckpt_path}")
                        break

            test_loader = _build_eval_loader(test_set, configured_test_max_nodes_per_batch)
            test_metrics_raw = compute_validation_loss(
                model=model,
                matcher=matcher,
                criterion=criterion,
                loader=test_loader,
                device=device,
                epoch=epochs,
                total_epochs=epochs,
                max_rmsd_batches=max(10_000_000, len(test_set)),
                dataset=dataset,
                warmup_epochs=warmup_epochs,
                graph_builder=graph_builder,
                collator=collator,
                crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                center_proposal_weight=center_proposal_weight,
                center_positive_radius=center_positive_radius,
                edge_guard_limit=max(1, int(
                    configured_test_max_nodes_per_batch * train_edge_budget_factor * eval_edge_guard_headroom
                )),
                ode_steps=val_ode_steps,
            )
            if isinstance(test_metrics_raw, dict):
                test_metrics = {f"local_{k}": v for k, v in dict(test_metrics_raw).items()}
                test_metrics["val_loss"] = float(test_metrics_raw.get("val_loss", float("nan")))
            else:
                test_metrics = {"val_loss": float(test_metrics_raw)}

            topn_loader = _build_eval_loader(test_set, configured_topn_max_nodes_per_batch)

            topn_eval = evaluate_topn_success(
                model=model,
                matcher=matcher,
                loader=topn_loader,
                device=device,
                graph_builder=graph_builder,
                collator=collator,
                topk_values=test_topk_values,
                num_pose_samples=max(test_pose_samples, max(test_topk_values)),
                center_topk=center_proposal_topk,
                refine_topk=center_refine_topk,
                center_nms_radius=center_nms_radius,
                stage1_pose_samples=stage1_pose_samples,
                stage2_pose_samples=stage2_pose_samples,
                crop_radius=float(pocket_radius if pocket_radius is not None else 10.0),
                ode_steps=val_ode_steps,
                warmup_epochs=warmup_epochs,
                edge_guard_limit=max(1, int(
                    configured_topn_max_nodes_per_batch * train_edge_budget_factor * eval_edge_guard_headroom
                )),
                center_hit_radius=center_positive_radius,
                fusion_weights=current_fusion_weights,
                return_candidate_records=True,
            )
            test_candidate_records = cast(list[dict[str, Any]], topn_eval.get("candidate_records", []))
            topn_metrics = summarize_blind_candidate_records(
                test_candidate_records,
                topk_values=test_topk_values,
                fusion_weights=current_fusion_weights,
            )
            topn_metrics["fusion_pose_weight"] = float(current_fusion_weights["pose_weight"])
            topn_metrics["fusion_center_weight"] = float(current_fusion_weights["center_weight"])
            topn_metrics["fusion_aff_weight"] = float(current_fusion_weights.get("aff_weight", 0.0))
            topn_metrics["fusion_clash_weight"] = float(current_fusion_weights.get("clash_weight", 0.0))
            topn_metrics["fusion_bias"] = float(current_fusion_weights["bias"])
            topn_metrics["topn_edge_guard_skips"] = float(topn_eval.get("topn_edge_guard_skips", 0.0))
            topn_metrics["topn_pose_samples"] = float(topn_eval.get("topn_pose_samples", 0.0))
            test_metrics.update(topn_metrics)

            report_dir = os.path.join(save_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "test_metrics.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved final test report to {report_path}")
            logger.info(f"[Test Summary] {test_metrics}")


@torch.no_grad()
def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int = 1,
    max_rmsd_batches: int = 10,
    dataset: PDBBindDataset | None = None,
    warmup_epochs: int = 20,
    graph_builder: Any | None = None,
    collator: GraphCollator | None = None,
    crop_radius: float = 10.0,
    center_proposal_weight: float = 0.15,
    center_positive_radius: float = 4.0,
    edge_guard_limit: int | None = None,
    ode_steps: int = 50,
) -> dict | float:
    """
    验证函数：计算 Loss 并统计全量 RMSD 指标
    """
    model.eval()
    total_loss = 0.0
    all_rmsd_init: list[torch.Tensor] = []
    all_rmsd_final: list[torch.Tensor] = []
    all_centroid_dist: list[torch.Tensor] = []   # 质心距离
    affinity_preds: list[torch.Tensor] = []
    affinity_targets: list[torch.Tensor] = []
    
    valid_batches = 0
    oom_batches = 0
    edge_guard_skips = 0
    
    # 固定随机种子 (保持验证集生成的一致性)
    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

    # 使用 tqdm 显示验证进度，按精确样本数统计
    val_total_graphs = len(cast(Any, loader).dataset) if hasattr(loader, "dataset") else 0
    pbar = tqdm(total=val_total_graphs, desc=f"Epoch {(epoch or 0) + 1} [Val]", leave=False, unit="graphs")

    if graph_builder is None or collator is None:
        raise ValueError("graph_builder and collator are required for runtime local cropping.")

    for i, batch in enumerate(loader):
        num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
        pbar.update(num_graphs)

        if edge_guard_limit is not None:
            total_edges_cpu = 0
            edge_types = getattr(batch, "edge_types", None)
            if edge_types:
                for edge_type in edge_types:
                    edge_store = batch[edge_type]
                    edge_index = getattr(edge_store, "edge_index", None)
                    if edge_index is not None and edge_index.ndim == 2:
                        total_edges_cpu += int(edge_index.size(1))

            if total_edges_cpu > edge_guard_limit:
                edge_guard_skips += 1
                logger.warning(
                    f"Validation batch {i}: preflight skip due to edge-heavy batch "
                    f"(total_edges={total_edges_cpu} > limit={edge_guard_limit})."
                )
                continue

        try:
            ligand_centers = scatter_mean(
                batch["ligand_atom"].pos,
                batch["ligand_atom"].batch,
                dim=0,
                dim_size=num_graphs,
            )
            proposal_loss = compute_proposal_loss(
                model,
                batch,
                device=device,
                positive_radius=center_positive_radius,
            )
            local_batch = build_local_batch_from_centers(
                batch,
                centers=ligand_centers,
                crop_radius=crop_radius,
                graph_builder=graph_builder,
                collator=collator,
            )
            batch = local_batch.to(device)
            crop_centers = ligand_centers.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
            apply_loss_context(
                batch,
                current_epoch=epoch if epoch is not None else total_epochs - 1,
                total_epochs_count=total_epochs,
                warmup_epochs_count=warmup_epochs,
                training=False,
            )
            x_1 = batch["ligand_atom"].pos

            # 1. 计算 Loss (用于早停和模型选择)
            t, x_t, targets = matcher.sample_location_and_target(
                x_1=x_1,
                data=batch,
                current_epoch=epoch if epoch is not None else 0,
                total_epochs=total_epochs,
                placement_centers=crop_centers,
            )

            batch["ligand_atom"].pos = x_t
            batch.t = t  # 注入时间步，供 Loss 时间掩码使用

            predictions = model(batch, t)

            # matcher 已返回分解好的 SE(3) 目标，直接补全结合能
            targets["binding_affinity_target"] = batch.get("y_energy", None)
            targets["pose_quality_target"] = compute_pose_quality_target(x_t, x_1, batch["ligand_atom"].batch)

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"] + center_proposal_weight * proposal_loss
            
            # 过滤爆炸 Loss
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                total_loss += loss.item()
                valid_batches += 1
                
            # [新增] 收集亲和力预测 (用于计算 RMSE/Pearson/Spearman)
            # 与 losses.py 保持一致：仅在 t > 0.8 时收集，避免噪声位姿下的能量预测污染统计
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                # 遵循 losses.py 中的物理约束阈值
                if t is not None:
                    valid_mask = t > 0.8
                else:
                    valid_mask = torch.ones_like(batch.get("y_energy", torch.zeros(1)), dtype=torch.bool)
                
                if valid_mask.any():
                    pred_aff = predictions.get("binding_affinity", None)
                    if pred_aff is not None:
                        # 仅选取 t > 0.5 的预测值
                        pred_aff_valid = pred_aff[valid_mask]
                        # 双重检查：预测值本身也不能含 NaN
                        if not torch.isnan(pred_aff_valid).any():
                            affinity_preds.append(pred_aff_valid.cpu())
                            # target 统一为 raw（若已提供 raw 则直接用，否则做一次反归一化）
                            if hasattr(batch, "y_energy_raw"):
                                target_raw_valid = batch.y_energy_raw[valid_mask]
                                affinity_targets.append(target_raw_valid.cpu())
                            else:
                                y_norm = batch.get("y_energy", None)
                                if y_norm is not None and dataset is not None:
                                    target_raw_valid = dataset.denormalize_affinity(y_norm[valid_mask].cpu())
                                    affinity_targets.append(target_raw_valid)
            
            # 2. 全量 RMSD 推演
            # -----------------------------------------------------------
            if i < max_rmsd_batches:
                try:
                    # 克隆数据用于推演
                    infer_batch = batch.clone()
                    infer_batch["ligand_atom"].pos = x_1 
                    
                    # 验证始终使用全难度扰动（与最终推理条件一致）
                    x_0_infer = matcher._generate_random_pose(
                        x_ref=x_1,
                        batch=infer_batch["ligand_atom"].batch,
                        B=int(infer_batch["ligand_atom"].batch.max().item()) + 1,
                        masses=infer_batch["ligand_atom"].masses,
                        torsion_indices=getattr(infer_batch, "torsion_indices", None),
                        torsion_moving_mask=getattr(infer_batch, "torsion_moving_mask", None),
                        seed_pos=infer_batch["ligand_atom"].get("start_pos", None),
                        protein_pos=infer_batch["protein_atom"].pos,
                        protein_batch=getattr(infer_batch["protein_atom"], "batch", None),
                        placement_centers=crop_centers,
                        epoch=warmup_epochs,
                    )
                    
                    # 记录初始 RMSD
                    sq_diff_init = ((x_0_infer - x_1) ** 2).sum(dim=-1)
                    msd_init = scatter_mean(sq_diff_init, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_init = torch.sqrt(msd_init)
                    # [修改] 强制转 CPU，切断 GPU 显存占用
                    all_rmsd_init.append(rmsd_init.detach().cpu())

                    infer_batch["ligand_atom"].pos = x_0_infer
                    final_pos, _ = matcher.ode_solve(
                        model=model,
                        data=infer_batch,
                        steps=ode_steps,
                        method="euler",
                        store_trajectory=False,
                    )
                    
                    # 记录最终 RMSD
                    sq_diff_final = ((final_pos - x_1) ** 2).sum(dim=-1)
                    msd_final = scatter_mean(sq_diff_final, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_final = torch.sqrt(msd_final)
                    all_rmsd_final.append(rmsd_final.detach().cpu())

                    # 记录质心距离（Centroid Distance）
                    B_infer = int(infer_batch["ligand_atom"].batch.max().item()) + 1
                    pred_centroid = scatter_mean(final_pos, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    true_centroid = scatter_mean(x_1, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    centroid_dist = torch.norm(pred_centroid - true_centroid, dim=-1)  # [B_infer]
                    all_centroid_dist.append(centroid_dist.detach().cpu())

                    del infer_batch, x_0_infer, final_pos, sq_diff_init, msd_init, rmsd_init
                    del sq_diff_final, msd_final, rmsd_final, pred_centroid, true_centroid, centroid_dist
                    
                except Exception as e:
                    logger.warning(f"RMSD inference failed for batch {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            # -----------------------------------------------------------

        except torch.cuda.OutOfMemoryError:
            oom_batches += 1
            logger.warning(f"Validation batch {i}: CUDA OOM, skipping and clearing cache.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        except Exception as e:
            logger.warning(f"Validation batch failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        del predictions, targets, loss_dict, loss, x_1, x_t, t, batch, proposal_loss, ligand_centers, local_batch, crop_centers

    # ========== 综合评估指标计算 ==========
    metrics: dict[str, float] = {}

    # --- 亲和力预测指标 ---
    pearson_r = 0.0
    spearman_rho = 0.0
    rmse_val = float("inf")
    mae_val = float("inf")

    if len(affinity_preds) > 0 and dataset is not None:
        cat_preds = torch.cat(affinity_preds).view(-1)
        cat_targets = torch.cat(affinity_targets).view(-1)
        
        # 模型输出是 norm，验证时仅在这里做一次反归一化
        raw_preds: torch.Tensor = dataset.denormalize_affinity(cat_preds)
        
        mse_val = F.mse_loss(raw_preds, cat_targets)
        rmse_val = torch.sqrt(mse_val).item()
        mae_val = F.l1_loss(raw_preds, cat_targets).item()

        # Pearson R（线性相关性，对应 CASF-2016 Scoring Power）
        # Spearman ρ（排序一致性，对应 CASF-2016 Ranking Power）
        pred_np = raw_preds.numpy()
        target_np = cat_targets.numpy()
        if len(pred_np) > 2 and np.std(pred_np) > 1e-6:
            pearson_res = scipy_stats.pearsonr(pred_np, target_np)
            spearman_res = scipy_stats.spearmanr(pred_np, target_np)
            pearson_r = float(cast(Any, pearson_res)[0])
            spearman_rho = float(cast(Any, spearman_res)[0])
        
        logger.info(f"[Validation Affinity] RMSE: {rmse_val:.4f} pKd | MAE: {mae_val:.4f} pKd")
        logger.info(f"  Pearson R: {pearson_r:.4f} | Spearman ρ: {spearman_rho:.4f}")

    metrics["affinity_rmse"] = rmse_val
    metrics["affinity_mae"] = mae_val
    metrics["pearson_r"] = pearson_r
    metrics["spearman_rho"] = spearman_rho

    # --- 对接精度指标 ---
    mean_final = float("inf")
    median_final = float("inf")
    success_2a = 0.0
    success_5a = 0.0
    mean_centroid = float("inf")
    median_centroid = float("inf")

    if len(all_rmsd_final) > 0:
        cat_rmsd_init = torch.cat(all_rmsd_init)
        cat_rmsd_final = torch.cat(all_rmsd_final)
        
        mean_init = cat_rmsd_init.mean().item()
        mean_final = cat_rmsd_final.mean().item()
        median_final = cat_rmsd_final.median().item()
        
        # 成功率（多阈值）
        success_2a = (cat_rmsd_final < 2.0).float().mean().item() * 100
        success_5a = (cat_rmsd_final < 5.0).float().mean().item() * 100

        # 质心距离
        if len(all_centroid_dist) > 0:
            cat_centroid = torch.cat(all_centroid_dist)
            mean_centroid = cat_centroid.mean().item()
            median_centroid = cat_centroid.median().item()
        
        logger.info("-" * 60)
        logger.info(f"[Validation Full Stats] Epoch {(epoch or 0) + 1}")
        logger.info(f"  Mean RMSD: {mean_init:.2f} -> {mean_final:.2f} Å | Median: {median_final:.2f} Å")
        logger.info(f"  Success Rate (<2Å): {success_2a:.2f}% | (<5Å): {success_5a:.2f}%")
        logger.info(f"  Centroid Distance: Mean {mean_centroid:.2f} Å | Median {median_centroid:.2f} Å")
        logger.info("-" * 60)

    metrics["mean_rmsd_final"] = mean_final
    metrics["median_rmsd_final"] = median_final
    metrics["success_2a"] = success_2a
    metrics["success_5a"] = success_5a
    metrics["single_shot_success_2a"] = success_2a
    metrics["single_shot_success_5a"] = success_5a
    metrics["centroid_dist_mean"] = mean_centroid
    metrics["centroid_dist_median"] = median_centroid
    metrics["oom_batches"] = float(oom_batches)
    metrics["valid_batches"] = float(valid_batches)
    metrics["edge_guard_skips"] = float(edge_guard_skips)

    if valid_batches == 0:
        metrics["val_loss"] = float("nan")
        return metrics

    # 显式清理现场
    del all_rmsd_init, all_rmsd_final, all_centroid_dist
    del affinity_preds, affinity_targets
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics["val_loss"] = total_loss / valid_batches
    return metrics


@torch.no_grad()
def evaluate_topn_success(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    loader: DataLoader,
    device: torch.device,
    graph_builder: Any,
    collator: GraphCollator,
    topk_values: tuple[int, ...] = (1, 5, 10),
    num_pose_samples: int = 10,
    center_topk: int = 8,
    refine_topk: int = 3,
    center_nms_radius: float = 6.0,
    stage1_pose_samples: int = 2,
    stage2_pose_samples: int = 4,
    crop_radius: float = 10.0,
    ode_steps: int = 50,
    warmup_epochs: int = 20,
    edge_guard_limit: int | None = None,
    center_hit_radius: float = 4.0,
    fusion_weights: dict[str, float] | None = None,
    return_candidate_records: bool = False,
) -> dict[str, Any]:
    """基于统一候选生成引擎的 Top-N 对接成功率评估。"""

    topk_unique = tuple(sorted({int(k) for k in topk_values if int(k) > 0}))
    if not topk_unique:
        raise ValueError("topk_values must contain at least one positive integer")

    candidate_records = generate_candidates_from_loader(
        model=model,
        matcher=matcher,
        loader=loader,
        device=device,
        graph_builder=graph_builder,
        collator=collator,
        center_topk=center_topk,
        refine_topk=refine_topk,
        center_nms_radius=center_nms_radius,
        stage1_pose_samples=stage1_pose_samples,
        stage2_pose_samples=stage2_pose_samples,
        crop_radius=crop_radius,
        ode_steps=ode_steps,
        warmup_epochs=warmup_epochs,
        center_hit_radius=center_hit_radius,
        fusion_weights=fusion_weights,
        edge_guard_limit=edge_guard_limit,
    )

    total_graphs = len(candidate_records)
    total_pose_budget = center_topk * stage1_pose_samples + refine_topk * stage2_pose_samples

    if total_graphs == 0:
        result: dict[str, Any] = {"topn_total_graphs": 0.0}
        if return_candidate_records:
            result["candidate_records"] = []
        return result

    metrics: dict[str, Any] = {
        "topn_total_graphs": float(total_graphs),
        "topn_pose_samples": float(total_pose_budget),
        "topn_edge_guard_skips": 0.0,
    }
    metrics.update(
        summarize_blind_candidate_records(
            candidate_records,
            topk_values=topk_unique,
            fusion_weights=fusion_weights,
        )
    )
    if return_candidate_records:
        metrics["candidate_records"] = candidate_records
    return metrics
