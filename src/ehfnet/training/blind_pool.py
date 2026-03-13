"""
Blind Candidate Pool — 可回放候选池系统。

核心设计原则：
- 缓存能被当前模型重放并重新打分的候选数据（pose_xyz + dataset_index + center_xyz）
- 不依赖 teacher logit 作为训练主输入
- 验证 / 测试 / pool refresh / 离线候选池 全部调用 candidate_generation 单一真源
"""

import os
import json
import logging
import random
import time
import gc
from typing import Any, cast
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_mean

from ehfnet.graph import GraphCollator, crop_graph_to_center
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from ehfnet.training.rerank_losses import (
    compute_rerank_losses,
    compute_center_value_loss,
    rmsd_to_soft_target,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 候选池生成（委托给 candidate_generation 单一真源）
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def refresh_blind_candidate_pool(
    *,
    model: torch.nn.Module,
    matcher: Any,
    loader: DataLoader,
    device: torch.device,
    graph_builder: Any,
    collator: GraphCollator,
    center_topk: int = 8,
    refine_topk: int = 3,
    center_nms_radius: float = 6.0,
    stage1_pose_samples: int = 2,
    stage2_pose_samples: int = 4,
    crop_radius: float = 10.0,
    ode_steps: int = 50,
    warmup_epochs: int = 20,
    center_hit_radius: float = 4.0,
    max_complexes: int | None = None,
    fusion_weights: dict[str, float] | None = None,
    pool_epoch: int = -1,
    generator_ckpt_id: str = "",
) -> list[dict[str, Any]]:
    """Run full blind pipeline via unified candidate_generation engine."""
    return generate_candidates_from_loader(
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
        max_complexes=max_complexes,
        fusion_weights=fusion_weights,
        pool_epoch=pool_epoch,
        generator_ckpt_id=generator_ckpt_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. 存储 / 加载
# ──────────────────────────────────────────────────────────────────────────────

def save_blind_pool(
    records: list[dict[str, Any]],
    cache_dir: str,
    *,
    epoch: int,
    meta: dict[str, Any] | None = None,
) -> str:
    """Save candidate pool to disk with manifest."""
    epoch_dir = os.path.join(cache_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    pool_path = os.path.join(epoch_dir, "pool.pt")
    torch.save(records, pool_path)

    total_poses = sum(len(r.get("poses", [])) for r in records)
    hit_2a = sum(1 for r in records for p in r.get("poses", []) if p.get("is_hit_2A", False))
    total_centers = sum(len(r.get("centers", [])) for r in records)
    center_hits = sum(1 for r in records for c in r.get("centers", []) if c.get("is_center_hit_4A", False))

    oracle_rmsds = []
    for r in records:
        poses = r.get("poses", [])
        if poses:
            oracle_rmsds.append(min(p["rmsd"] for p in poses))

    manifest = {
        "epoch": epoch,
        "n_complexes": len(records),
        "n_total_poses": total_poses,
        "n_total_centers": total_centers,
        "n_hit_2a": hit_2a,
        "hit_rate_2a": hit_2a / max(1, total_poses),
        "n_center_hits_4a": center_hits,
        "center_hit_rate": center_hits / max(1, total_centers),
        "oracle_mean_rmsd": float(sum(oracle_rmsds) / max(1, len(oracle_rmsds))) if oracle_rmsds else 999.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool_path": pool_path,
    }
    if meta:
        manifest.update(meta)

    manifest_path = os.path.join(epoch_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved blind pool | epoch=%d | complexes=%d | poses=%d | hit_2a=%.2f%% | center_hit=%.2f%% | oracle_rmsd=%.3f",
        epoch, len(records), total_poses,
        manifest["hit_rate_2a"] * 100, manifest["center_hit_rate"] * 100,
        manifest["oracle_mean_rmsd"],
    )
    return pool_path


def load_blind_pool(cache_dir: str, epoch: int | None = None) -> list[dict[str, Any]]:
    """Load most recent (or specified) pool from cache."""
    if not os.path.isdir(cache_dir):
        return []

    if epoch is not None:
        pool_path = os.path.join(cache_dir, f"epoch_{epoch:04d}", "pool.pt")
        if os.path.exists(pool_path):
            return torch.load(pool_path, weights_only=False)

    epoch_dirs = sorted(Path(cache_dir).glob("epoch_*"), reverse=True)
    for d in epoch_dirs:
        pool_path = d / "pool.pt"
        if pool_path.exists():
            return torch.load(str(pool_path), weights_only=False)

    return []


# ──────────────────────────────────────────────────────────────────────────────
# 3. 可回放候选数据集 (BlindCandidateReplayDataset)
# ──────────────────────────────────────────────────────────────────────────────

class BlindCandidateReplayDataset(Dataset):
    """从 blind pool 中按 complex 采样可回放候选组。

    每个 item 返回一个 complex 的多个候选，每个候选包含：
    - dataset_index: 用于从 train_set 取 full-protein sample
    - center_xyz: 用于 crop_graph_to_center
    - pose_xyz: 用于覆盖 ligand_atom.pos
    - rmsd, soft_target: 用于计算损失
    - center_value_target: center-level supervision

    不返回 teacher logit 作为训练主输入。
    """

    def __init__(
        self,
        pool: list[dict[str, Any]],
        *,
        candidates_per_complex: int = 8,
        positive_rmsd_threshold: float = 2.0,
        hard_negative_clash_threshold: float = 5.0,
    ):
        self.pool = [r for r in pool if len(r.get("poses", [])) >= 2]
        self.candidates_per_complex = candidates_per_complex
        self.pos_thresh = positive_rmsd_threshold
        self.clash_thresh = hard_negative_clash_threshold

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.pool[idx]
        poses = record.get("poses", [])
        centers = record.get("centers", [])
        dataset_index = record.get("dataset_index", idx)

        # 分层采样候选
        positives = [p for p in poses if p["rmsd"] < self.pos_thresh]
        pos_center_ids = {p["center_id"] for p in positives}
        same_center_negs = [
            p for p in poses
            if p["rmsd"] >= self.pos_thresh and p["center_id"] in pos_center_ids
        ]
        wrong_center_negs = [
            p for p in poses
            if p["rmsd"] >= self.pos_thresh
            and p["center_id"] not in pos_center_ids
            and p.get("steric_clash_teacher", 0) <= self.clash_thresh
        ]
        deceptive_negs = [
            p for p in poses
            if p["rmsd"] >= self.pos_thresh
            and p.get("steric_clash_teacher", 0) <= self.clash_thresh
        ]

        selected: list[dict[str, Any]] = []

        # 按优先级填充：先保证有 positives 和各类 negatives
        n = self.candidates_per_complex
        if positives:
            k = min(max(1, n // 3), len(positives))
            selected.extend(random.sample(positives, k))

        remaining = n - len(selected)
        if remaining > 0:
            for pool_list in [same_center_negs, wrong_center_negs, deceptive_negs]:
                if pool_list and remaining > 0:
                    k = min(max(1, remaining // 2), len(pool_list))
                    selected.extend(random.sample(pool_list, k))
                    remaining = n - len(selected)

        # 随机补齐
        if len(selected) < n:
            others = [p for p in poses if p not in selected]
            if others:
                k = min(n - len(selected), len(others))
                selected.extend(random.sample(others, k))

        # 构造回放数据
        candidates = []
        for p in selected:
            candidate_role = "random"
            if p in positives:
                candidate_role = "positive"
            elif p in same_center_negs:
                candidate_role = "same_center_negative"
            elif p in wrong_center_negs:
                candidate_role = "wrong_center_negative"
            elif p in deceptive_negs:
                candidate_role = "deceptive_negative"
            candidates.append({
                "dataset_index": dataset_index,
                "center_xyz": self._get_center_xyz(p, centers),
                "pose_xyz": p["pose_xyz"],
                "rmsd": p["rmsd"],
                "soft_target": p.get("soft_target", rmsd_to_soft_target(torch.tensor(p["rmsd"])).item()),
                "center_id": p["center_id"],
                "stage_id": p.get("stage_id", "unknown"),
                "rank_bucket": p.get("rank_bucket", "bad"),
                "is_hit_2A": p.get("is_hit_2A", False),
                "candidate_role": candidate_role,
                "binding_affinity_teacher": p.get("binding_affinity_teacher", 0.0),
                "steric_clash_teacher": p.get("steric_clash_teacher", 0.0),
            })

        # center-value targets
        center_values = {}
        for c in centers:
            label = c.get("center_success_label", "negative")
            if label == "strong_positive":
                center_values[c["center_id"]] = 1.0
            elif label == "weak_positive":
                center_values[c["center_id"]] = 0.5
            else:
                center_values[c["center_id"]] = 0.0

        return {
            "complex_id": record.get("complex_id", ""),
            "dataset_index": dataset_index,
            "gt_center_xyz": record.get("gt_center_xyz", [0.0, 0.0, 0.0]),
            "candidates": candidates,
            "centers": centers,
            "center_values": center_values,
            "n_ligand_atoms": record.get("n_ligand_atoms", 0),
        }

    def _get_center_xyz(self, pose: dict, centers: list[dict]) -> list[float]:
        cid = pose["center_id"]
        for c in centers:
            if c["center_id"] == cid:
                return c["center_xyz"]
        return [0.0, 0.0, 0.0]


# ──────────────────────────────────────────────────────────────────────────────
# 4. 回放前向 + 损失计算
# ──────────────────────────────────────────────────────────────────────────────

def replay_and_compute_losses(
    *,
    model: torch.nn.Module,
    replay_items: list[dict[str, Any]],
    train_set: Any,
    graph_builder: Any,
    collator: GraphCollator,
    device: torch.device,
    crop_radius: float = 10.0,
    margin: float = 0.5,
    lambda_bce: float = 1.0,
    lambda_pair: float = 1.0,
    lambda_list: float = 0.5,
    lambda_center_value: float = 0.3,
    use_pose_rank_head: bool = True,
) -> dict[str, Tensor]:
    """Replay cached candidates through the current model and compute rerank losses.

    This is the core function that makes blind pool training *actually*
    train the current model. Instead of operating on cached teacher logits,
    it:
    1. Loads full-protein sample by dataset_index
    2. Crops to center_xyz
    3. Overrides ligand_atom.pos with pose_xyz
    4. Forwards through current model
    5. Computes BCE + pairwise + listwise losses against RMSD targets

    Args:
        replay_items: list of items from BlindCandidateReplayDataset
        train_set: the original training dataset for looking up samples by index
    """
    per_group_totals: list[Tensor] = []
    per_group_bce: list[Tensor] = []
    per_group_pair: list[Tensor] = []
    per_group_list: list[Tensor] = []
    total_pairs = 0
    all_center_logits: list[Tensor] = []
    all_center_targets: list[Tensor] = []

    for item in replay_items:
        candidates = item["candidates"]
        center_values = item.get("center_values", {})
        if not candidates:
            continue

        ds_idx = int(item["dataset_index"])

        try:
            sample = _resolve_replay_sample(train_set, ds_idx)
        except Exception:
            continue

        group_logits: list[Tensor] = []
        group_rmsd: list[Tensor] = []

        for cand in candidates:
            try:
                center_xyz = torch.tensor(cand["center_xyz"], dtype=torch.float32)
                pose_xyz = torch.tensor(cand["pose_xyz"], dtype=torch.float32)

                local_sample = crop_graph_to_center(
                    sample, center=center_xyz, radius=crop_radius,
                    graph_builder=graph_builder,
                )

                infer_batch = cast(Any, collator.collate([local_sample])).to(device)

                n_lig = infer_batch["ligand_atom"].pos.size(0)
                if pose_xyz.size(0) != n_lig:
                    continue

                infer_batch["ligand_atom"].pos = pose_xyz.to(device=device, dtype=infer_batch["ligand_atom"].pos.dtype)
                t_ones = torch.ones(1, device=device, dtype=infer_batch["ligand_atom"].pos.dtype)

                out = model(infer_batch, t_ones)

                if use_pose_rank_head and "pose_rank_score" in out:
                    logit = out["pose_rank_score"].view(-1)
                else:
                    logit = out["pose_quality"].view(-1)

                group_logits.append(logit)
                group_rmsd.append(torch.tensor([cand["rmsd"]], device=device, dtype=torch.float32))

                del infer_batch, out

            except torch.cuda.OutOfMemoryError:
                logger.warning("Replay OOM on candidate, skipping.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception:
                continue

        if group_logits:
            logits_cat = torch.cat(group_logits)
            rmsd_cat = torch.cat(group_rmsd)
            pair_indices = _build_group_pair_indices(candidates, device=device)
            rerank_results = compute_rerank_losses(
                logits_cat,
                rmsd_cat,
                margin=margin,
                pair_indices=pair_indices,
                lambda_bce=lambda_bce,
                lambda_pair=lambda_pair,
                lambda_list=lambda_list,
            )
            per_group_totals.append(rerank_results["rerank_total"])
            per_group_bce.append(rerank_results["rerank_bce"])
            per_group_pair.append(rerank_results["rerank_pairwise"])
            per_group_list.append(rerank_results["rerank_listwise"])
            total_pairs += int(rerank_results["rerank_n_pairs"].item())

        # center-value: run proposal on the full-protein sample once per complex
        if lambda_center_value > 0 and center_values:
            try:
                sample_batch = cast(Any, collator.collate([sample])).to(device)

                from ehfnet.training.trainer import predict_center_proposal_logits
                prop_logits, res_pos, res_batch, _ = predict_center_proposal_logits(
                    model, sample_batch, device=device,
                )

                for cid, cv_target in center_values.items():
                    center_rec = None
                    for c in item.get("centers", []):
                        if c.get("center_id") == cid:
                            center_rec = c
                            break
                    if center_rec is None:
                        continue

                    center_xyz_t = torch.tensor(center_rec["center_xyz"], device=device, dtype=res_pos.dtype)
                    dists = torch.norm(res_pos - center_xyz_t.unsqueeze(0), dim=-1)
                    nearest_idx = dists.argmin()
                    all_center_logits.append(prop_logits[nearest_idx].view(-1))
                    all_center_targets.append(torch.tensor([cv_target], device=device, dtype=torch.float32))

                del sample_batch
            except Exception:
                    pass

    result: dict[str, Tensor] = {}

    if per_group_totals:
        result["rerank_total"] = torch.stack(per_group_totals).mean()
        result["rerank_bce"] = torch.stack(per_group_bce).mean()
        result["rerank_pairwise"] = torch.stack(per_group_pair).mean()
        result["rerank_listwise"] = torch.stack(per_group_list).mean()
        result["rerank_n_pairs"] = torch.tensor(float(total_pairs), device=device)
    else:
        result["rerank_total"] = torch.tensor(0.0, device=device)
        result["rerank_bce"] = torch.tensor(0.0, device=device)
        result["rerank_pairwise"] = torch.tensor(0.0, device=device)
        result["rerank_listwise"] = torch.tensor(0.0, device=device)
        result["rerank_n_pairs"] = torch.tensor(0.0, device=device)

    if all_center_logits and lambda_center_value > 0:
        center_logits_cat = torch.cat(all_center_logits)
        center_targets_cat = torch.cat(all_center_targets)
        result["center_value_loss"] = compute_center_value_loss(center_logits_cat, center_targets_cat)
    else:
        result["center_value_loss"] = torch.tensor(0.0, device=device)

    return result


def _resolve_replay_sample(train_subset_or_dataset: Any, dataset_index: int) -> Any:
    """Resolve replay samples against the base dataset.

    blind pool stores indices from the underlying dataset; when training uses
    torch.utils.data.Subset wrappers, replay must bypass subset-local indexing.
    """
    if hasattr(train_subset_or_dataset, "dataset") and hasattr(train_subset_or_dataset, "indices"):
        return train_subset_or_dataset.dataset[int(dataset_index)]
    return train_subset_or_dataset[int(dataset_index)]


def _build_group_pair_indices(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device,
    positive_rmsd_threshold: float = 2.0,
    min_rmsd_gap: float = 0.25,
) -> Tensor | None:
    if len(candidates) < 2:
        return None

    rmsd_values = [float(c["rmsd"]) for c in candidates]
    positive_indices = [i for i, c in enumerate(candidates) if float(c["rmsd"]) < positive_rmsd_threshold]
    pair_list: list[tuple[int, int]] = []

    if positive_indices:
        positive_center_ids = {int(candidates[i]["center_id"]) for i in positive_indices}
        for pos_idx in positive_indices:
            pos_center = int(candidates[pos_idx]["center_id"])
            for neg_idx, cand in enumerate(candidates):
                if neg_idx == pos_idx:
                    continue
                neg_rmsd = float(cand["rmsd"])
                if neg_rmsd <= rmsd_values[pos_idx] + min_rmsd_gap:
                    continue
                neg_center = int(cand["center_id"])
                role = str(cand.get("candidate_role", "random"))
                if neg_center == pos_center or neg_center not in positive_center_ids or role.endswith("negative"):
                    pair_list.append((pos_idx, neg_idx))

    if not pair_list:
        ordered = sorted(range(len(candidates)), key=lambda i: rmsd_values[i])
        for better, worse in zip(ordered[:-1], ordered[1:]):
            if rmsd_values[worse] - rmsd_values[better] >= min_rmsd_gap:
                pair_list.append((better, worse))

    if not pair_list:
        return None

    return torch.tensor(pair_list, device=device, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 调度 + 统计
# ──────────────────────────────────────────────────────────────────────────────

def should_refresh_pool(
    epoch: int,
    *,
    refresh_every: int = 5,
    min_start_epoch: int = 10,
    best_updated_this_epoch: bool = False,
) -> bool:
    if epoch < min_start_epoch:
        return False
    if best_updated_this_epoch:
        return True
    return epoch % refresh_every == 0


def get_pool_stats(pool: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize pool for logging."""
    if not pool:
        return {"pool_complexes": 0.0}

    total_poses = sum(len(r.get("poses", [])) for r in pool)
    hit_2a = sum(1 for r in pool for p in r.get("poses", []) if p.get("is_hit_2A", False))
    hit_5a = sum(1 for r in pool for p in r.get("poses", []) if p.get("is_hit_5A", False))
    total_centers = sum(len(r.get("centers", [])) for r in pool)
    center_hits = sum(1 for r in pool for c in r.get("centers", []) if c.get("is_center_hit_4A", False))

    oracle_rmsds = []
    for r in pool:
        poses = r.get("poses", [])
        if poses:
            oracle_rmsds.append(min(p["rmsd"] for p in poses))

    strong_centers = sum(
        1 for r in pool for c in r.get("centers", [])
        if c.get("center_success_label") == "strong_positive"
    )
    weak_centers = sum(
        1 for r in pool for c in r.get("centers", [])
        if c.get("center_success_label") == "weak_positive"
    )

    return {
        "pool_complexes": float(len(pool)),
        "pool_total_poses": float(total_poses),
        "pool_total_centers": float(total_centers),
        "pool_hit_2a_rate": 100.0 * hit_2a / max(1, total_poses),
        "pool_hit_5a_rate": 100.0 * hit_5a / max(1, total_poses),
        "pool_center_hit_rate": 100.0 * center_hits / max(1, total_centers),
        "pool_oracle_mean_rmsd": float(sum(oracle_rmsds) / max(1, len(oracle_rmsds))),
        "pool_poses_per_complex": float(total_poses) / max(1, len(pool)),
        "pool_strong_center_pct": 100.0 * strong_centers / max(1, total_centers),
        "pool_weak_center_pct": 100.0 * weak_centers / max(1, total_centers),
    }
