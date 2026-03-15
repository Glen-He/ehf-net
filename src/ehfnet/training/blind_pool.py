"""
Blind pool 工具。

负责候选池结果的缓存、回放、采样与统计，
支撑两阶段训练中的候选重放流程。
"""


import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ehfnet.graph import GraphCollator, crop_graph_to_center
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from ehfnet.training.inference import predict_center_proposal_logits
from ehfnet.training.rerank_losses import (
    compute_center_value_loss,
    compute_rerank_losses,
    rmsd_to_soft_target,
)

logger = logging.getLogger(__name__)

PAIR_POSITIVE_RMSD_THRESHOLD = 2.0
PAIR_MIN_RMSD_GAP = 0.25


def _load_pool_manifest(epoch_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(epoch_dir) / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read blind pool manifest %s: %s", manifest_path, exc)
        return None

    return loaded if isinstance(loaded, dict) else None


def _pool_manifest_matches(
    manifest: dict[str, Any] | None,
    expected_signature: dict[str, Any] | None,
) -> bool:
    if expected_signature is None:
        return True
    if manifest is None:
        return False
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        return False
    return signature == expected_signature


@torch.no_grad()
def refresh_blind_candidate_pool(
    *,
    model: torch.nn.Module,
    matcher: Any,
    loader: DataLoader,
    device: torch.device,
    graph_builder: Any,
    collator: GraphCollator,
    center_topk: int,
    refine_topk: int,
    center_nms_radius: float,
    stage1_pose_samples: int,
    stage2_pose_samples: int,
    crop_radius: float,
    ode_steps: int,
    warmup_epochs: int,
    center_hit_radius: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    max_complexes: int | None = None,
    fusion_weights: dict[str, float] | None = None,
    use_learned_center_scores: bool = True,
    pool_epoch: int = -1,
    generator_ckpt_id: str = "",
) -> list[dict[str, Any]]:
    """
    刷新 blind 候选池。

    运行完整候选生成流程并更新缓存候选池内容，
    为后续回放训练提供新的局部候选样本。

    Args:
        model: 当前使用的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        loader: 提供批次数据的 DataLoader。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        center_topk: 中心提议阶段保留的候选中心数量。
        refine_topk: 局部重排序阶段保留的候选构象数量。
        center_nms_radius: 中心去重时使用的最小间距半径。
        stage1_pose_samples: 第一阶段局部对接生成的候选构象数。
        stage2_pose_samples: 第二阶段精排生成的候选构象数。
        crop_radius: 局部裁剪半径。
        ode_steps: ODE 推理积分步数。
        warmup_epochs: 课程学习预热轮数。
        center_hit_radius: 判断中心命中的距离阈值。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        max_complexes: 本轮最多处理的复合物数量。
        fusion_weights: 融合不同分支分数时使用的权重字典。
        use_learned_center_scores: 是否优先使用模型学习得到的中心分数。
        pool_epoch: 当前候选池对应的训练轮次。
        generator_ckpt_id: 生成候选时使用的 checkpoint 标识。

    Returns:
        list[dict[str, Any]]: 当前轮次生成的候选池记录列表。
    """
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
        crop_min_residues=crop_min_residues,
        crop_atom_margin=crop_atom_margin,
        max_complexes=max_complexes,
        fusion_weights=fusion_weights,
        use_learned_center_scores=use_learned_center_scores,
        pool_epoch=pool_epoch,
        generator_ckpt_id=generator_ckpt_id,
    )


def save_blind_pool(
    records: list[dict[str, Any]],
    cache_dir: str,
    *,
    epoch: int,
    meta: dict[str, Any] | None = None,
) -> str:
    """
    保存 blind 候选池。

    将候选记录、签名字典和清单文件写入磁盘，
    供训练恢复或后续回放阶段重复使用。

    Args:
        records: 待保存的候选池记录列表。
        cache_dir: 缓存目录路径。
        epoch: 当前训练轮次。
        meta: 与候选池一同保存的附加元数据。

    Returns:
        str: 返回当前轮次 blind pool 缓存文件的保存路径。
    """
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


def load_blind_pool(
    cache_dir: str,
    epoch: int | None = None,
    *,
    expected_signature: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    加载 blind 候选池。

    从磁盘读取最近一次或指定运行的候选池缓存，
    并在必要时完成缓存签名校验。

    Args:
        cache_dir: 缓存目录路径。
        epoch: 当前训练轮次。
        expected_signature: 读取缓存时期望匹配的 blind pool 签名字典。

    Returns:
        list[dict[str, Any]]: 从缓存中读取到的候选池记录列表。
    """
    if not os.path.isdir(cache_dir):
        return []

    def _try_load_epoch(epoch_dir: str | Path) -> list[dict[str, Any]] | None:
        pool_path = Path(epoch_dir) / "pool.pt"
        if not pool_path.exists():
            return None
        manifest = _load_pool_manifest(epoch_dir)
        if not _pool_manifest_matches(manifest, expected_signature):
            logger.info(
                "Skipping blind pool cache with mismatched signature at %s.",
                epoch_dir,
            )
            return None
        return torch.load(str(pool_path), weights_only=False)

    if epoch is not None:
        epoch_dir = os.path.join(cache_dir, f"epoch_{epoch:04d}")
        pool = _try_load_epoch(epoch_dir)
        if pool is not None:
            return pool

    epoch_dirs = sorted(Path(cache_dir).glob("epoch_*"), reverse=True)
    for d in epoch_dirs:
        pool = _try_load_epoch(d)
        if pool is not None:
            return pool

    return []


class BlindCandidateReplayDataset(Dataset):
    """
    从 blind pool 中按 complex 采样可回放候选组。

    每个条目返回一个复合物的多个候选，每个候选包含：
    - dataset_index: 用于从 train_set 取缓存样本
    - center_xyz: 用于 crop_graph_to_center
    - pose_xyz: 用于覆盖 ligand_atom.pos
    - rmsd, soft_target: 用于计算损失
    - center_value_target: 中心价值监督目标

    不返回教师模型打分作为训练主输入。
    """

    def __init__(
        self,
        pool: list[dict[str, Any]],
        *,
        candidates_per_complex: int,
        positive_rmsd_threshold: float,
        hard_negative_clash_threshold: float,
    ):
        """
        初始化 blind pool 回放数据集。

        根据候选池记录配置每个复合物的采样与配对规则，
        为回放训练阶段提供可迭代的数据接口。

        Args:
            pool: 候选池记录或其汇总对象。
            candidates_per_complex: 候选集合percomplex。
            positive_rmsd_threshold: 正例RMSD使用的阈值。
            hard_negative_clash_threshold: hard负例位阻使用的阈值。
        """
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

        n = self.candidates_per_complex
        if positives:
            k = min(max(1, n // 3), len(positives))
            self._extend_unique(selected, positives, k)

        remaining = n - len(selected)
        if remaining > 0:
            for pool_list in [same_center_negs, wrong_center_negs, deceptive_negs]:
                if pool_list and remaining > 0:
                    k = min(max(1, remaining // 2), len(pool_list))
                    self._extend_unique(selected, pool_list, k)
                    remaining = n - len(selected)

        if len(selected) < n:
            seen = {self._candidate_key(p) for p in selected}
            others = [p for p in poses if self._candidate_key(p) not in seen]
            if others:
                k = min(n - len(selected), len(others))
                self._extend_unique(selected, others, k)

        candidates = []
        for p in selected:
            center_xyz = self._get_center_xyz(p, centers)
            if center_xyz is None:
                continue
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
                "center_xyz": center_xyz,
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

    @staticmethod
    def _candidate_key(pose: dict[str, Any]) -> tuple[Any, ...]:
        pose_id = pose.get("pose_id")
        if pose_id is not None:
            return ("pose_id", int(pose_id))
        return (
            int(pose.get("center_id", -1)),
            str(pose.get("stage_id", "")),
            round(float(pose.get("rmsd", 999.0)), 4),
        )

    def _extend_unique(
        self,
        selected: list[dict[str, Any]],
        pool_list: list[dict[str, Any]],
        k: int,
    ) -> None:
        if k <= 0:
            return
        seen = {self._candidate_key(p) for p in selected}
        available = [p for p in pool_list if self._candidate_key(p) not in seen]
        if not available:
            return
        selected.extend(random.sample(available, min(k, len(available))))

    def _get_center_xyz(self, pose: dict, centers: list[dict]) -> list[float] | None:
        cid = pose["center_id"]
        for c in centers:
            if c["center_id"] == cid:
                return c["center_xyz"]
        return None


def replay_and_compute_losses(
    *,
    model: torch.nn.Module,
    replay_items: list[dict[str, Any]],
    train_set: Any,
    graph_builder: Any,
    collator: GraphCollator,
    device: torch.device,
    crop_radius: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    margin: float,
    lambda_bce: float,
    lambda_pair: float,
    lambda_list: float,
    lambda_center_value: float,
    use_pose_rank_head: bool,
) -> dict[str, Tensor]:
    """
    将缓存候选通过当前模型重放并计算 rerank 损失。

    这是 blind pool 训练真正驱动当前模型学习的核心函数。
    与使用缓存 teacher logit 不同，本函数执行以下流程：
    1. 通过 dataset_index 加载缓存样本
    2. 按 center_xyz 裁剪图
    3. 用 pose_xyz 覆盖 ligand_atom.pos
    4. 对当前模型做前向推理
    5. 基于 RMSD 目标计算 BCE + pairwise + listwise 损失

    Args:
        model: 当前使用的模型实例。
        replay_items: replayitems。
        train_set: trainset。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        crop_radius: 局部裁剪半径。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        margin: 排序损失中的最小边界间隔。
        lambda_bce: BCE 损失权重。
        lambda_pair: pairwise 损失权重。
        lambda_list: listwise 损失权重。
        lambda_center_value: lambda中心数值。
        use_pose_rank_head: 是否使用构象rankhead。

    Returns:
        dict[str, Tensor | float]: 当前模型在 blind pool 回放样本上的损失与统计信息。

    Raises:
        RuntimeError: 当回放过程中没有得到任何有效局部样本时抛出。
    """
    per_group_totals: list[Tensor] = []
    per_group_bce: list[Tensor] = []
    per_group_pair: list[Tensor] = []
    per_group_list: list[Tensor] = []
    total_pairs = 0
    all_center_logits: list[Tensor] = []
    all_center_targets: list[Tensor] = []
    resolved_groups = 0
    candidate_successes = 0
    candidate_failures = 0
    sample_failures = 0
    center_value_failures = 0

    for item in replay_items:
        candidates = item["candidates"]
        center_values = item.get("center_values", {})
        if not candidates:
            continue

        ds_idx = int(item["dataset_index"])

        try:
            sample = _resolve_replay_sample(train_set, ds_idx)
        except Exception as exc:
            sample_failures += 1
            if sample_failures <= 5:
                logger.warning(
                    "Replay sample resolution failed for dataset_index=%d: %s",
                    ds_idx,
                    exc,
                )
            continue

        group_logits: list[Tensor] = []
        group_rmsd: list[Tensor] = []
        valid_candidates: list[dict[str, Any]] = []

        for cand in candidates:
            try:
                center_xyz = torch.tensor(cand["center_xyz"], dtype=torch.float32)
                pose_xyz = torch.tensor(cand["pose_xyz"], dtype=torch.float32)

                local_sample = crop_graph_to_center(
                    sample, center=center_xyz, radius=crop_radius,
                    min_residues=crop_min_residues,
                    atom_margin=crop_atom_margin,
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
                valid_candidates.append(cand)
                candidate_successes += 1

                del infer_batch, out

            except torch.cuda.OutOfMemoryError:
                candidate_failures += 1
                logger.warning("Replay OOM on candidate, skipping.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as exc:
                candidate_failures += 1
                if candidate_failures <= 5:
                    logger.warning(
                        "Replay candidate failed for complex=%s center_id=%s: %s",
                        item.get("complex_id", ""),
                        cand.get("center_id", "?"),
                        exc,
                    )
                continue

        if group_logits:
            resolved_groups += 1
            logits_cat = torch.cat(group_logits)
            rmsd_cat = torch.cat(group_rmsd)
            pair_indices = _build_group_pair_indices(
                valid_candidates,
                device=device,
                positive_rmsd_threshold=PAIR_POSITIVE_RMSD_THRESHOLD,
                min_rmsd_gap=PAIR_MIN_RMSD_GAP,
            )
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

        if lambda_center_value > 0 and center_values:
            try:
                sample_batch = cast(Any, collator.collate([sample])).to(device)

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
            except Exception as exc:
                center_value_failures += 1
                if center_value_failures <= 5:
                    logger.warning(
                        "Center-value replay failed for complex=%s: %s",
                        item.get("complex_id", ""),
                        exc,
                    )

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

    if resolved_groups == 0 and (candidate_failures > 0 or sample_failures > 0):
        raise RuntimeError(
            "Replay supervision produced zero valid groups. "
            f"sample_failures={sample_failures}, candidate_failures={candidate_failures}, "
            f"center_value_failures={center_value_failures}"
        )

    if candidate_failures > 0 or sample_failures > 0 or center_value_failures > 0:
        logger.warning(
            "Replay supervision summary | valid_groups=%d | candidate_successes=%d | "
            "sample_failures=%d | candidate_failures=%d | center_value_failures=%d",
            resolved_groups,
            candidate_successes,
            sample_failures,
            candidate_failures,
            center_value_failures,
        )

    return result


def _resolve_replay_sample(train_subset_or_dataset: Any, dataset_index: int) -> Any:
    """
    根据底层数据集解析 replay 样本。

    blind pool 存储的是底层数据集的索引；当训练使用
    torch.utils.data.Subset 包装器时，replay 必须绕过子集局部索引。

    Args:
        train_subset_or_dataset: 训练阶段使用的数据集或 `Subset` 包装对象。
        dataset_index: blind pool 中记录的底层数据集样本索引。

    Returns:
        Any: 返回与底层数据集索引对应的原始训练样本。
    """
    if hasattr(train_subset_or_dataset, "dataset") and hasattr(train_subset_or_dataset, "indices"):
        return train_subset_or_dataset.dataset[int(dataset_index)]
    return train_subset_or_dataset[int(dataset_index)]


def _build_group_pair_indices(
    candidates: list[dict[str, Any]],
    *,
    device: torch.device,
    positive_rmsd_threshold: float,
    min_rmsd_gap: float,
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


def should_refresh_pool(
    epoch: int,
    *,
    refresh_every: int,
    min_start_epoch: int,
    best_updated_this_epoch: bool,
) -> bool:
    """
    判断是否刷新候选池。

    根据训练轮次和刷新间隔决定当前是否需要重建 blind pool，
    用于控制候选池更新频率和训练开销。

    Args:
        epoch: 当前训练轮次。
        refresh_every: 候选池刷新间隔。
        min_start_epoch: 允许开始刷新候选池的最小训练轮次。
        best_updated_this_epoch: 当前轮次是否更新了最佳模型。

    Returns:
        bool: 返回布尔判断结果。
    """
    if epoch < min_start_epoch:
        return False
    if best_updated_this_epoch:
        return True
    return epoch % refresh_every == 0


def get_pool_stats(pool: list[dict[str, Any]]) -> dict[str, float]:
    """
    汇总候选池统计。

    从候选记录中整理关键统计量，
    供日志输出和训练监控流程直接使用。

    Args:
        pool: 候选池记录或其汇总对象。

    Returns:
        dict[str, float]: 适合直接写入日志的候选池统计信息。
    """
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
