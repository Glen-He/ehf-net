"""
候选生成流程。

负责执行中心提议到局部对接的完整候选生成，
是 blind pipeline 的统一实现入口。
"""


import gc
import logging
import traceback
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean

from ehfnet.graph import GraphCollator, crop_graph_to_center
from ehfnet.training.rerank_losses import rmsd_to_soft_target
from ehfnet.training.inference import (
    combine_center_pose_score,
    compute_center_guidance_scores,
    predict_center_proposal_logits,
    select_diverse_center_indices,
)

logger = logging.getLogger(__name__)


def _classify_rank_bucket(rmsd: float) -> str:
    if rmsd < 2.0:
        return "near"
    if rmsd < 5.0:
        return "medium"
    return "bad"


def _classify_center_success(best_rmsd: float) -> str:
    if best_rmsd < 2.0:
        return "strong_positive"
    if best_rmsd < 5.0:
        return "weak_positive"
    return "negative"


@torch.no_grad()
def generate_blind_candidates(
    *,
    model: torch.nn.Module,
    matcher: Any,
    samples: list[Any],
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
    fusion_weights: dict[str, float] | None = None,
    use_learned_center_scores: bool = True,
    dataset_indices: list[int] | None = None,
    pool_epoch: int = -1,
    generator_ckpt_id: str = "",
) -> list[dict[str, Any]]:
    """
    在给定样本列表上运行完整的两阶段 blind pipeline。

    本函数是候选生成的 *唯一真实来源*。
    验证指标计算和 blind pool 刷新均调用此函数。

    Args:
        model: 当前使用的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        samples: 待拼接或处理的样本列表。
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
        fusion_weights: 融合不同分支分数时使用的权重字典。
        use_learned_center_scores: 是否优先使用模型学习得到的中心分数。
        dataset_indices: 与样本一一对应的原始数据集索引。
        pool_epoch: 当前候选池对应的训练轮次。
        generator_ckpt_id: 生成候选时使用的 checkpoint 标识。

    Returns:
        list[dict[str, Any]]: 每个复合物对应的候选中心与构象记录列表。

    Raises:
        RuntimeError: 当候选生成过程出现不可恢复的局部对接错误时抛出。
    """
    model.eval()
    records: list[dict[str, Any]] = []
    failed_samples = 0

    for sample_idx, sample in enumerate(samples):
        try:
            batch = collator.collate([sample])
            proposal_logits, residue_pos_device, residue_batch_device, residue_prior_feat = predict_center_proposal_logits(
                model, batch, device=device,
            )
            graph_logits = compute_center_guidance_scores(
                proposal_logits.detach().cpu().view(-1),
                residue_prior_feat.detach().cpu(),
                use_learned_scores=use_learned_center_scores,
            )
            graph_positions = residue_pos_device.detach().cpu()

            if graph_positions.numel() == 0:
                continue

            pdb_id = getattr(sample, "pdb_id", f"sample_{sample_idx}")
            gt_pos = sample["ligand_atom"].pos.detach().cpu()
            gt_center = gt_pos.mean(dim=0)

            center_indices = select_diverse_center_indices(
                graph_logits, graph_positions,
                topk=min(center_topk, graph_positions.size(0)),
                min_distance=center_nms_radius,
            ).cpu()

            center_records: list[dict[str, Any]] = []
            pose_records: list[dict[str, Any]] = []
            center_best_stage1: list[tuple[float, int, Tensor, float]] = []
            pose_counter = 0

            for proposal_rank, center_idx in enumerate(center_indices.tolist(), start=1):
                center_pos = graph_positions[center_idx]
                center_logit = float(graph_logits[center_idx].item())
                center_to_gt = float(torch.norm(center_pos - gt_center).item())

                center_rec: dict[str, Any] = {
                    "center_id": proposal_rank,
                    "center_xyz": center_pos.tolist(),
                    "proposal_logit": center_logit,
                    "proposal_rank": proposal_rank,
                    "is_center_hit_4A": center_to_gt <= center_hit_radius,
                    "is_center_hit_6A": center_to_gt <= 6.0,
                    "center_to_gt_dist": center_to_gt,
                }

                local_sample = crop_graph_to_center(
                    sample, center=center_pos, radius=crop_radius,
                    min_residues=crop_min_residues,
                    atom_margin=crop_atom_margin,
                    graph_builder=graph_builder,
                )

                best_s1_score = -1e9
                best_s1_rmsd = 999.0

                for pose_id in range(stage1_pose_samples):
                    pose_rec = _generate_single_pose(
                        local_sample=local_sample,
                        center_pos=center_pos,
                        center_logit=center_logit,
                        gt_pos=gt_pos,
                        model=model,
                        matcher=matcher,
                        collator=collator,
                        device=device,
                        ode_steps=ode_steps,
                        warmup_epochs=warmup_epochs,
                        epoch_offset=pose_id,
                        stage_id="stage1",
                        center_id=proposal_rank,
                        pose_counter=pose_counter,
                        fusion_weights=fusion_weights,
                    )
                    if pose_rec is not None:
                        pose_records.append(pose_rec)
                        pose_counter += 1
                        best_s1_score = max(best_s1_score, pose_rec["combined_score"])
                        best_s1_rmsd = min(best_s1_rmsd, pose_rec["rmsd"])

                center_rec["stage1_best_score"] = best_s1_score
                center_rec["stage1_best_rmsd"] = best_s1_rmsd
                center_records.append(center_rec)
                center_best_stage1.append((best_s1_score, proposal_rank, center_pos, center_logit))

            center_best_stage1.sort(key=lambda x: x[0], reverse=True)
            refined_center_ids = set()

            for _, c_rank, center_pos, center_logit_val in center_best_stage1[:max(1, min(refine_topk, len(center_best_stage1)))]:
                refined_center_ids.add(c_rank)
                local_sample = crop_graph_to_center(
                    sample, center=center_pos, radius=crop_radius,
                    min_residues=crop_min_residues,
                    atom_margin=crop_atom_margin,
                    graph_builder=graph_builder,
                )
                for pose_id in range(stage2_pose_samples):
                    pose_rec = _generate_single_pose(
                        local_sample=local_sample,
                        center_pos=center_pos,
                        center_logit=center_logit_val,
                        gt_pos=gt_pos,
                        model=model,
                        matcher=matcher,
                        collator=collator,
                        device=device,
                        ode_steps=ode_steps,
                        warmup_epochs=warmup_epochs,
                        epoch_offset=stage1_pose_samples + pose_id,
                        stage_id="stage2",
                        center_id=c_rank,
                        pose_counter=pose_counter,
                        fusion_weights=fusion_weights,
                    )
                    if pose_rec is not None:
                        pose_records.append(pose_rec)
                        pose_counter += 1

            if not pose_records:
                continue

            for crec in center_records:
                cid = crec["center_id"]
                center_poses = [p for p in pose_records if p["center_id"] == cid]
                if center_poses:
                    best_rmsd = min(p["rmsd"] for p in center_poses)
                    crec["center_success_label"] = _classify_center_success(best_rmsd)
                else:
                    crec["center_success_label"] = "negative"

            sample_dataset_index = getattr(sample, "dataset_index", None)
            if sample_dataset_index is None and dataset_indices is not None:
                sample_dataset_index = dataset_indices[sample_idx]
            if sample_dataset_index is None:
                sample_dataset_index = sample_idx

            records.append({
                "complex_id": str(pdb_id),
                "dataset_index": int(sample_dataset_index),
                "pdb_id": str(pdb_id),
                "gt_center_xyz": gt_center.tolist(),
                "n_ligand_atoms": int(gt_pos.size(0)),
                "pool_epoch": pool_epoch,
                "generator_ckpt_id": generator_ckpt_id,
                "pipeline_config": {
                    "proposal_topk": center_topk,
                    "center_refine_topk": refine_topk,
                    "stage1_pose_samples": stage1_pose_samples,
                    "stage2_pose_samples": stage2_pose_samples,
                    "ode_steps": ode_steps,
                    "crop_radius": crop_radius,
                },
                "centers": center_records,
                "poses": pose_records,
            })

            del batch
        except torch.cuda.OutOfMemoryError:
            failed_samples += 1
            logger.warning("Candidate generation: OOM on sample %d, skipping.", sample_idx)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failed_samples += 1
            logger.warning("Candidate generation: sample %d failed: %s\n%s", sample_idx, exc, traceback.format_exc())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not records and failed_samples > 0 and samples:
        raise RuntimeError(
            f"Blind candidate generation failed for all samples in batch. failed_samples={failed_samples}"
        )

    if failed_samples > 0:
        logger.warning(
            "Candidate generation skipped %d/%d samples.",
            failed_samples,
            len(samples),
        )

    return records


def _generate_single_pose(
    *,
    local_sample: Any,
    center_pos: Tensor,
    center_logit: float,
    gt_pos: Tensor,
    model: torch.nn.Module,
    matcher: Any,
    collator: GraphCollator,
    device: torch.device,
    ode_steps: int,
    warmup_epochs: int,
    epoch_offset: int,
    stage_id: str,
    center_id: int,
    pose_counter: int,
    fusion_weights: dict[str, float] | None,
) -> dict[str, Any] | None:
    """
    在给定中心点生成一个构象。发生 OOM 时返回 None。

    Returns:
        dict[str, Any] | None: 返回单个候选构象的记录字典；若生成过程中发生 OOM 则返回 `None`。
    """
    try:
        infer_batch = cast(Any, collator.collate([local_sample])).to(device)
        x_ref = infer_batch["ligand_atom"].pos
        lig_batch = infer_batch["ligand_atom"].batch
        masses = infer_batch["ligand_atom"].masses

        x0 = matcher._generate_random_pose(
            x_ref=x_ref, batch=lig_batch, B=1, masses=masses,
            torsion_indices=getattr(infer_batch, "torsion_indices", None),
            torsion_moving_mask=getattr(infer_batch, "torsion_moving_mask", None),
            seed_pos=infer_batch["ligand_atom"].get("start_pos", None),
            protein_pos=infer_batch["protein_atom"].pos,
            protein_batch=getattr(infer_batch["protein_atom"], "batch", None),
            placement_centers=center_pos.view(1, 3).to(device=device, dtype=x_ref.dtype),
            epoch=warmup_epochs + epoch_offset,
        )
        infer_batch["ligand_atom"].pos = x0
        final_pos, _ = matcher.ode_solve(
            model=model, data=infer_batch, steps=ode_steps,
            method="euler", store_trajectory=False,
        )

        sq_diff = ((final_pos - x_ref) ** 2).sum(dim=-1)
        rmsd = float(torch.sqrt(scatter_mean(sq_diff, lig_batch, dim=0, dim_size=1))[0].item())

        centroid_pred = final_pos.mean(dim=0).detach().cpu()
        centroid_gt = x_ref.mean(dim=0).detach().cpu()
        centroid_dist = float(torch.norm(centroid_pred - centroid_gt).item())

        score_batch = infer_batch.clone()
        score_batch["ligand_atom"].pos = final_pos
        score_out = model(score_batch, torch.ones(1, device=device, dtype=final_pos.dtype))

        pose_quality_logit = float(score_out["pose_quality"].detach().cpu().view(-1)[0].item())
        aff_raw = score_out.get("binding_affinity", torch.zeros(1))
        clash_raw = score_out.get("steric_clash_batch", None)
        force_atom = score_out.get("ligand_force", None)
        rank_score_raw = score_out.get("pose_rank_score", None)

        aff_val = float(aff_raw.detach().cpu().view(-1)[0].item())
        clash_val = float(clash_raw.detach().cpu().view(-1)[0].item()) if clash_raw is not None else 0.0
        rank_score = float(rank_score_raw.detach().cpu().view(-1)[0].item()) if rank_score_raw is not None else None
        primary_ranking_logit = rank_score if rank_score is not None else pose_quality_logit

        center_logit_t = torch.tensor([center_logit], dtype=torch.float32)
        pose_logit_t = torch.tensor([primary_ranking_logit], dtype=torch.float32)
        combined_score = float(combine_center_pose_score(
            center_logit_t, pose_logit_t,
            aff_logit=torch.tensor([aff_val]),
            clash_value=torch.tensor([clash_val]),
            fusion_weights=fusion_weights,
        )[0].item())

        pose_xyz = final_pos.detach().cpu().tolist()

        del infer_batch, x_ref, lig_batch, masses, x0, final_pos, sq_diff
        del score_batch, score_out

        rec: dict[str, Any] = {
            "pose_id": pose_counter,
            "center_id": center_id,
            "stage_id": stage_id,
            "pose_xyz": pose_xyz,
            "rmsd": rmsd,
            "centroid_dist": centroid_dist,
            "ranking_logit": primary_ranking_logit,
            "pose_quality_logit": pose_quality_logit,
            "binding_affinity_teacher": aff_val,
            "steric_clash_teacher": clash_val,
            "center_logit": center_logit,
            "combined_score": combined_score,
            "is_hit_2A": rmsd < 2.0,
            "is_hit_5A": rmsd < 5.0,
            "soft_target": float(
                rmsd_to_soft_target(torch.tensor(rmsd, dtype=torch.float32)).item()
            ),
            "rank_bucket": _classify_rank_bucket(rmsd),
        }
        return rec

    except torch.cuda.OutOfMemoryError:
        logger.warning("Single pose generation OOM at center_id=%d, stage=%s", center_id, stage_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


@torch.no_grad()
def generate_candidates_from_loader(
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
    fusion_weights: dict[str, float] | None = None,
    use_learned_center_scores: bool = True,
    edge_guard_limit: int | None = None,
    max_complexes: int | None = None,
    pool_epoch: int = -1,
    generator_ckpt_id: str = "",
) -> list[dict[str, Any]]:
    """
    便捷封装：遍历 DataLoader，拆分 batch 后逐样本生成候选。

    本函数取代了 ``evaluate_topn_success``（部分）和
    ``refresh_blind_candidate_pool`` 的功能。

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
        fusion_weights: 融合不同分支分数时使用的权重字典。
        use_learned_center_scores: 是否优先使用模型学习得到的中心分数。
        edge_guard_limit: 候选生成阶段的边数保护上限。
        max_complexes: 本轮最多处理的复合物数量。
        pool_epoch: 当前候选池对应的训练轮次。
        generator_ckpt_id: 生成候选时使用的 checkpoint 标识。

    Returns:
        list[dict[str, Any]]: 按 DataLoader 遍历得到的完整候选记录列表。

    Raises:
        RuntimeError: 当批次拆分或候选生成流程失败时抛出。
    """
    all_records: list[dict[str, Any]] = []
    total_complexes = 0
    batches_seen = 0
    failed_batches = 0

    for batch_idx, batch in enumerate(loader):
        batches_seen += 1
        try:
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
                    logger.warning("Candidate gen batch %d: edge-guard skip (%d > %d)", batch_idx, total_edges_cpu, edge_guard_limit)
                    continue

            data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
            batch_dataset_indices = [
                int(getattr(sample, "dataset_index"))
                for sample in data_list
                if getattr(sample, "dataset_index", None) is not None
            ]

            if max_complexes is not None and total_complexes + len(data_list) > max_complexes:
                data_list = data_list[:max(1, max_complexes - total_complexes)]

            batch_records = generate_blind_candidates(
                model=model,
                matcher=matcher,
                samples=data_list,
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
                fusion_weights=fusion_weights,
                use_learned_center_scores=use_learned_center_scores,
                dataset_indices=batch_dataset_indices if len(batch_dataset_indices) == len(data_list) else None,
                pool_epoch=pool_epoch,
                generator_ckpt_id=generator_ckpt_id,
            )
            all_records.extend(batch_records)
            total_complexes += len(batch_records)

            del batch, data_list

        except torch.cuda.OutOfMemoryError:
            failed_batches += 1
            logger.warning("Candidate gen batch %d: CUDA OOM, skipping.", batch_idx)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failed_batches += 1
            logger.warning("Candidate gen batch %d failed: %s\n%s", batch_idx, exc, traceback.format_exc())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if max_complexes is not None and total_complexes >= max_complexes:
            break

    if not all_records and failed_batches > 0 and batches_seen > 0:
        raise RuntimeError(
            f"Candidate generation failed for all processed batches. failed_batches={failed_batches}"
        )

    return all_records
