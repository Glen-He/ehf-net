"""
验证评估工具。

负责验证损失计算、RMSD 统计和 Top-N 成功率评估，
服务训练过程中的模型监控与阶段性评测。
"""


import gc
import logging
import traceback
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from ehfnet.data import ProteinLigandDataset
from ehfnet.geometry import compute_batch_symmetry_aware_rmsd
from ehfnet.graph import GraphCollator
from ehfnet.training.adaptive_batching import (
    estimate_runtime_batch_cost,
    split_collated_batch,
)
from ehfnet.training.batch_helpers import apply_loss_context, build_local_batch_from_centers, compute_pose_rank_target
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.inference import (
    compute_center_guidance_scores,
    predict_center_proposal_logits,
    select_diverse_center_indices,
)
from ehfnet.training.losses import FlowMatchingLoss

logger = logging.getLogger(__name__)


class ValidationFailureBudgetExceeded(RuntimeError):
    """Raised when validation exceeds its configured failure budget."""


def _evaluate_local_center_recall(
    *,
    model: torch.nn.Module,
    batch: Any,
    ligand_pos: torch.Tensor,
    device: torch.device,
    topk_values: tuple[int, ...],
    center_hit_radius: float,
    center_nms_radius: float,
    learned_score_fraction: float,
) -> tuple[dict[int, int], dict[int, int], float, int]:
    """Evaluate center proposal recall inside the validation local crop."""
    if not topk_values:
        return {}, {}, 0.0, 0

    proposal_logits, residue_pos_device, residue_batch_device, residue_prior_feat = (
        predict_center_proposal_logits(model, batch, device=device)
    )
    center_scores = compute_center_guidance_scores(
        proposal_logits.detach().cpu().view(-1),
        residue_prior_feat.detach().cpu(),
        learned_score_fraction=learned_score_fraction,
    )
    residue_pos = residue_pos_device.detach().cpu()
    residue_batch = residue_batch_device.detach().cpu()
    ligand_batch = batch["ligand_atom"].batch.detach().cpu()
    num_graphs = int(ligand_batch.max().item()) + 1 if ligand_batch.numel() > 0 else 0
    true_centers = scatter_mean(
        ligand_pos.detach().cpu(),
        ligand_batch,
        dim=0,
        dim_size=num_graphs,
    )

    max_topk = max(topk_values)
    hits_4a = {topk: 0 for topk in topk_values}
    hits_6a = {topk: 0 for topk in topk_values}
    min_dist_sum = 0.0
    evaluated_graphs = 0

    for graph_idx in range(num_graphs):
        graph_mask = residue_batch == graph_idx
        graph_positions = residue_pos[graph_mask]
        if graph_positions.numel() == 0:
            continue

        graph_scores = center_scores[graph_mask]
        selected_indices = select_diverse_center_indices(
            graph_scores,
            graph_positions,
            topk=min(max_topk, graph_positions.size(0)),
            min_distance=center_nms_radius,
        ).cpu()
        if selected_indices.numel() == 0:
            continue

        selected_positions = graph_positions[selected_indices]
        distances = torch.norm(
            selected_positions - true_centers[graph_idx].view(1, 3),
            dim=-1,
        )
        min_dist_sum += float(distances.min().item())
        evaluated_graphs += 1

        for topk in topk_values:
            k = min(topk, distances.numel())
            if k <= 0:
                continue
            topk_distances = distances[:k]
            hits_4a[topk] += int(torch.any(topk_distances <= center_hit_radius).item())
            hits_6a[topk] += int(torch.any(topk_distances <= 6.0).item())

    return hits_4a, hits_6a, min_dist_sum, evaluated_graphs


def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int,
    max_rmsd_batches: int | None,
    dataset: ProteinLigandDataset | None = None,
    warmup_epochs: int,
    graph_builder: Any | None = None,
    collator: GraphCollator | None = None,
    crop_radius: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    cost_guard_limit: int | None = None,
    ode_steps: int,
    ode_method: str,
    progress_desc: str | None = None,
    num_gnn_blocks: int,
    dynamic_inter_max_neighbors: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
    phase_multiplier: float = 1.0,
    max_oom_retry_splits: int = 0,
    max_non_oom_failures: int = 0,
    max_oom_failures: int = 3,
    center_hit_radius: float = 4.0,
    center_recall_topk_values: tuple[int, ...] = (1, 3, 8),
    center_nms_radius: float = 6.0,
    learned_score_fraction: float = 1.0,
) -> dict[str, float]:
    """
    计算验证集损失与 RMSD 指标。

    在验证阶段执行前向推理、损失汇总和 RMSD 统计，
    用于训练过程中的性能监控。

    Args:
        model: 当前使用的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        criterion: 训练或验证阶段使用的损失函数对象。
        loader: 提供批次数据的 DataLoader。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        epoch: 当前训练轮次。
        total_epochs: 训练总轮数。
        max_rmsd_batches: 验证阶段允许执行 RMSD 推演的最大 batch 数；为 `None` 时表示覆盖全部 batch。
        dataset: 参与处理或划分的数据集对象。
        warmup_epochs: 课程学习预热轮数。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        crop_radius: 局部裁剪半径。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        cost_guard_limit: 验证阶段的成本保护上限。
        ode_steps: ODE 推理积分步数。
        ode_method: 验证阶段 RMSD rollout 使用的 ODE 积分方法。
        progress_desc: 终端中显示的验证进度条名称。
        num_gnn_blocks: 主干 GNN 块数量，用于估计动态图成本。
        dynamic_inter_max_neighbors: 动态原子跨图边的单源邻居上限。
        dynamic_residue_max_neighbors: 动态配体-残基边的单源邻居上限。
        dynamic_residue_candidate_topk: 动态配体-残基边每个复合物保留的候选残基数。
        phase_multiplier: 当前验证阶段的成本倍率。
        max_oom_retry_splits: 单个验证 batch 允许递归拆分重试的最大深度。
        max_non_oom_failures: Maximum non-OOM validation failures allowed before failing validation.
        max_oom_failures: Maximum irreducible OOM validation failures allowed before failing validation.
        center_hit_radius: Local center hit radius used for validation recall.
        center_recall_topk_values: Top-K values reported for local center recall.
        center_nms_radius: Minimum distance used when selecting diverse center proposals.
        learned_score_fraction: Fraction of learned center score mixed into proposal ranking.

    Returns:
        dict[str, float]: 验证损失、RMSD、亲和力和中心提议相关指标的汇总字典。

    Raises:
        RuntimeError: 当验证阶段局部裁剪或推理过程发生不可恢复错误时抛出。
    """
    model.eval()
    use_configured_cuda = device.type == "cuda"

    if use_configured_cuda:
        torch.cuda.empty_cache()

    total_loss = 0.0
    all_rmsd_init: list[torch.Tensor] = []
    all_rmsd_final: list[torch.Tensor] = []
    all_centroid_dist: list[torch.Tensor] = []
    affinity_preds: list[torch.Tensor] = []
    affinity_targets: list[torch.Tensor] = []

    valid_batches = 0
    valid_graphs = 0
    oom_batches = 0
    oom_failed_batches = 0
    oom_failed_graphs = 0
    failed_batches = 0
    failed_graphs = 0
    non_oom_failed_batches = 0
    non_oom_failed_graphs = 0
    invalid_loss_batches = 0
    invalid_loss_graphs = 0
    cost_guard_skips = 0
    cost_guard_skipped_graphs = 0
    rmsd_attempted_graphs = 0
    rmsd_valid_graphs = 0
    rmsd_failed_batches = 0
    rmsd_failed_graphs = 0
    center_topk_values = tuple(
        sorted({int(topk) for topk in center_recall_topk_values if int(topk) > 0})
    )
    center_hits_4a = {topk: 0 for topk in center_topk_values}
    center_hits_6a = {topk: 0 for topk in center_topk_values}
    center_min_dist_sum = 0.0
    center_eval_graphs = 0
    center_failed_batches = 0
    center_failed_graphs = 0

    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if use_configured_cuda:
            torch.cuda.manual_seed(42 + epoch)

    val_total_graphs = len(cast(Any, loader).dataset) if hasattr(loader, "dataset") else 0
    pbar_desc = progress_desc or f"Epoch {(epoch or 0) + 1} [Val]"
    pbar = tqdm(total=val_total_graphs, desc=pbar_desc, leave=False, unit="graphs")

    if graph_builder is None or collator is None:
        raise ValueError("graph_builder and collator are required for runtime local cropping.")

    for i, batch in enumerate(loader):
        orig_num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
        pbar.update(orig_num_graphs)
        pending_batches: list[tuple[HeteroData, int]] = [(batch, 0)]
        while pending_batches:
            batch, split_depth = pending_batches.pop(0)
            num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
            batch_samples = (
                batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
            )
            batch_cost = estimate_runtime_batch_cost(
                batch,
                num_gnn_blocks=num_gnn_blocks,
                dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
                dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
                dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
                phase_multiplier=phase_multiplier,
            )
            current_budget_limit = cost_guard_limit

            if current_budget_limit is not None and batch_cost > current_budget_limit:
                split_batches = (
                    split_collated_batch(batch, collator=collator)
                    if split_depth < max_oom_retry_splits
                    else None
                )
                if split_batches is not None:
                    pending_batches = [
                        (split_batches[0], split_depth + 1),
                        (split_batches[1], split_depth + 1),
                    ] + pending_batches
                    continue
                cost_guard_skips += 1
                cost_guard_skipped_graphs += num_graphs
                failed_graphs += num_graphs
                logger.warning(
                    f"Validation batch {i}: skip due to oversized cost batch "
                    f"(cost={batch_cost} > limit={current_budget_limit})."
                )
                continue

            ligand_centers = None
            local_batch = None
            crop_centers = None
            x_1 = None
            x_t = None
            t = None
            targets = None
            predictions = None
            loss_dict = None
            loss = None

            try:
                if use_configured_cuda:
                    torch.cuda.reset_peak_memory_stats(device=device)
                with torch.no_grad():
                    ligand_centers = scatter_mean(
                        batch["ligand_atom"].pos,
                        batch["ligand_atom"].batch,
                        dim=0,
                        dim_size=num_graphs,
                    )
                    local_batch = build_local_batch_from_centers(
                        batch,
                        centers=ligand_centers,
                        crop_radius=crop_radius,
                        crop_min_residues=crop_min_residues,
                        crop_atom_margin=crop_atom_margin,
                        graph_builder=graph_builder,
                        collator=collator,
                    )
                    batch = local_batch.to(device)
                    local_batch_samples = (
                        local_batch.to_data_list()
                        if hasattr(local_batch, "to_data_list")
                        else [local_batch]
                    )
                    crop_centers = ligand_centers.to(
                        device=device,
                        dtype=batch["ligand_atom"].pos.dtype,
                    )
                    apply_loss_context(
                        batch,
                        current_epoch=epoch if epoch is not None else total_epochs - 1,
                        total_epochs_count=total_epochs,
                        warmup_epochs_count=warmup_epochs,
                        training=False,
                    )
                    x_1 = batch["ligand_atom"].pos

                    t, x_t, targets = matcher.sample_location_and_target(
                        x_1=x_1,
                        data=batch,
                        current_epoch=epoch if epoch is not None else 0,
                        total_epochs=total_epochs,
                        placement_centers=crop_centers,
                    )

                    batch["ligand_atom"].pos = x_t
                    batch.t = t

                    predictions = model(batch, t)

                    targets["binding_affinity_target"] = batch.get("y_energy", None)
                    targets["pose_rank_target"] = compute_pose_rank_target(
                        x_t,
                        x_1,
                        batch_idx=batch["ligand_atom"].batch,
                        samples=local_batch_samples,
                        dataset_raw_dir=dataset.raw_dir,
                    )

                    loss_dict = criterion(predictions, targets, batch)
                    loss = loss_dict["total"]

                    if center_topk_values:
                        try:
                            batch_hits_4a, batch_hits_6a, batch_min_dist_sum, batch_eval_graphs = (
                                _evaluate_local_center_recall(
                                    model=model,
                                    batch=batch,
                                    ligand_pos=x_1,
                                    device=device,
                                    topk_values=center_topk_values,
                                    center_hit_radius=center_hit_radius,
                                    center_nms_radius=center_nms_radius,
                                    learned_score_fraction=learned_score_fraction,
                                )
                            )
                            for topk in center_topk_values:
                                center_hits_4a[topk] += batch_hits_4a.get(topk, 0)
                                center_hits_6a[topk] += batch_hits_6a.get(topk, 0)
                            center_min_dist_sum += batch_min_dist_sum
                            center_eval_graphs += batch_eval_graphs
                        except torch.cuda.OutOfMemoryError:
                            raise
                        except Exception as e:
                            center_failed_batches += 1
                            center_failed_graphs += num_graphs
                            logger.warning(
                                "Local center recall evaluation failed for validation batch %d "
                                "with %d graphs: %s",
                                i,
                                num_graphs,
                                e,
                            )

                loss_is_valid = (
                    not torch.isnan(loss)
                    and not torch.isinf(loss)
                    and loss.item() < 1e6
                )
                if loss_is_valid:
                    total_loss += loss.item()
                    valid_batches += 1
                    valid_graphs += num_graphs

                if loss_is_valid:
                    if t is not None:
                        valid_mask = t > 0.8
                    else:
                        valid_mask = torch.ones_like(
                            batch.get("y_energy", torch.zeros(1)),
                            dtype=torch.bool,
                        )

                    if valid_mask.any():
                        pred_aff = predictions.get("binding_affinity", None)
                        if pred_aff is not None:
                            pred_aff_valid = pred_aff[valid_mask]
                            if not torch.isnan(pred_aff_valid).any():
                                affinity_preds.append(pred_aff_valid.cpu())
                                if hasattr(batch, "y_energy_raw"):
                                    target_raw_valid = batch.y_energy_raw[valid_mask]
                                    affinity_targets.append(target_raw_valid.cpu())
                                else:
                                    y_norm = batch.get("y_energy", None)
                                    if y_norm is not None and dataset is not None:
                                        target_raw_valid = dataset.denormalize_affinity(
                                            y_norm[valid_mask].cpu()
                                        )
                                        affinity_targets.append(target_raw_valid)

                else:
                    failed_batches += 1
                    failed_graphs += num_graphs
                    non_oom_failed_batches += 1
                    non_oom_failed_graphs += num_graphs
                    invalid_loss_batches += 1
                    invalid_loss_graphs += num_graphs
                    logger.warning(
                        "Validation batch %d produced invalid loss; skipping metrics for %d graphs.",
                        i,
                        num_graphs,
                    )
                    if (
                        max_non_oom_failures >= 0
                        and non_oom_failed_batches > max_non_oom_failures
                    ):
                        raise ValidationFailureBudgetExceeded(
                            "Validation exceeded non-OOM failure budget: "
                            f"{non_oom_failed_batches} failures > {max_non_oom_failures}."
                        )

                if loss_is_valid and (max_rmsd_batches is None or i < max_rmsd_batches):
                    rmsd_attempted_graphs += num_graphs
                    try:
                        with torch.no_grad():
                            infer_batch = batch.clone()
                            infer_batch["ligand_atom"].pos = x_1

                            x_0_infer = matcher._generate_random_pose(
                                x_ref=x_1,
                                batch=infer_batch["ligand_atom"].batch,
                                B=int(infer_batch["ligand_atom"].batch.max().item()) + 1,
                                masses=infer_batch["ligand_atom"].masses,
                                torsion_indices=getattr(
                                    infer_batch,
                                    "torsion_indices",
                                    None,
                                ),
                                torsion_moving_mask=getattr(
                                    infer_batch,
                                    "torsion_moving_mask",
                                    None,
                                ),
                                seed_pos=infer_batch["ligand_atom"].get(
                                    "start_pos",
                                    None,
                                ),
                                protein_pos=infer_batch["protein_atom"].pos,
                                protein_batch=getattr(
                                    infer_batch["protein_atom"],
                                    "batch",
                                    None,
                                ),
                                placement_centers=crop_centers,
                                epoch=warmup_epochs,
                            )

                            all_rmsd_init.append(
                                torch.sqrt(
                                    scatter_mean(
                                        ((x_0_infer - x_1) ** 2).sum(dim=-1),
                                        infer_batch["ligand_atom"].batch,
                                        dim=0,
                                    )
                                ).detach().cpu()
                            )

                            infer_batch["ligand_atom"].pos = x_0_infer
                            final_pos, _ = matcher.ode_solve(
                                model=model,
                                data=infer_batch,
                                steps=ode_steps,
                                method=ode_method,
                                store_trajectory=False,
                            )

                            final_rmsd = compute_batch_symmetry_aware_rmsd(
                                current_pos=final_pos,
                                target_pos=x_1,
                                batch_idx=infer_batch["ligand_atom"].batch,
                                samples=local_batch_samples,
                                dataset_raw_dir=dataset.raw_dir,
                            ).detach().cpu()
                            all_rmsd_final.append(final_rmsd)
                            rmsd_valid_graphs += int(final_rmsd.numel())

                            B_infer = int(infer_batch["ligand_atom"].batch.max().item()) + 1
                            pred_centroid = scatter_mean(
                                final_pos,
                                infer_batch["ligand_atom"].batch,
                                dim=0,
                                dim_size=B_infer,
                            )
                            true_centroid = scatter_mean(
                                x_1,
                                infer_batch["ligand_atom"].batch,
                                dim=0,
                                dim_size=B_infer,
                            )
                            all_centroid_dist.append(
                                torch.norm(pred_centroid - true_centroid, dim=-1).detach().cpu()
                            )

                        del infer_batch, x_0_infer, final_pos

                    except Exception as e:
                        rmsd_failed_batches += 1
                        rmsd_failed_graphs += num_graphs
                        logger.warning(
                            "RMSD inference failed for batch %d with %d graphs: %s\n%s",
                            i,
                            num_graphs,
                            e,
                            traceback.format_exc(),
                        )
                        gc.collect()
                        if use_configured_cuda:
                            torch.cuda.empty_cache()

            except ValidationFailureBudgetExceeded:
                raise

            except torch.cuda.OutOfMemoryError:
                oom_batches += 1
                del predictions, loss_dict, loss, x_1, x_t, t, targets, ligand_centers, local_batch, crop_centers
                gc.collect()
                if use_configured_cuda:
                    torch.cuda.empty_cache()
                split_batches: tuple[HeteroData, HeteroData] | None = None
                if split_depth < max_oom_retry_splits:
                    try:
                        split_batches = split_collated_batch(batch, collator=collator)
                    except Exception:
                        gc.collect()
                        if use_configured_cuda:
                            torch.cuda.empty_cache()
                if split_batches is not None:
                    logger.warning(
                        f"Validation batch {i}: CUDA OOM, retrying with split depth {split_depth + 1}."
                    )
                    pending_batches = [
                        (split_batches[0], split_depth + 1),
                        (split_batches[1], split_depth + 1),
                    ] + pending_batches
                    continue
                oom_failed_batches += 1
                oom_failed_graphs += num_graphs
                failed_batches += 1
                failed_graphs += num_graphs
                logger.warning(f"Validation batch {i}: CUDA OOM, skipping and clearing cache.")
                if max_oom_failures >= 0 and oom_failed_batches > max_oom_failures:
                    raise RuntimeError(
                        "Validation exceeded irreducible OOM failure budget: "
                        f"{oom_failed_batches} failures > {max_oom_failures}."
                    )
                continue

            except Exception as e:
                failed_batches += 1
                failed_graphs += num_graphs
                non_oom_failed_batches += 1
                non_oom_failed_graphs += num_graphs
                logger.warning(
                    "Validation batch %d failed with %d graphs: %s\n%s",
                    i,
                    num_graphs,
                    e,
                    traceback.format_exc(),
                )
                del predictions, loss_dict, loss, x_1, x_t, t, targets, ligand_centers, local_batch, crop_centers
                gc.collect()
                if use_configured_cuda:
                    torch.cuda.empty_cache()
                if (
                    max_non_oom_failures >= 0
                    and non_oom_failed_batches > max_non_oom_failures
                ):
                    raise RuntimeError(
                        "Validation exceeded non-OOM failure budget: "
                        f"{non_oom_failed_batches} failures > {max_non_oom_failures}."
                    )
                continue

            del predictions, targets, loss_dict, loss, x_1, x_t, t, batch, ligand_centers, local_batch, crop_centers

    pbar.close()

    metrics: dict[str, float] = {}

    pearson_r = 0.0
    spearman_rho = 0.0
    rmse_val = float("inf")
    mae_val = float("inf")

    if len(affinity_preds) > 0 and dataset is not None:
        cat_preds = torch.cat(affinity_preds).view(-1)
        cat_targets = torch.cat(affinity_targets).view(-1)

        raw_preds: torch.Tensor = dataset.denormalize_affinity(cat_preds).detach()
        cat_targets = cat_targets.detach()

        mse_val = F.mse_loss(raw_preds, cat_targets)
        rmse_val = torch.sqrt(mse_val).item()
        mae_val = F.l1_loss(raw_preds, cat_targets).item()

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

        success_2a = (cat_rmsd_final < 2.0).float().mean().item() * 100
        success_5a = (cat_rmsd_final < 5.0).float().mean().item() * 100

        if len(all_centroid_dist) > 0:
            cat_centroid = torch.cat(all_centroid_dist)
            mean_centroid = cat_centroid.mean().item()
            median_centroid = cat_centroid.median().item()

        logger.info("-" * 60)
        logger.info(f"[Validation Full Stats] Epoch {(epoch or 0) + 1}")
        logger.info(f"  Mean RMSD: {mean_init:.2f} -> {mean_final:.2f} A | Median: {median_final:.2f} A")
        logger.info(f"  Success Rate (<2Å): {success_2a:.2f}% | (<5Å): {success_5a:.2f}%")
        logger.info(f"  Centroid Distance: Mean {mean_centroid:.2f} Å | Median {median_centroid:.2f} Å")
        logger.info("-" * 60)

    metrics["mean_rmsd_final"] = mean_final
    metrics["median_rmsd_final"] = median_final
    metrics["single_shot_success_2a"] = success_2a
    metrics["single_shot_success_5a"] = success_5a
    metrics["centroid_dist_mean"] = mean_centroid
    metrics["centroid_dist_median"] = median_centroid
    center_mean_min_dist = (
        center_min_dist_sum / center_eval_graphs
        if center_eval_graphs > 0
        else float("inf")
    )
    metrics["local_center_mean_min_dist"] = center_mean_min_dist
    metrics["local_center_eval_graphs"] = float(center_eval_graphs)
    metrics["local_center_failed_batches"] = float(center_failed_batches)
    metrics["local_center_failed_graphs"] = float(center_failed_graphs)
    for topk in center_topk_values:
        recall_4a = (
            100.0 * center_hits_4a[topk] / center_eval_graphs
            if center_eval_graphs > 0
            else 0.0
        )
        recall_6a = (
            100.0 * center_hits_6a[topk] / center_eval_graphs
            if center_eval_graphs > 0
            else 0.0
        )
        metrics[f"local_center_recall@{topk}_4a"] = recall_4a
        metrics[f"local_center_recall@{topk}_6a"] = recall_6a
    if center_eval_graphs > 0:
        topk_label = "/".join(str(topk) for topk in center_topk_values)
        recall_4a_label = "/".join(
            f"{metrics[f'local_center_recall@{topk}_4a']:.2f}"
            for topk in center_topk_values
        )
        recall_6a_label = "/".join(
            f"{metrics[f'local_center_recall@{topk}_6a']:.2f}"
            for topk in center_topk_values
        )
        logger.info(
            "  Local Center Recall@%s: 4A %s%% | 6A %s%% | Mean min dist %.2f A",
            topk_label,
            recall_4a_label,
            recall_6a_label,
            center_mean_min_dist,
        )
    metrics["oom_batches"] = float(oom_batches)
    metrics["oom_failed_batches"] = float(oom_failed_batches)
    metrics["oom_failed_graphs"] = float(oom_failed_graphs)
    metrics["valid_batches"] = float(valid_batches)
    metrics["valid_graphs"] = float(valid_graphs)
    metrics["failed_batches"] = float(failed_batches)
    metrics["failed_graphs"] = float(failed_graphs)
    metrics["non_oom_failed_batches"] = float(non_oom_failed_batches)
    metrics["non_oom_failed_graphs"] = float(non_oom_failed_graphs)
    metrics["invalid_loss_batches"] = float(invalid_loss_batches)
    metrics["invalid_loss_graphs"] = float(invalid_loss_graphs)
    metrics["cost_guard_skips"] = float(cost_guard_skips)
    metrics["cost_guard_skipped_graphs"] = float(cost_guard_skipped_graphs)
    metrics["rmsd_attempted_graphs"] = float(rmsd_attempted_graphs)
    metrics["rmsd_valid_graphs"] = float(rmsd_valid_graphs)
    metrics["rmsd_failed_batches"] = float(rmsd_failed_batches)
    metrics["rmsd_failed_graphs"] = float(rmsd_failed_graphs)
    total_graphs = val_total_graphs if val_total_graphs > 0 else valid_graphs + failed_graphs
    val_coverage = float(valid_graphs / total_graphs) if total_graphs > 0 else 0.0
    rmsd_coverage = (
        float(rmsd_valid_graphs / rmsd_attempted_graphs)
        if rmsd_attempted_graphs > 0
        else val_coverage
    )
    metrics["validation_total_graphs"] = float(total_graphs)
    metrics["val_coverage"] = val_coverage
    metrics["rmsd_coverage"] = rmsd_coverage
    metrics["val_selection_coverage"] = min(val_coverage, rmsd_coverage)

    if valid_batches == 0:
        metrics["val_loss"] = float("nan")
        return metrics

    del all_rmsd_init, all_rmsd_final, all_centroid_dist
    del affinity_preds, affinity_targets

    if use_configured_cuda:
        torch.cuda.empty_cache()

    metrics["val_loss"] = total_loss / valid_batches
    return metrics
