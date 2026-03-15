"""
验证评估工具。

负责验证损失计算、RMSD 统计和 Top-N 成功率评估，
服务训练过程中的模型监控与阶段性评测。
"""


import gc
import logging
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from ehfnet.data import ProteinLigandDataset
from ehfnet.graph import GraphCollator
from ehfnet.training.batch_helpers import apply_loss_context, build_local_batch_from_centers, compute_pose_quality_target
from ehfnet.training.candidate_generation import generate_candidates_from_loader
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.inference import summarize_blind_candidate_records
from ehfnet.training.losses import FlowMatchingLoss

logger = logging.getLogger(__name__)


def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int,
    max_rmsd_batches: int,
    dataset: ProteinLigandDataset | None = None,
    warmup_epochs: int,
    graph_builder: Any | None = None,
    collator: GraphCollator | None = None,
    crop_radius: float,
    center_proposal_weight: float,
    center_positive_radius: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    edge_guard_limit: int | None = None,
    ode_steps: int,
) -> dict | float:
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
        max_rmsd_batches: 验证阶段允许执行 RMSD 推演的最大 batch 数。
        dataset: 参与处理或划分的数据集对象。
        warmup_epochs: 课程学习预热轮数。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        crop_radius: 局部裁剪半径。
        center_proposal_weight: 中心提议损失在验证汇总中的权重。
        center_positive_radius: 中心判定为正样本时使用的距离半径。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        edge_guard_limit: 候选生成阶段的边数保护上限。
        ode_steps: ODE 推理积分步数。

    Returns:
        dict[str, float]: 验证损失、RMSD、亲和力和中心提议相关指标的汇总字典。

    Raises:
        RuntimeError: 当验证阶段局部裁剪或推理过程发生不可恢复错误时抛出。
    """
    model.eval()
    total_loss = 0.0
    all_rmsd_init: list[torch.Tensor] = []
    all_rmsd_final: list[torch.Tensor] = []
    all_centroid_dist: list[torch.Tensor] = []
    affinity_preds: list[torch.Tensor] = []
    affinity_targets: list[torch.Tensor] = []

    valid_batches = 0
    oom_batches = 0
    edge_guard_skips = 0

    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

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
            crop_centers = ligand_centers.to(device=device, dtype=batch["ligand_atom"].pos.dtype)
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
            targets["pose_quality_target"] = compute_pose_quality_target(
                x_t, x_1, batch_idx=batch["ligand_atom"].batch
            )

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"]

            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                total_loss += loss.item()
                valid_batches += 1

            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                if t is not None:
                    valid_mask = t > 0.8
                else:
                    valid_mask = torch.ones_like(batch.get("y_energy", torch.zeros(1)), dtype=torch.bool)

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
                                    target_raw_valid = dataset.denormalize_affinity(y_norm[valid_mask].cpu())
                                    affinity_targets.append(target_raw_valid)

            if i < max_rmsd_batches:
                try:
                    infer_batch = batch.clone()
                    infer_batch["ligand_atom"].pos = x_1

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

                    all_rmsd_init.append(
                        torch.sqrt(scatter_mean(((x_0_infer - x_1) ** 2).sum(dim=-1), infer_batch["ligand_atom"].batch, dim=0)).detach().cpu()
                    )

                    infer_batch["ligand_atom"].pos = x_0_infer
                    final_pos, _ = matcher.ode_solve(
                        model=model,
                        data=infer_batch,
                        steps=ode_steps,
                        method="euler",
                        store_trajectory=False,
                    )

                    all_rmsd_final.append(
                        torch.sqrt(scatter_mean(((final_pos - x_1) ** 2).sum(dim=-1), infer_batch["ligand_atom"].batch, dim=0)).detach().cpu()
                    )

                    B_infer = int(infer_batch["ligand_atom"].batch.max().item()) + 1
                    pred_centroid = scatter_mean(final_pos, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    true_centroid = scatter_mean(x_1, infer_batch["ligand_atom"].batch, dim=0, dim_size=B_infer)
                    all_centroid_dist.append(torch.norm(pred_centroid - true_centroid, dim=-1).detach().cpu())

                    del infer_batch, x_0_infer, final_pos

                except Exception as e:
                    logger.warning(f"RMSD inference failed for batch {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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

        del predictions, targets, loss_dict, loss, x_1, x_t, t, batch, ligand_centers, local_batch, crop_centers

    metrics: dict[str, float] = {}

    pearson_r = 0.0
    spearman_rho = 0.0
    rmse_val = float("inf")
    mae_val = float("inf")

    if len(affinity_preds) > 0 and dataset is not None:
        cat_preds = torch.cat(affinity_preds).view(-1)
        cat_targets = torch.cat(affinity_targets).view(-1)

        raw_preds: torch.Tensor = dataset.denormalize_affinity(cat_preds)

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
    topk_values: tuple[int, ...],
    num_pose_samples: int,
    center_topk: int,
    refine_topk: int,
    center_nms_radius: float,
    stage1_pose_samples: int,
    stage2_pose_samples: int,
    crop_radius: float,
    ode_steps: int,
    warmup_epochs: int,
    crop_min_residues: int,
    crop_atom_margin: float,
    edge_guard_limit: int | None = None,
    center_hit_radius: float,
    fusion_weights: dict[str, float] | None = None,
    return_candidate_records: bool = False,
) -> dict[str, Any]:
    """
    评估 Top-N 对接成功率。

    基于统一候选生成流程统计 Top-N 成功率与相关指标，
    用于衡量完整两阶段盲对接流水线的效果。

    Args:
        model: 当前使用的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        loader: 提供批次数据的 DataLoader。
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        topk_values: 需要统计的 Top-K 指标列表。
        num_pose_samples: 每个复合物采样的候选构象数。
        center_topk: 中心提议阶段保留的候选中心数量。
        refine_topk: 局部重排序阶段保留的候选构象数量。
        center_nms_radius: 中心去重时使用的最小间距半径。
        stage1_pose_samples: 第一阶段局部对接生成的候选构象数。
        stage2_pose_samples: 第二阶段精排生成的候选构象数。
        crop_radius: 局部裁剪半径。
        ode_steps: ODE 推理积分步数。
        warmup_epochs: 课程学习预热轮数。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        edge_guard_limit: 候选生成阶段的边数保护上限。
        center_hit_radius: 判断中心命中的距离阈值。
        fusion_weights: 融合不同分支分数时使用的权重字典。
        return_candidate_records: 是否同时返回候选记录明细。

    Returns:
        dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]: Top-N 指标字典，必要时附带候选记录明细。

    Raises:
        RuntimeError: 当候选生成流程发生不可恢复错误时抛出。
    """

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
        crop_min_residues=crop_min_residues,
        crop_atom_margin=crop_atom_margin,
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
