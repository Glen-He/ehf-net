"""
训练 batch 辅助工具。

负责局部 batch 构造、上下文字段注入和 ranking 输入整理，
为训练主循环提供批级别辅助逻辑。
"""


from typing import Any

import torch

from ehfnet.training.rerank_losses import SOFT_TARGET_CENTER, SOFT_TARGET_SCALE
from torch_scatter import scatter_mean

from ehfnet.graph import GraphCollator, crop_graph_to_center


def compute_pose_quality_target(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """
    计算 pose 质量软目标。

    根据 RMSD 将真实构象质量映射为平滑监督值，
    供排序和 pose quality 相关损失使用。

    Args:
        current_pos: 当前构象坐标。
        target_pos: 目标构象坐标。
        batch_idx: 原子或样本所属的 batch 索引。

    Returns:
        Tensor: 与 batch 对齐的 pose quality 软目标张量。
    """
    sq_diff = ((current_pos - target_pos) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(scatter_mean(sq_diff, batch_idx, dim=0) + 1e-8)
    return torch.sigmoid((SOFT_TARGET_CENTER - rmsd) / SOFT_TARGET_SCALE).unsqueeze(-1)


def select_pose_ranking_logit(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    选择排序打分张量。

    从模型预测结果中提取用于排序的主 logit，
    在不同输出字段之间提供统一访问入口。

    Args:
        predictions: 模型前向传播返回的预测结果字典。

    Returns:
        Tensor: 用于排序损失的主 logit 张量。

    Raises:
        KeyError: 当预测结果中既不存在 `pose_rank_score` 也不存在 `pose_quality` 时抛出。
    """
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
    """
    注入损失上下文。

    将课程学习和训练阶段相关字段写入 batch 对象，
    便于损失函数在不依赖外部状态的情况下读取上下文。

    Args:
        batch_obj: 当前批次图对象。
        current_epoch: 当前训练轮次。
        total_epochs_count: 训练总轮数。
        warmup_epochs_count: 课程预热轮数。
        training: 当前是否处于训练模式。
    """
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
    crop_min_residues: int,
    crop_atom_margin: float,
    graph_builder: Any,
    collator: GraphCollator,
) -> Any:
    """
    构造局部裁剪 batch。

    围绕给定中心裁剪样本并重新拼接成局部 batch，
    供局部对接训练和 bootstrap 流程复用。

    Args:
        batch_obj: 当前批次图对象。
        centers: 待用于裁剪或评估的中心坐标集合。
        crop_radius: 局部裁剪半径。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。

    Returns:
        Any: 由局部裁剪样本拼接得到的局部裁剪 batch。

    Raises:
        RuntimeError: 当局部裁剪后没有得到任何有效样本时抛出。
    """
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
            min_residues=crop_min_residues,
            atom_margin=crop_atom_margin,
            graph_builder=graph_builder,
        )
        for i, sample in enumerate(samples)
    ]
    return collator.collate(cropped_samples)
