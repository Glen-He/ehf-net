"""
训练损失模块。

负责计算流匹配训练中的各项损失，
并管理课程权重与时间门控逻辑。
"""


import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class FlowMatchingLoss(nn.Module):
    """
    流匹配损失函数。

    直接对平移、旋转、扭转的切向量进行监督。
    包含时间步 t 掩码机制，确保物理头仅在有效范围内优化。
    """

    def __init__(
        self,
        characteristic_scale: float,
        weight_translation: float,
        *,
        weight_rotation: float,
        weight_torsion: float,
        weight_energy: float,
        weight_clash: float,
        weight_pose_rank: float,
        curriculum_weights: dict[str, dict[str, float]],
        refine_start: float,
        pose_gate_epoch_start: float,
        pose_gate_epoch_end: float,
        pose_gate_tau_start: float,
        pose_gate_tau_end: float,
        pose_gate_temperature: float,
    ) -> None:
        """
        初始化流匹配损失函数。

        Args:
            characteristic_scale: 平衡平移与旋转量纲的特征长度尺度。
            weight_translation: 平移损失权重。
            weight_rotation: 旋转损失权重。
            weight_torsion: 扭转损失权重。
            weight_energy: 亲和力损失权重。
            weight_clash: 位阻损失权重。
            weight_pose_rank: 构象排序损失权重。
            curriculum_weights: 课程学习各阶段的损失权重配置。
            refine_start: 进入细化阶段时对应的训练进度阈值。
            pose_gate_epoch_start: 构象相关损失开始打开门控的训练进度。
            pose_gate_epoch_end: 构象相关损失完全打开门控的训练进度。
            pose_gate_tau_start: 构象门控在初期使用的时间阈值。
            pose_gate_tau_end: 构象门控在后期使用的时间阈值。
            pose_gate_temperature: 构象时间门控的温度系数。
        """
        super().__init__()

        self.L = characteristic_scale

        self.weights = {
            "translation": weight_translation,
            "rotation": weight_rotation,
            "torsion": weight_torsion,
            "energy": weight_energy,
            "clash": weight_clash,
            "pose_rank": weight_pose_rank,
        }

        self.curriculum_weights = curriculum_weights
        self.refine_start = float(refine_start)
        self.pose_gate_epoch_start = float(pose_gate_epoch_start)
        self.pose_gate_epoch_end = float(pose_gate_epoch_end)
        self.pose_gate_tau_start = float(pose_gate_tau_start)
        self.pose_gate_tau_end = float(pose_gate_tau_end)
        self.pose_gate_temperature = float(pose_gate_temperature)
        self._energy_nan_warn_count = 0

    @staticmethod
    def _clamp_progress(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _lerp_dict(
        start: dict[str, float],
        end: dict[str, float],
        *,
        alpha: float,
    ) -> dict[str, float]:
        """
        按 alpha 在 start 与 end 之间线性插值，键集合以 start 为准。

        Args:
            start: 起始权重字典。
            end: 终止权重字典。
            alpha: 插值系数，[0, 1] 外会被截断。

        Returns:
            dict[str, float]: 插值后的权重字典。
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        return {
            key: float(start[key] + (end[key] - start[key]) * alpha)
            for key in start
        }

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, *, value: float) -> float:
        if edge1 <= edge0:
            return 1.0 if value >= edge1 else 0.0
        t = (value - edge0) / (edge1 - edge0)
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _get_loss_schedule(self, data: Any) -> dict[str, float]:
        progress = self._clamp_progress(getattr(data, "loss_progress", 0.0))
        warmup_end = self._clamp_progress(getattr(data, "loss_warmup_end", 0.2))
        refine_start = self._clamp_progress(self.refine_start)

        if progress <= warmup_end:
            alpha = 1.0 if warmup_end <= 1e-8 else progress / max(warmup_end, 1e-8)
            schedule = self._lerp_dict(
                self.curriculum_weights["coarse"],
                self.curriculum_weights["transition"],
                alpha=alpha,
            )
        elif progress <= refine_start:
            schedule = dict(self.curriculum_weights["transition"])
        else:
            alpha = (progress - refine_start) / max(1.0 - refine_start, 1e-8)
            schedule = self._lerp_dict(
                self.curriculum_weights["transition"],
                self.curriculum_weights["refine"],
                alpha=alpha,
            )

        return {k: schedule[k] * self.weights.get(k, 1.0) for k in schedule}

    def _get_pose_focus_gate(self, data: Any, t_val: Tensor | None, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        progress = self._clamp_progress(getattr(data, "loss_progress", 0.0))
        epoch_gate = self._smoothstep(
            self.pose_gate_epoch_start,
            self.pose_gate_epoch_end,
            value=progress,
        )

        if t_val is None:
            return torch.tensor(epoch_gate, device=device, dtype=dtype)

        tau = self.pose_gate_tau_start + (
            self.pose_gate_tau_end - self.pose_gate_tau_start
        ) * progress
        pose_gate = torch.sigmoid(
            (t_val.to(dtype=dtype) - tau)
            / max(self.pose_gate_temperature, 1e-6)
        )
        return pose_gate * epoch_gate

    def forward(
        self,
        predictions: dict[str, Tensor | None],
        targets: dict[str, Tensor | None],
        data: Any,
    ) -> dict[str, Tensor]:
        """
        计算损失。

        Args:
            predictions: 模型前向传播返回的预测结果字典。
            targets: 与预测字段对齐的监督目标张量字典。
            data: 当前处理的图数据对象。

        Returns:
            dict[str, Tensor]: 返回各项损失分量及总损失组成的结果字典。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """

        loss_dict: dict[str, Tensor] = {}

        pred_translation = predictions.get("v_translation")

        if pred_translation is None:
            raise ValueError("Key 'v_translation' is missing in predictions.")

        device = pred_translation.device
        schedule = self._get_loss_schedule(data)

        target_translation = targets.get("v_translation_target")

        if target_translation is None:
            raise ValueError("Key 'v_translation_target' is missing in targets.")

        loss_translation = F.huber_loss(
            pred_translation,
            target_translation,
            delta=1.0,
        )
        loss_dict["loss_translation"] = loss_translation.detach()
        loss_dict["_raw_loss_translation"] = loss_translation

        pred_rotation = predictions.get("v_rotation")
        target_rotation = targets.get("v_rotation_target")

        if pred_rotation is None or target_rotation is None:
            raise ValueError("Missing rotation data in predictions or targets.")

        loss_rotation = F.huber_loss(
            pred_rotation * self.L,
            target_rotation * self.L,
            delta=1.0,
        )
        loss_dict["loss_rotation"] = loss_rotation.detach()
        loss_dict["_raw_loss_rotation"] = loss_rotation

        loss_torsion = torch.tensor(0.0, device=device)
        pred_torsion = predictions.get("v_torsion")
        target_torsion = targets.get("v_torsion_target")

        if (
            pred_torsion is not None
            and target_torsion is not None
            and target_torsion.numel() > 0
        ):

            if pred_torsion.dim() == 1:
                pred_torsion = pred_torsion.view(-1, 1)

            if target_torsion.dim() == 1:
                target_torsion = target_torsion.view(-1, 1)

            cos_diff = 1.0 - torch.cos(pred_torsion - target_torsion)
            loss_torsion = torch.mean(cos_diff) * (self.L / 2.0)

        if torch.isnan(loss_torsion.detach()):
            loss_torsion = torch.tensor(0.0, device=device)

        loss_dict["loss_torsion"] = loss_torsion.detach()
        loss_dict["_raw_loss_torsion"] = loss_torsion

        loss_energy = torch.tensor(0.0, device=device)
        energy_nan_skipped = torch.tensor(0.0, device=device)
        pred_affinity = predictions.get("binding_affinity")
        gt_affinity = targets.get("binding_affinity_target")

        if pred_affinity is not None and gt_affinity is not None:

            if not torch.isfinite(pred_affinity).all() or not torch.isfinite(gt_affinity).all():
                energy_nan_skipped = torch.tensor(1.0, device=device)
                self._energy_nan_warn_count += 1
                if self._energy_nan_warn_count <= 3 or self._energy_nan_warn_count % 100 == 0:
                    logger.warning(
                        "Skipping energy loss due to non-finite affinity values (count=%d).",
                        self._energy_nan_warn_count,
                    )

            else:
                t_val = getattr(data, "t", None)
                pose_focus_gate = self._get_pose_focus_gate(
                    data,
                    t_val,
                    device=device,
                    dtype=pred_affinity.dtype,
                )

                if t_val is not None:
                    per_sample_energy = F.huber_loss(
                        pred_affinity.view(-1),
                        gt_affinity.view(-1),
                        delta=2.0,
                        reduction="none",
                    )
                    gate_sum = pose_focus_gate.sum()
                    if gate_sum > 1e-8:
                        loss_energy = (per_sample_energy * pose_focus_gate).sum() / gate_sum
                else:
                    loss_energy = F.huber_loss(
                        pred_affinity.view(-1), gt_affinity.view(-1), delta=2.0
                    )

        loss_dict["loss_energy"] = loss_energy.detach()
        loss_dict["energy_nan_skipped"] = energy_nan_skipped.detach()
        loss_dict["_raw_loss_energy"] = loss_energy

        loss_clash = torch.tensor(0.0, device=device)
        clash_batch = predictions.get("steric_clash_batch")

        if clash_batch is not None and not torch.isnan(clash_batch).any():
            t_val = getattr(data, "t", None)
            pose_focus_gate = self._get_pose_focus_gate(
                data,
                t_val,
                device=device,
                dtype=clash_batch.dtype,
            )
            gate_sum = pose_focus_gate.sum() if pose_focus_gate.ndim > 0 else pose_focus_gate
            if torch.is_tensor(gate_sum):
                if gate_sum > 1e-8:
                    loss_clash = (clash_batch.view(-1) * pose_focus_gate.view(-1)).sum() / gate_sum
            elif gate_sum > 1e-8:
                loss_clash = clash_batch.mean() * float(gate_sum)

        loss_dict["loss_clash"] = loss_clash.detach()
        loss_dict["_raw_loss_clash"] = loss_clash
        loss_pose_rank = torch.tensor(0.0, device=device)
        pred_pose_rank = predictions.get("pose_rank_score")
        gt_pose_rank = targets.get("pose_rank_target")

        if pred_pose_rank is not None and gt_pose_rank is not None:
            pred_pose_rank = pred_pose_rank.view(-1)
            gt_pose_rank = gt_pose_rank.view(-1).to(device=device, dtype=pred_pose_rank.dtype)
            if not torch.isnan(pred_pose_rank).any() and not torch.isnan(gt_pose_rank).any():
                weight = 1.0 + 2.0 * gt_pose_rank
                per_sample_bce = F.binary_cross_entropy_with_logits(
                    pred_pose_rank,
                    gt_pose_rank.clamp(min=0.0, max=1.0),
                    reduction="none",
                )
                t_val = getattr(data, "t", None)
                if t_val is not None:
                    pose_focus_gate = self._get_pose_focus_gate(
                        data,
                        t_val,
                        device=device,
                        dtype=pred_pose_rank.dtype,
                    ).view(-1)
                    gate_sum = pose_focus_gate.sum()
                    if gate_sum > 1e-8:
                        loss_pose_rank = (
                            per_sample_bce * weight * pose_focus_gate
                        ).sum() / gate_sum
                else:
                    loss_pose_rank = (per_sample_bce * weight).mean()

        loss_dict["loss_pose_rank_bce"] = loss_pose_rank.detach()
        loss_dict["weight_translation"] = torch.tensor(
            schedule["translation"],
            device=device,
        )
        loss_dict["weight_rotation"] = torch.tensor(
            schedule["rotation"],
            device=device,
        )
        loss_dict["weight_torsion"] = torch.tensor(schedule["torsion"], device=device)
        loss_dict["weight_energy"] = torch.tensor(schedule["energy"], device=device)
        loss_dict["weight_clash"] = torch.tensor(schedule["clash"], device=device)
        loss_dict["weight_pose_rank"] = torch.tensor(schedule["pose_rank"], device=device)
        loss_dict["_raw_loss_pose_rank_bce"] = loss_pose_rank

        total_loss = (
            schedule["translation"] * loss_translation
            + schedule["rotation"] * loss_rotation
            + schedule["torsion"] * loss_torsion
            + schedule["energy"] * loss_energy
            + schedule["clash"] * loss_clash
            + schedule["pose_rank"] * loss_pose_rank
        )

        loss_dict["total"] = total_loss

        return loss_dict
