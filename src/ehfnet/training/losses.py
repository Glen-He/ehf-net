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
        weight_trans: float,
        *,
        weight_rot: float,
        weight_torsion: float,
        weight_energy: float,
        weight_clash: float,
        weight_pose_quality: float,
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
            weight_trans: 平移损失权重。
            weight_rot: 旋转损失权重。
            weight_torsion: 扭转损失权重。
            weight_energy: 亲和力损失权重。
            weight_clash: 位阻损失权重。
            weight_pose_quality: 构象质量损失权重。
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
            "trans": weight_trans,
            "rot": weight_rot,
            "torsion": weight_torsion,
            "energy": weight_energy,
            "clash": weight_clash,
            "pose_quality": weight_pose_quality,
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

        pred_trans = predictions.get("v_translation")

        if pred_trans is None:
            raise ValueError("Key 'v_translation' is missing in predictions.")

        device = pred_trans.device
        schedule = self._get_loss_schedule(data)

        gt_trans = targets.get("v_trans_target")

        if gt_trans is None:
            raise ValueError("Key 'v_trans_target' is missing in targets.")

        loss_trans = F.huber_loss(pred_trans, gt_trans, delta=1.0)
        loss_dict["loss_trans"] = loss_trans.detach()

        pred_rot = predictions.get("v_rotation")
        gt_rot = targets.get("v_rot_target")

        if pred_rot is None or gt_rot is None:
            raise ValueError("Missing rotation data in predictions or targets.")

        loss_rot = F.huber_loss(pred_rot * self.L, gt_rot * self.L, delta=1.0)
        loss_dict["loss_rot"] = loss_rot.detach()

        loss_torsion = torch.tensor(0.0, device=device)
        pred_tor = predictions.get("v_torsion")
        gt_tor = targets.get("v_torsion_target")

        if pred_tor is not None and gt_tor is not None and gt_tor.numel() > 0:

            if pred_tor.dim() == 1:
                pred_tor = pred_tor.view(-1, 1)

            if gt_tor.dim() == 1:
                gt_tor = gt_tor.view(-1, 1)

            cos_diff = 1.0 - torch.cos(pred_tor - gt_tor)
            loss_torsion = torch.mean(cos_diff) * (self.L / 2.0)

        if torch.isnan(loss_torsion.detach()):
            loss_torsion = torch.tensor(0.0, device=device)

        loss_dict["loss_torsion"] = loss_torsion.detach()

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
        loss_pose_quality = torch.tensor(0.0, device=device)
        pred_pose_quality = predictions.get("pose_quality")
        gt_pose_quality = targets.get("pose_quality_target")

        if pred_pose_quality is not None and gt_pose_quality is not None:
            pred_pose_quality = pred_pose_quality.view(-1)
            gt_pose_quality = gt_pose_quality.view(-1).to(device=device, dtype=pred_pose_quality.dtype)
            if not torch.isnan(pred_pose_quality).any() and not torch.isnan(gt_pose_quality).any():
                weight = 1.0 + 2.0 * gt_pose_quality
                per_sample_bce = F.binary_cross_entropy_with_logits(
                    pred_pose_quality,
                    gt_pose_quality.clamp(min=0.0, max=1.0),
                    reduction="none",
                )
                t_val = getattr(data, "t", None)
                if t_val is not None:
                    pose_focus_gate = self._get_pose_focus_gate(
                        data,
                        t_val,
                        device=device,
                        dtype=pred_pose_quality.dtype,
                    ).view(-1)
                    gate_sum = pose_focus_gate.sum()
                    if gate_sum > 1e-8:
                        loss_pose_quality = (
                            per_sample_bce * weight * pose_focus_gate
                        ).sum() / gate_sum
                else:
                    loss_pose_quality = (per_sample_bce * weight).mean()

        loss_dict["loss_pose_quality"] = loss_pose_quality.detach()
        loss_dict["weight_trans"] = torch.tensor(schedule["trans"], device=device)
        loss_dict["weight_rot"] = torch.tensor(schedule["rot"], device=device)
        loss_dict["weight_torsion"] = torch.tensor(schedule["torsion"], device=device)
        loss_dict["weight_energy"] = torch.tensor(schedule["energy"], device=device)
        loss_dict["weight_clash"] = torch.tensor(schedule["clash"], device=device)
        loss_dict["weight_pose_quality"] = torch.tensor(schedule["pose_quality"], device=device)

        total_loss = (
            schedule["trans"] * loss_trans
            + schedule["rot"] * loss_rot
            + schedule["torsion"] * loss_torsion
            + schedule["energy"] * loss_energy
            + schedule["clash"] * loss_clash
            + schedule["pose_quality"] * loss_pose_quality
        )

        loss_dict["total"] = total_loss

        return loss_dict
