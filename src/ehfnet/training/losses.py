"""
流匹配损失函数

移除同方差不确定性加权，采用静态几何尺度平衡。
直接在 SE(3) x T^m 切空间计算 Huber Loss。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from torch import Tensor


class FlowMatchingLoss(nn.Module):
    """
    流匹配损失函数

    直接对平移、旋转、扭转的切向量进行监督。
    包含时间步 t 掩码机制，确保物理头仅在有效范围内优化。
    """

    def __init__(
        self,
        characteristic_scale: float = 5.0,
        weight_trans: float = 1.0,
        weight_rot: float = 1.0,
        weight_torsion: float = 0.2,
        weight_energy: float = 0.05,
        weight_clash: float = 0.001,  # 初始极小权重：clash_batch 量级 O(10²)，须防止压垮 SE(3) 损失
    ) -> None:
        """
        初始化损失函数。

        Args:
            characteristic_scale: 特征长度尺度 L (Å)，用于平衡旋转和平移的量纲。
            weight_trans: 平移损失权重。
            weight_rot: 旋转损失权重。
            weight_torsion: 扭转角损失权重。
            weight_energy: 结合能损失权重。
            weight_clash: 位阻惩罚权重，初始极小防止压垮 SE(3) 损失。
        """
        super().__init__()

        self.L = characteristic_scale

        self.weights = {
            "trans": weight_trans,
            "rot": weight_rot,
            "torsion": weight_torsion,
            "energy": weight_energy,
            "clash": weight_clash,
        }

    def forward(
        self,
        predictions: dict[str, Tensor | None],
        targets: dict[str, Tensor | None],
        data: Any,
    ) -> dict[str, Tensor]:
        """
        计算损失。

        Args:
            predictions: 预测结果字典。
            targets: 目标值字典。
            data: 数据批次对象。

        Returns:
            损失字典。
        """
        loss_dict: dict[str, Tensor] = {}

        pred_trans = predictions.get("v_translation")
        if pred_trans is None:
            raise ValueError("Key 'v_translation' is missing in predictions.")
        device = pred_trans.device

        # 1. 平移损失
        gt_trans = targets.get("v_trans_target")
        if gt_trans is None:
            raise ValueError("Key 'v_trans_target' is missing in targets.")

        loss_trans = F.huber_loss(pred_trans, gt_trans, delta=1.0)
        loss_dict["loss_trans"] = loss_trans.detach()

        # 2. 旋转损失
        pred_rot = predictions.get("v_rotation")
        gt_rot = targets.get("v_rot_target")
        if pred_rot is None or gt_rot is None:
            raise ValueError("Missing rotation data in predictions or targets.")

        loss_rot = F.huber_loss(pred_rot * self.L, gt_rot * self.L, delta=1.0)
        loss_dict["loss_rot"] = loss_rot.detach()

        # 3. 扭转损失（周期性余弦损失）
        loss_torsion = torch.tensor(0.0, device=device)
        pred_tor = predictions.get("v_torsion")
        gt_tor = targets.get("v_torsion_target")

        if pred_tor is not None and gt_tor is not None and gt_tor.numel() > 0:
            if pred_tor.dim() == 1:
                pred_tor = pred_tor.view(-1, 1)
            if gt_tor.dim() == 1:
                gt_tor = gt_tor.view(-1, 1)

            # 物理周期性损失：1 - cos(pred - target)
            # 自动处理 -π/+π 等价性，输出严格有界 [0, 2]，杜绝梯度爆炸
            cos_diff = 1.0 - torch.cos(pred_tor - gt_tor)
            loss_torsion = torch.mean(cos_diff) * (self.L / 2.0)

        # NaN 守卫（理论上 cos_diff 有界，但防御性保留）
        if torch.isnan(loss_torsion.detach()):
            loss_torsion = torch.tensor(0.0, device=device)

        loss_dict["loss_torsion"] = loss_torsion.detach()

        # 4. 物理亲和力损失 (带时间掩码)
        loss_energy = torch.tensor(0.0, device=device)
        pred_affinity = predictions.get("binding_affinity")
        gt_affinity = targets.get("binding_affinity_target")

        if pred_affinity is not None and gt_affinity is not None:
            # NaN 守卫：预测头在训练初期可能因权重随机而输出 NaN，直接跳过避免污染梯度
            if torch.isnan(pred_affinity).any() or torch.isnan(gt_affinity).any():
                pass
            else:
                t_val = getattr(data, "t", None)

                if t_val is not None:
                    # 仅在 t > 0.5 时（配体已较为靠近真实结合态）计算亲和力损失
                    valid_mask = t_val > 0.5
                    if valid_mask.any():
                        loss_energy = F.huber_loss(
                            pred_affinity[valid_mask].view(-1),
                            gt_affinity[valid_mask].view(-1),
                            delta=2.0,
                        )
                else:
                    # 兼容性回退：无时间步信息时直接计算
                    loss_energy = F.huber_loss(
                        pred_affinity.view(-1), gt_affinity.view(-1), delta=2.0
                    )

        loss_dict["loss_energy"] = loss_energy.detach()

        # 5. 物理位阻惩罚损失 (Time-Masked Steric Clash)
        # 仅在 t > 0.8（配体已进入口袋微调阶段）施加；早期大位移阶段允许穿模
        loss_clash = torch.tensor(0.0, device=device)
        clash_batch = predictions.get("steric_clash_batch")

        if clash_batch is not None and not torch.isnan(clash_batch).any():
            t_val = getattr(data, "t", None)
            if t_val is not None:
                valid_mask = t_val > 0.8
                if valid_mask.any():
                    loss_clash = clash_batch[valid_mask].mean()
            else:
                # data.t 未设置时跳过（避免无掩码全量施加位阻损失破坏早期训练）
                pass

        loss_dict["loss_clash"] = loss_clash.detach()

        # 6. 总损失
        total_loss = (
            self.weights["trans"] * loss_trans
            + self.weights["rot"] * loss_rot
            + self.weights["torsion"] * loss_torsion
            + self.weights["energy"] * loss_energy
            + self.weights["clash"] * loss_clash
        )

        loss_dict["total"] = total_loss

        return loss_dict
