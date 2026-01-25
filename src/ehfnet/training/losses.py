"""
流匹配损失函数

基于同方差不确定性的自动任务平衡。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from torch import Tensor

from ehfnet.geometry.dynamics import VelocityDecomposer, PhysicsConstants


class FlowMatchingLoss(nn.Module):
    """
    流匹配损失函数

    通过可学习的对数方差参数自动加权不同损失分量，避免手动调参。
    损失公式：L_total = sum(exp(-s_i) * L_i + 0.5 * s_i)
    其中 s_i = log(sigma_i^2) 为可学习参数，exp(-s_i) 为自动学到的权重。
    参考：Kendall 等，CVPR 2018。
    """

    def __init__(
        self,
        *,
        eps: float = PhysicsConstants.EPSILON,
        log_var_min: float = -5.0,
        log_var_max: float = 5.0,
    ) -> None:
        """
        Args:
            eps: 数值稳定性保护参数
            log_var_min: 对数方差最小值
            log_var_max: 对数方差最大值
        """

        super().__init__()
        self.decomposer = VelocityDecomposer(eps=eps)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # 可学习的对数方差 s = log(sigma^2)，初始化为 0（即初始权重=1）
        self.log_vars = nn.ParameterDict(
            {
                "trans": nn.Parameter(torch.zeros(1)),
                "rot": nn.Parameter(torch.zeros(1)),
                "torsion": nn.Parameter(torch.zeros(1)),
                "energy": nn.Parameter(torch.zeros(1)),
            }
        )


    def forward(
        self,
        predictions: dict[str, Tensor | None],
        targets: dict[str, Tensor | None],
        data: Any,
    ) -> dict[str, Tensor]:
        """
        前向传播

        Args:
            predictions: 模型预测结果
            targets: 目标值
            data: 数据对象

        Returns:
            损失字典
        """

        v_atomic = predictions["v_atomic"]
        
        if v_atomic is None:
            raise ValueError("predictions['v_atomic'] must not be None.")

        device = v_atomic.device
        loss_dict: dict[str, Tensor] = {}

        # 1. 分解目标速度
        v_target_atomic = targets["v_atomic_target"]

        if v_target_atomic is None:
            raise ValueError("targets['v_atomic_target'] must not be None.")
            
        masses = data["ligand_atom"].masses
        batch = data["ligand_atom"].batch

        torsion_indices = getattr(
            data, "torsion_indices", torch.empty((0, 4), dtype=torch.long, device=device)
        )
        torsion_moving_mask = getattr(
            data,
            "torsion_moving_mask",
            torch.empty((0, masses.size(0)), dtype=torch.bool, device=device),
        )

        pos_t = data["ligand_atom"].pos
        gt_trans, gt_rot, gt_torsion = self.decomposer.decompose(
            pos=pos_t,
            vel=v_target_atomic,
            masses=masses,
            batch=batch,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
        )

        # 2. 计算原始损失
        pred_trans = predictions["v_translation"]

        if pred_trans is None:
            raise ValueError("predictions['v_translation'] must not be None.")

        raw_loss_trans = F.mse_loss(pred_trans, gt_trans)
        loss_dict["raw_loss_trans"] = raw_loss_trans.detach()

        pred_rot = predictions["v_rotation"]

        if pred_rot is None:
            raise ValueError("predictions['v_rotation'] must not be None.")

        raw_loss_rot = F.mse_loss(pred_rot, gt_rot)
        loss_dict["raw_loss_rot"] = raw_loss_rot.detach()

        pred_torsion = predictions.get("v_torsion", None)

        if pred_torsion is not None and gt_torsion.numel() > 0:

            if pred_torsion.dim() == 1:
                pred_torsion = pred_torsion.unsqueeze(-1)

            if gt_torsion.dim() == 1:
                gt_torsion = gt_torsion.unsqueeze(-1)

            raw_loss_torsion = F.mse_loss(pred_torsion, gt_torsion)

        else:
            raw_loss_torsion = torch.tensor(0.0, device=device)

        loss_dict["raw_loss_torsion"] = raw_loss_torsion.detach()

        pred_affinity = predictions.get("binding_affinity", None)
        gt_affinity = targets.get("binding_affinity_target", None)

        if pred_affinity is not None and gt_affinity is not None:

            if pred_affinity.dim() == 1:
                pred_affinity = pred_affinity.unsqueeze(-1)

            if gt_affinity.dim() == 1:
                gt_affinity = gt_affinity.unsqueeze(-1)

            raw_loss_energy = F.mse_loss(pred_affinity, gt_affinity)

        else:
            raw_loss_energy = torch.tensor(0.0, device=device)

        loss_dict["raw_loss_energy"] = raw_loss_energy.detach()

        # 3. 不确定性加权
        total_loss = torch.zeros((), device=device)

        s_trans = torch.clamp(self.log_vars["trans"], self.log_var_min, self.log_var_max)
        loss_trans = torch.exp(-s_trans) * raw_loss_trans + 0.5 * s_trans
        total_loss += loss_trans

        s_rot = torch.clamp(self.log_vars["rot"], self.log_var_min, self.log_var_max)
        loss_rot = torch.exp(-s_rot) * raw_loss_rot + 0.5 * s_rot
        total_loss += loss_rot

        s_torsion = torch.clamp(self.log_vars["torsion"], self.log_var_min, self.log_var_max)

        if pred_torsion is not None and gt_torsion.numel() > 0:
            loss_torsion = torch.exp(-s_torsion) * raw_loss_torsion + 0.5 * s_torsion
            total_loss += loss_torsion

        s_energy = torch.clamp(self.log_vars["energy"], self.log_var_min, self.log_var_max)

        if pred_affinity is not None and gt_affinity is not None:
            loss_energy = torch.exp(-s_energy) * raw_loss_energy + 0.5 * s_energy
            total_loss += loss_energy

        # 记录学到的权重
        loss_dict["weight_trans"] = torch.exp(-s_trans).detach()
        loss_dict["weight_rot"] = torch.exp(-s_rot).detach()
        loss_dict["weight_torsion"] = torch.exp(-s_torsion).detach()
        loss_dict["weight_energy"] = torch.exp(-s_energy).detach()

        loss_dict["total"] = total_loss

        return loss_dict
