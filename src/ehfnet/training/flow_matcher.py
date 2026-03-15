"""
流匹配控制器。

负责时间采样、目标构造和 ODE 积分，
连接训练监督过程与推理轨迹生成。
"""


import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from ehfnet.geometry import (
    PathInterpolator,
    PhysicsConstants,
    PoseUpdater,
    TangentTargetProjector,
    compute_center_of_mass,
)


logger = logging.getLogger(__name__)


class ConditionalFlowMatcher:
    """
    流匹配控制器。

    训练时：基于 PathInterpolator 生成物理插值路径（Kabsch + Torsion）
    推理时：基于 PoseUpdater 进行刚体 ODE 演化（支持 Euler/RK4）
    """

    def __init__(
        self,
        sigma_min: float,
        spatial_sigma_min: float,
        *,
        spatial_sigma_max: float,
        warmup_epochs: int,
        fd_dt: float,
        rotation_angle_min: float,
        rotation_angle_max: float,
        torsion_scale_min: float,
        torsion_scale_max: float,
    ) -> None:
        """
        初始化流匹配控制器。

        Args:
            sigma_min: 时间采样时使用的最小噪声水平。
            spatial_sigma_min: 平移扰动课程的最小尺度。
            spatial_sigma_max: 平移扰动课程的最大尺度。
            warmup_epochs: 课程学习预热轮数。
            fd_dt: 有限差分计算目标速度时使用的时间步长。
            rotation_angle_min: 课程初期允许的最大旋转角。
            rotation_angle_max: 课程后期允许的最大旋转角。
            torsion_scale_min: 课程初期的扭转扰动缩放系数。
            torsion_scale_max: 课程后期的扭转扰动缩放系数。
        """

        self.sigma_min = sigma_min
        self.fd_dt = fd_dt

        self.spatial_sigma_min = spatial_sigma_min
        self.spatial_sigma_max = spatial_sigma_max
        self.warmup_epochs = warmup_epochs
        self.rotation_angle_min = rotation_angle_min
        self.rotation_angle_max = rotation_angle_max
        self.torsion_scale_min = torsion_scale_min
        self.torsion_scale_max = torsion_scale_max

        self.interpolator = PathInterpolator(eps=PhysicsConstants.EPSILON, fd_dt=fd_dt)
        self.updater = PoseUpdater(eps=PhysicsConstants.EPSILON)
        self.target_projector = TangentTargetProjector(eps=PhysicsConstants.EPSILON)


    def get_curriculum_ratio(self, epoch: int) -> float:
        """
        统一的课程学习进度比例 [0.0, 1.0]。

        Args:
            epoch: 当前训练轮次。

        Returns:
            float: 返回当前 epoch 对应的课程学习进度比例。
        """
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)


    def get_spatial_scale(self, epoch: int) -> float:
        """
        根据当前 epoch 计算平移扰动尺度。

        Args:
            epoch: 当前训练轮次。

        Returns:
            float: 返回当前 epoch 使用的平移扰动尺度。
        """
        ratio = self.get_curriculum_ratio(epoch)
        return self.spatial_sigma_min + (self.spatial_sigma_max - self.spatial_sigma_min) * ratio


    def get_rotation_scale(self, epoch: int) -> float:
        """
        根据当前 epoch 计算最大旋转角。

        Args:
            epoch: 当前训练轮次。

        Returns:
            float: 返回当前 epoch 使用的最大旋转角尺度。
        """
        ratio = self.get_curriculum_ratio(epoch)
        return self.rotation_angle_min + (self.rotation_angle_max - self.rotation_angle_min) * ratio


    def get_torsion_scale(self, epoch: int) -> float:
        """
        根据当前 epoch 计算扭转角缩放系数。

        Args:
            epoch: 当前训练轮次。

        Returns:
            float: 返回当前 epoch 使用的扭转扰动缩放系数。
        """
        ratio = self.get_curriculum_ratio(epoch)
        return self.torsion_scale_min + (self.torsion_scale_max - self.torsion_scale_min) * ratio


    def sample_location_and_target(
        self,
        *,
        x_1: Tensor,
        data: HeteroData,
        x_0: Tensor | None = None,
        current_epoch: int,
        total_epochs: int,
        placement_centers: Tensor | None = None,
        t_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        [训练接口] 采样时间 t，构造插值坐标 x_t 和 SE(3) x T^m 目标字典

        Args:
            x_1: 目标构象坐标。
            data: 当前处理的图数据对象。
            x_0: 初始构象坐标。
            current_epoch: 当前训练轮次。
            total_epochs: 训练总轮数。
            placement_centers: 指定初始放置或 bootstrap 使用的中心集合。
            t_override: 可选的时间步覆盖值。

        Returns:
            t: 采样的时间步 [B]
            x_t: t 时刻的插值坐标 [N, 3]（模型输入）
            targets: SE(3) x T^m 切向量目标字典，包含:
                - v_trans_target [B, 3]
                - v_rot_target   [B, 3]
                - v_torsion_target [T]
        """

        batch = data["ligand_atom"].batch
        masses = data["ligand_atom"].masses
        device = x_1.device

        torsion_indices = getattr(
            data, "torsion_indices", torch.empty((0, 4), dtype=torch.long, device=device)
        )
        torsion_moving_mask = getattr(
            data,
            "torsion_moving_mask",
            torch.empty((0, x_1.size(0)), dtype=torch.bool, device=device),
        )

        if batch is not None and batch.numel() > 0:
            B = int(batch.max().item()) + 1
        else:
            B = 1

        if x_0 is None:
            seed_pos = data["ligand_atom"].get("start_pos", None)
            x_0 = self._generate_random_pose(
                x_ref=x_1,
                batch=batch,
                B=B,
                masses=masses,
                torsion_indices=torsion_indices,
                torsion_moving_mask=torsion_moving_mask,
                seed_pos=seed_pos,
                protein_pos=data["protein_atom"].pos if "protein_atom" in data else None,
                protein_batch=getattr(data["protein_atom"], "batch", None) if "protein_atom" in data else None,
                placement_centers=placement_centers,
                epoch=current_epoch,
            )

        if t_override is not None:
            t = t_override.to(device=device, dtype=x_1.dtype)
        else:
            sigma = float(self.sigma_min)
            sigma = max(0.0, min(0.49, sigma))
            t = torch.rand(B, device=device) * (1.0 - 2 * sigma) + sigma

        path_params = self.interpolator.compute_path_parameters(
            pos_0=x_0,
            pos_1=x_1,
            masses=masses,
            batch=batch,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
        )

        try:
            x_t, cartesian_velocity = self.interpolator.interpolate(path_params, t)
        except Exception as e:
            logger.error(f"Error during interpolation: {e}")
            x_t = x_0
            cartesian_velocity = torch.zeros_like(x_0)

        v_trans, v_rot, v_torsion = self.target_projector.decompose(
            pos=x_t,
            vel=cartesian_velocity,
            masses=masses,
            batch=batch,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
        )

        targets: dict[str, Tensor] = {
            "v_trans_target": v_trans,
            "v_rot_target": v_rot,
            "v_torsion_target": v_torsion,
        }

        return t, x_t, targets


    @torch.no_grad()
    def ode_solve(
        self,
        *,
        model: torch.nn.Module,
        data: HeteroData,
        steps: int,
        inference_t_start: float = 0.0,
        method: str = "rk4",
        store_trajectory: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """
        [推理接口] 生成轨迹

        Args:
            model: 当前使用的模型实例。
            data: 当前处理的图数据对象。
            steps: ODE 积分步数。
            inference_t_start: 推理积分的起始时间点。
            method: 轨迹积分方法，支持 `euler` 或 `rk4`。
            store_trajectory: 是否记录并返回每一步的中间坐标轨迹。

        Returns:
            tuple[Tensor, list[Tensor] | None]: 返回最终生成的坐标，以及可选的中间轨迹列表。
        """

        device = data["ligand_atom"].pos.device
        dtype = data["ligand_atom"].pos.dtype

        current_pos = data["ligand_atom"].pos.clone()
        masses = data["ligand_atom"].masses
        batch = data["ligand_atom"].batch

        if batch is None:
            batch = torch.zeros(current_pos.size(0), dtype=torch.long, device=device)

        torsion_indices = getattr(data, "torsion_indices", None)
        torsion_moving_mask = getattr(data, "torsion_moving_mask", None)

        if torsion_indices is not None and torsion_indices.numel() == 0:
            torsion_indices = None
            torsion_moving_mask = None

        trajectory = [current_pos.clone()] if store_trajectory else None
        dt = (1.0 - inference_t_start) / steps
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if B == 0:
            return current_pos, trajectory

        with torch.inference_mode():
            for i in range(steps):
                t_val = inference_t_start + i * dt

                if method == "euler":
                    v_trans, v_rot, v_torsion = self._predict_velocity(
                        model=model,
                        data=data,
                        pos=current_pos,
                        t_val=t_val,
                        B=B,
                        device=device,
                        dtype=dtype,
                    )

                    current_pos = self.updater.update(
                        pos=current_pos,
                        masses=masses,
                        batch=batch,
                        v_trans=v_trans,
                        v_rot=v_rot,
                        v_torsion=v_torsion,
                        torsion_indices=torsion_indices,
                        torsion_moving_mask=torsion_moving_mask,
                        dt=dt,
                    )

                elif method == "rk4":
                    v1_trans, v1_rot, v1_tor = self._predict_velocity(
                        model=model,
                        data=data,
                        pos=current_pos,
                        t_val=t_val,
                        B=B,
                        device=device,
                        dtype=dtype,
                    )
                    k1_pos = self.updater.update(
                        pos=current_pos,
                        masses=masses,
                        batch=batch,
                        v_trans=v1_trans,
                        v_rot=v1_rot,
                        v_torsion=v1_tor,
                        torsion_indices=torsion_indices,
                        torsion_moving_mask=torsion_moving_mask,
                        dt=dt / 2.0,
                    )

                    v2_trans, v2_rot, v2_tor = self._predict_velocity(
                        model=model,
                        data=data,
                        pos=k1_pos,
                        t_val=t_val + dt / 2.0,
                        B=B,
                        device=device,
                        dtype=dtype,
                    )
                    k2_pos = self.updater.update(
                        pos=current_pos,
                        masses=masses,
                        batch=batch,
                        v_trans=v2_trans,
                        v_rot=v2_rot,
                        v_torsion=v2_tor,
                        torsion_indices=torsion_indices,
                        torsion_moving_mask=torsion_moving_mask,
                        dt=dt / 2.0,
                    )

                    v3_trans, v3_rot, v3_tor = self._predict_velocity(
                        model=model,
                        data=data,
                        pos=k2_pos,
                        t_val=t_val + dt / 2.0,
                        B=B,
                        device=device,
                        dtype=dtype,
                    )
                    k3_pos = self.updater.update(
                        pos=current_pos,
                        masses=masses,
                        batch=batch,
                        v_trans=v3_trans,
                        v_rot=v3_rot,
                        v_torsion=v3_tor,
                        torsion_indices=torsion_indices,
                        torsion_moving_mask=torsion_moving_mask,
                        dt=dt,
                    )

                    v4_trans, v4_rot, v4_tor = self._predict_velocity(
                        model=model,
                        data=data,
                        pos=k3_pos,
                        t_val=t_val + dt,
                        B=B,
                        device=device,
                        dtype=dtype,
                    )

                    v_trans_avg = (v1_trans + 2 * v2_trans + 2 * v3_trans + v4_trans) / 6.0
                    v_rot_avg = (v1_rot + 2 * v2_rot + 2 * v3_rot + v4_rot) / 6.0

                    if v1_tor is not None:
                        v_tor_avg = (v1_tor + 2 * v2_tor + 2 * v3_tor + v4_tor) / 6.0
                    else:
                        v_tor_avg = None

                    current_pos = self.updater.update(
                        pos=current_pos,
                        masses=masses,
                        batch=batch,
                        v_trans=v_trans_avg,
                        v_rot=v_rot_avg,
                        v_torsion=v_tor_avg,
                        torsion_indices=torsion_indices,
                        torsion_moving_mask=torsion_moving_mask,
                        dt=dt,
                    )

                if trajectory is not None:
                    trajectory.append(current_pos.clone())

        return current_pos, trajectory


    def _predict_velocity(
        self,
        *,
        model: torch.nn.Module,
        data: HeteroData,
        pos: Tensor,
        t_val: float,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        辅助函数：运行模型一次，获取分解后的速度分量

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: 返回模型预测的平移、旋转和扭转速度分量。
        """
        original_pos = data["ligand_atom"].pos
        data["ligand_atom"].pos = pos
        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

        try:
            out = model(data, t_tensor)
        finally:
            data["ligand_atom"].pos = original_pos

        v_trans = out["v_translation"]
        v_rot = out["v_rotation"]
        v_torsion = out.get("v_torsion", None)

        if v_torsion is not None:
            v_torsion = v_torsion.reshape(-1)

        return v_trans, v_rot, v_torsion


    def _generate_random_pose(
        self,
        *,
        x_ref: Tensor,
        batch: Tensor,
        B: int,
        masses: Tensor,
        torsion_indices: Tensor | None,
        torsion_moving_mask: Tensor | None,
        seed_pos: Tensor | None = None,
        protein_pos: Tensor | None = None,
        protein_batch: Tensor | None = None,
        placement_centers: Tensor | None = None,
        epoch: int = 0,
    ) -> Tensor:
        """
        生成随机初始位姿（联合三自由度课程学习）

        根据 epoch 进度同步控制平移尺度、旋转角度、扭转角范围。

        Returns:
            Tensor: 返回计算得到的张量结果。
        """

        device = x_ref.device
        dtype = x_ref.dtype

        current_trans_scale = self.get_spatial_scale(epoch)
        current_rot_max = self.get_rotation_scale(epoch)
        current_tor_scale = self.get_torsion_scale(epoch)

        base_pos = x_ref
        if seed_pos is not None and seed_pos.shape == x_ref.shape:
            base_pos = seed_pos.to(device=device, dtype=dtype)

        x_torsioned = base_pos.clone()

        if torsion_indices is not None and torsion_moving_mask is not None:
            T = torsion_indices.shape[0]

            if T > 0:
                rand_angles = (
                    (torch.rand(T, device=device, dtype=dtype) * 2 - 1)
                    * math.pi * current_tor_scale
                )

                x_torsioned = self.updater.update(
                    pos=x_torsioned,
                    masses=masses,
                    batch=batch,
                    v_trans=torch.zeros(B, 3, device=device, dtype=dtype),
                    v_rot=torch.zeros(B, 3, device=device, dtype=dtype),
                    v_torsion=rand_angles,
                    torsion_indices=torsion_indices,
                    torsion_moving_mask=torsion_moving_mask,
                    dt=1.0,
                )

        lig_center = compute_center_of_mass(x_torsioned, batch, masses, dim_size=B)
        x_centered = x_torsioned - lig_center[batch]

        if placement_centers is not None and placement_centers.shape == (B, 3):
            placement_center = placement_centers.to(device=device, dtype=dtype)
        elif protein_pos is not None:
            if protein_batch is None:
                protein_batch = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)
            placement_center = scatter_mean(protein_pos.to(device=device, dtype=dtype), protein_batch, dim=0, dim_size=B)
        else:
            placement_center = compute_center_of_mass(x_ref, batch, masses, dim_size=B)

        rot_matrices = self._random_bounded_rotation(B, device, dtype, max_angle=current_rot_max)
        x_rotated = torch.bmm(rot_matrices[batch], x_centered.unsqueeze(-1)).squeeze(-1)

        translation_offset = torch.randn(B, 3, device=device, dtype=dtype) * current_trans_scale

        return x_rotated + placement_center[batch] + translation_offset[batch]


    @staticmethod
    def _random_bounded_rotation(
        B: int, device: torch.device, dtype: torch.dtype, max_angle: float,
    ) -> Tensor:
        """
        在 SO(3) 上采样旋转角不超过 max_angle 的随机旋转

        Args:
            B: 批次大小
            device: 设备
            dtype: 数据类型
            max_angle: 最大旋转角（弧度）

        Returns:
            旋转矩阵 [B, 3, 3]
        """
        axis = F.normalize(torch.randn(B, 3, device=device, dtype=dtype), dim=-1)
        angle = torch.rand(B, device=device, dtype=dtype) * max_angle
        return PoseUpdater._axis_angle_to_matrix_batched(axis, angle)
