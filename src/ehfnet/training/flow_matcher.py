"""
流匹配控制器

训练时：基于 PathInterpolator 生成物理插值路径
推理时：基于 PoseUpdater 进行刚体 ODE 演化
"""

import torch
import logging

from torch import Tensor
from torch_geometric.data import HeteroData
from ehfnet.geometry.dynamics import PathInterpolator, PoseUpdater, PhysicsConstants, compute_center_of_mass, VelocityDecomposer


logger = logging.getLogger(__name__)


class ConditionalFlowMatcher:
    """
    流匹配控制器

    训练时：基于 PathInterpolator 生成物理插值路径（Kabsch + Torsion）
    推理时：基于 PoseUpdater 进行刚体 ODE 演化（支持 Euler/RK4）
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        spatial_sigma_min: float = 1.0,
        spatial_sigma_max: float = 6.0,
        warmup_epochs: int = 20,
        fd_dt: float = 0.05,
    ) -> None:
        """
        Args:
            sigma_min: 最小噪声水平，用于防止 t=1 时的数值不稳定
            spatial_sigma_min: 空间课程学习起始尺度 (Å)
            spatial_sigma_max: 空间课程学习结束尺度 (Å)
            warmup_epochs: 空间课程学习预热轮数
            fd_dt: 速度有限差分步长，透传至 PathInterpolator（v = Δpos / fd_dt）
        """

        self.sigma_min = sigma_min
        self.fd_dt = fd_dt

        # 课程学习参数
        self.spatial_sigma_min = spatial_sigma_min
        self.spatial_sigma_max = spatial_sigma_max
        self.warmup_epochs = warmup_epochs

        self.interpolator = PathInterpolator(eps=PhysicsConstants.EPSILON, fd_dt=fd_dt)
        self.updater = PoseUpdater(eps=PhysicsConstants.EPSILON)
        # [新增] 在流匹配器中直接持有分解器，训练时生成纯净的 SE(3) x T^m 目标
        self.decomposer = VelocityDecomposer(eps=PhysicsConstants.EPSILON)


    def get_spatial_scale(self, epoch: int) -> float:
        """
        根据当前 epoch 计算空间扰动尺度
        """
        if self.warmup_epochs <= 0:
            return self.spatial_sigma_max
            
        progress = min(1.0, epoch / self.warmup_epochs)
        return self.spatial_sigma_min + (self.spatial_sigma_max - self.spatial_sigma_min) * progress


    def sample_location_and_target(
        self,
        *,
        x_1: Tensor,
        data: HeteroData,
        x_0: Tensor | None = None,
        current_epoch: int = 0,
        total_epochs: int = 1,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        [训练接口] 采样时间 t，构造插值坐标 x_t 和 SE(3) x T^m 目标字典

        Args:
            x_1: 真实结合构象 [N, 3] (Ground Truth)
            data: 图数据对象（包含 batch, masses, torsion_indices 等）
            x_0: 初始构象（可选）。如果为 None，则基于 x_1 进行随机刚体变换生成
            current_epoch: 当前训练轮数 (用于课程学习)
            total_epochs: 总训练轮数 (用于课程学习)

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

        B = int(batch.max().item()) + 1 if batch is not None else 1

        # 1. 生成 x_0
        if x_0 is None:
            current_scale = self.get_spatial_scale(current_epoch)
            x_0 = self._generate_random_pose(
                x_ref=x_1,
                batch=batch,
                B=B,
                masses=masses,
                torsion_indices=torsion_indices,
                torsion_moving_mask=torsion_moving_mask,
                translation_scale=current_scale,
            )

        # 2. 采样时间 t，显式避开边界：t ~ U[sigma_min, 1-sigma_min]
        sigma = float(self.sigma_min)
        sigma = max(0.0, min(0.49, sigma))
        t = torch.rand(B, device=device) * (1.0 - 2 * sigma) + sigma

        # 3. 计算物理插值路径参数
        path_params = self.interpolator.compute_path_parameters(
            pos_0=x_0,
            pos_1=x_1,
            masses=masses,
            batch=batch,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
        )

        # 4. 插值得到 x_t 和瞬时笛卡尔速度 v_t
        try:
            x_t, v_t = self.interpolator.interpolate(path_params, t)
            # [移除] 不再对笛卡尔速度做硬截断，decomposer 内部已处理饱和
        except Exception as e:
            logger.error(f"Error during interpolation: {e}")
            x_t = x_0
            v_t = torch.zeros_like(x_0)

        # 5. [核心] 将笛卡尔速度分解为 SE(3) x T^m 切向量，彻底消灭极端速度问题
        v_trans, v_rot, v_torsion = self.decomposer.decompose(
            pos=x_t,
            vel=v_t,
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
        steps: int = 20,
        inference_t_start: float = 0.0,
        method: str = "rk4",
    ) -> tuple[Tensor, list[Tensor]]:
        """
        [推理接口] 生成轨迹

        Args:
            model: 训练好的 EHFNet 模型
            data: 包含初始状态 x_0 的数据对象
            steps: 积分步数
            inference_t_start: 起始时间（通常为 0.0）
            method: 积分方法（"euler" 或 "rk4"）

        Returns:
            final_pos: 最终生成的坐标 [N, 3]
            trajectory: 轨迹列表
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

        trajectory = [current_pos.clone()]
        dt = (1.0 - inference_t_start) / steps
        B = int(batch.max().item()) + 1

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
                    # RK4 积分器
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
        """

        data["ligand_atom"].pos = pos

        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)

        out = model(data, t_tensor)

        v_trans = out["v_translation"]
        v_rot = out["v_rotation"]
        v_torsion = out.get("v_torsion", None)

        if v_torsion is not None:
            v_torsion = v_torsion.squeeze(-1)

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
        translation_scale: float = 10.0,
    ) -> Tensor:
        """
        生成随机初始位姿（盲对接）
        """

        device = x_ref.device
        dtype = x_ref.dtype

        # 1. 随机扭转
        x_torsioned = x_ref.clone()

        if torsion_indices is not None and torsion_moving_mask is not None:
            T = torsion_indices.shape[0]

            if T > 0:
                rand_angles = (torch.rand(T, device=device, dtype=dtype) * 2 - 1) * torch.pi

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

        # 2. 随机刚体位姿
        # 获取当前（扭转后）的质心 (Center of Mass)
        # 使用统一的质心计算工具
        center = compute_center_of_mass(x_torsioned, batch, masses, dim_size=B)
        
        x_centered = x_torsioned - center[batch]

        # 随机旋转
        rot_matrices = self._random_rotation_matrix(B, device, dtype)
        x_rotated = torch.bmm(rot_matrices[batch], x_centered.unsqueeze(-1)).squeeze(-1)

        # 随机位移（相对于原始中心 center 的偏移，而不是相对于原点）
        # 使用传入的 translation_scale 控制随机初始范围
        translation_offset = torch.randn(B, 3, device=device, dtype=dtype) * translation_scale

        # 最终坐标 = 旋转后的坐标 + 原始中心 + 随机位移偏移
        return x_rotated + center[batch] + translation_offset[batch]


    @staticmethod
    def _random_rotation_matrix(
        B: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """
        使用 Haar Measure 在 SO(3) 上均匀采样随机旋转矩阵
        """

        m = torch.randn(B, 3, 3, device=device, dtype=dtype)
        q, r = torch.linalg.qr(m)

        d = torch.diagonal(r, dim1=-2, dim2=-1)
        q *= torch.sign(d).unsqueeze(-2)

        det = torch.linalg.det(q)
        mask = det < 0

        if mask.any():
            q[mask, :, 0] *= -1

        return q
