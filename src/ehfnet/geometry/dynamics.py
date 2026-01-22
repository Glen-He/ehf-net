"""
动力学几何计算

提供训练和推理阶段的动力学几何计算，包括速度分解、位姿更新和路径插值。
"""

import logging
import torch
import torch.nn.functional as F

from typing import Any
from torch import Tensor
from torch_scatter import scatter_sum

logger = logging.getLogger(__name__)


# 物理化学常量
class PhysicsConstants:
    """
    物理化学常量定义
    """

    # 数值稳定性
    EPSILON = 1e-8              # 通用数值保护
    MIN_NORM = 1e-7             # 最小向量模长

    # 正则化参数
    DAMPING_FACTOR = 1e-4       # Tikhonov 正则化系数

    # 旋转相关
    MIN_ROTATION_ANGLE = 1e-6   # 最小旋转角度（弧度）


class VelocityDecomposer:
    """
    速度场分解器

    将原子速度场分解为物理约束的分量：
    - 刚体平移速度
    - 刚体旋转速度
    - 扭转角速度
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON):
        """
        Args:
            eps: 数值稳定性保护参数
        """

        self.eps = eps


    def decompose(
        self,
        *,
        pos: Tensor,
        vel: Tensor,
        masses: Tensor,
        batch: Tensor,
        torsion_indices: Tensor,
        torsion_moving_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        联合最小二乘法分解速度场。

        Args:
            pos: 原子坐标 [N, 3]
            vel: 原子速度 [N, 3]
            masses: 原子质量 [N] 或 [N, 1]
            batch: 批次索引 [N]
            torsion_indices: 扭转角定义 [T, 4]
            torsion_moving_mask: 移动原子掩码 [T, N]

        Returns:
            v_trans: 平移速度 [B, 3]
            v_rot: 旋转速度（轴角表示）[B, 3]
            v_torsion: 扭转角速度 [T]
        """

        device = pos.device
        dtype = pos.dtype
        N = pos.shape[0]

        if N == 0:
            return (
                torch.zeros(0, 3, device=device, dtype=dtype),
                torch.zeros(0, 3, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
            )

        B = int(batch.max().item()) + 1
        T = torsion_indices.shape[0] if torsion_indices is not None else 0
        total_dofs = 6 * B + T

        # 1. 预计算质心（质量加权）
        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)

        mass_per_mol = scatter_sum(masses, batch, dim=0, dim_size=B)
        mass_per_mol = torch.clamp(mass_per_mol, min=self.eps)
        com = scatter_sum(pos * masses, batch, dim=0, dim_size=B) / mass_per_mol
        r_rel = pos - com[batch]

        # 2. 构建线性方程组 Ax = v
        A = torch.zeros((N * 3, total_dofs), device=device, dtype=dtype)

        # 2.1 填充刚体基（平移+旋转）
        for b in range(B):
            mask = batch == b
            idx_rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)

            if idx_rows.numel() == 0:
                continue

            # 平移：Identity
            for k in range(3):
                A[idx_rows * 3 + k, 6 * b + k] = 1.0

            # 旋转：r 的反对称矩阵
            r = r_rel[idx_rows]
            n_atoms = r.shape[0]
            cross_mat = torch.zeros((n_atoms, 3, 3), device=device, dtype=dtype)
            cross_mat[:, 0, 1] = -r[:, 2]
            cross_mat[:, 0, 2] = r[:, 1]
            cross_mat[:, 1, 0] = r[:, 2]
            cross_mat[:, 1, 2] = -r[:, 0]
            cross_mat[:, 2, 0] = -r[:, 1]
            cross_mat[:, 2, 1] = r[:, 0]

            row_indices = (
                idx_rows.unsqueeze(1) * 3 + torch.arange(3, device=device).unsqueeze(0)
            ).view(-1)
            A[row_indices, 6 * b + 3 : 6 * b + 6] = cross_mat.view(-1, 3)

        # 2.2 填充扭转基
        if T > 0:
            u = pos[torsion_indices[:, 1]]
            v = pos[torsion_indices[:, 2]]
            axis = F.normalize(v - u, dim=-1, eps=self.eps)

            t_idx, n_idx = torsion_moving_mask.nonzero(as_tuple=True)

            if t_idx.numel() > 0:
                r_rot = pos[n_idx] - u[t_idx]
                axis_vec = axis[t_idx]
                vel_tor = torch.cross(axis_vec, r_rot, dim=-1)  # v = w x r

                col_indices = 6 * B + t_idx
                for k in range(3):
                    A[n_idx * 3 + k, col_indices] = vel_tor[:, k]

        # 3. 求解 Ax = b（使用阻尼最小二乘法）
        b_vec = vel.view(-1)  # [3N]

        # 计算 A.T @ A 和 A.T @ b
        AT = A.T
        ATA = torch.matmul(AT, A)
        ATb = torch.matmul(AT, b_vec)

        # 添加 Tikhonov 正则化（阻尼项）
        diag_indices = torch.arange(total_dofs, device=device)
        ATA[diag_indices, diag_indices] += PhysicsConstants.DAMPING_FACTOR

        try:
            # 使用 Cholesky 分解求解（比 inv 更快更稳）
            L = torch.linalg.cholesky(ATA)
            solution = torch.cholesky_solve(ATb.unsqueeze(-1), L).squeeze(-1)

        except RuntimeError:

            # 回退到鲁棒的 solve
            try:
                solution = torch.linalg.solve(ATA, ATb)

            except RuntimeError:
                # 最后的兜底：全零
                logger.warning("Failed to decompose velocity, returning zeros")
                solution = torch.zeros(total_dofs, device=device, dtype=dtype)

        # 4. 拆解结果
        rigid_sol = solution[: 6 * B].view(B, 6)
        v_trans = rigid_sol[:, :3]
        v_rot = rigid_sol[:, 3:]

        v_torsion = (
            solution[6 * B :]
            if T > 0
            else torch.zeros(0, device=device, dtype=dtype)
        )

        return v_trans, v_rot, v_torsion


class PoseUpdater:
    """
    位姿更新器

    根据速度场更新分子位姿，支持刚体变换和扭转角变化。
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON):
        """
        Args:
            eps: 数值稳定性保护参数
        """

        self.eps = eps


    def update(
        self,
        *,
        pos: Tensor,
        masses: Tensor,
        batch: Tensor,
        v_trans: Tensor,
        v_rot: Tensor,
        v_torsion: Tensor | None,
        torsion_indices: Tensor | None,
        torsion_moving_mask: Tensor | None,
        dt: float = 1.0,
    ) -> Tensor:
        """
        更新位姿：先应用扭转，再应用刚体变换。

        Args:
            pos: 当前坐标 [N, 3]
            masses: 原子质量 [N] 或 [N, 1]
            batch: 批次索引 [N]
            v_trans: 平移速度 [B, 3]
            v_rot: 旋转速度 [B, 3]
            v_torsion: 扭转角速度 [T] 或 None
            torsion_indices: 扭转角定义 [T, 4] 或 None
            torsion_moving_mask: 移动原子掩码 [T, N] 或 None
            dt: 时间步长

        Returns:
            更新后的坐标 [N, 3]
        """

        device = pos.device
        dtype = pos.dtype

        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)

        B = v_trans.shape[0]
        T = torsion_indices.shape[0] if torsion_indices is not None else 0
        new_pos = pos.clone()

        # 1. 扭转更新
        if (
            T > 0
            and v_torsion is not None
            and torsion_indices is not None
            and torsion_moving_mask is not None
        ):
            angles = v_torsion * dt

            for i in range(T):
                angle = angles[i]

                if torch.abs(angle) < PhysicsConstants.MIN_ROTATION_ANGLE:
                    continue

                idx0, idx1 = torsion_indices[i, 1], torsion_indices[i, 2]
                origin = new_pos[idx0]
                axis_vec = new_pos[idx1] - origin
                axis_norm = torch.norm(axis_vec)

                if axis_norm < self.eps:
                    continue

                axis = axis_vec / (axis_norm + self.eps)

                mask = torsion_moving_mask[i]

                if mask.dtype != torch.bool:
                    mask = mask > 0.5

                if not mask.any():
                    continue

                rot_mat = self._axis_angle_to_matrix(axis, angle)
                rel_pts = new_pos[mask] - origin

                # (R @ rel^T)^T = rel @ R^T
                new_pos[mask] = torch.matmul(rel_pts, rot_mat.T) + origin

        # 2. 刚体更新
        mass_per_mol = torch.clamp(
            scatter_sum(masses, batch, dim=0, dim_size=B), min=self.eps
        )
        com = scatter_sum(new_pos * masses, batch, dim=0, dim_size=B) / mass_per_mol

        d_trans = v_trans * dt
        d_rot_vec = v_rot * dt
        theta = torch.norm(d_rot_vec, dim=-1, keepdim=True)
        rot_axis = d_rot_vec / (theta + self.eps)

        for b in range(B):
            mask = batch == b

            if not mask.any():
                continue

            angle = theta[b, 0]

            if angle > PhysicsConstants.MIN_ROTATION_ANGLE:
                R = self._axis_angle_to_matrix(rot_axis[b], angle)
                rel = new_pos[mask] - com[b]
                new_pos[mask] = torch.matmul(rel, R.T) + com[b]

            new_pos[mask] += d_trans[b]

        return new_pos


    @staticmethod
    def _axis_angle_to_matrix(axis: Tensor, angle: Tensor) -> Tensor:
        """
        Rodrigues 公式：轴角表示转旋转矩阵

        Args:
            axis: 旋转轴（单位向量）[3]
            angle: 旋转角度（弧度）标量

        Returns:
            旋转矩阵 [3, 3]
        """

        zero = torch.zeros_like(axis[0])
        row0 = torch.stack([zero, -axis[2], axis[1]])
        row1 = torch.stack([axis[2], zero, -axis[0]])
        row2 = torch.stack([-axis[1], axis[0], zero])
        K = torch.stack([row0, row1, row2])

        I = torch.eye(3, device=axis.device, dtype=axis.dtype)
        return I + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * torch.matmul(K, K)


class PathInterpolator:
    """
    路径插值器

    计算两个构象之间的物理合理插值路径（Kabsch对齐 + 扭转角插值）。
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON):
        """
        Args:
            eps: 数值稳定性保护参数
        """

        self.eps = eps

    def compute_path_parameters(
        self,
        pos_0: Tensor,
        pos_1: Tensor,
        masses: Tensor,
        batch: Tensor,
        torsion_indices: Tensor | None,
        torsion_moving_mask: Tensor | None,
    ) -> dict[str, Any]:
        """
        计算 pos_0 到 pos_1 的最优变换参数（Kabsch + 扭转差）

        Args:
            pos_0: 初始坐标 [N, 3]
            pos_1: 目标坐标 [N, 3]
            masses: 原子质量 [N] 或 [N, 1]
            batch: 批次索引 [N]
            torsion_indices: 扭转角定义 [T, 4] 或 None
            torsion_moving_mask: 移动原子掩码 [T, N] 或 None

        Returns:
            包含变换参数的字典
        """

        device = pos_0.device
        dtype = pos_0.dtype
        B = int(batch.max().item()) + 1
        T = torsion_indices.shape[0] if torsion_indices is not None else 0

        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)

        # 1. 质心对齐
        mass_per_mol = torch.clamp(
            scatter_sum(masses, batch, dim=0, dim_size=B), min=self.eps
        )
        com_0 = scatter_sum(pos_0 * masses, batch, dim=0, dim_size=B) / mass_per_mol
        com_1 = scatter_sum(pos_1 * masses, batch, dim=0, dim_size=B) / mass_per_mol
        delta_trans = com_1 - com_0

        pos_0_centered = pos_0 - com_0[batch]
        pos_1_centered = pos_1 - com_1[batch]

        # 2. 计算最佳旋转（Kabsch 算法）
        R = torch.zeros(B, 3, 3, device=device, dtype=dtype)

        for b in range(B):
            mask_b = batch == b

            if not mask_b.any():
                R[b] = torch.eye(3, device=device, dtype=dtype)
                continue

            # 质量加权的协方差矩阵
            P = pos_0_centered[mask_b]  # [N_b, 3]
            Q = pos_1_centered[mask_b]  # [N_b, 3]
            w = masses[mask_b]  # [N_b, 1]
            
            # H = P^T @ W @ Q
            H = torch.matmul(P.T, w * Q)  # [3, 3]
            
            try:
                # SVD 分解：H = U @ S @ V^T
                U, S, Vh = torch.linalg.svd(H)
                
                # 计算旋转矩阵：R = V @ U^T
                R_b = torch.matmul(Vh.T, U.T)
                
                # 确保 det(R) = +1（右手系）
                if torch.linalg.det(R_b) < 0:
                    Vh_corrected = Vh.clone()
                    Vh_corrected[-1, :] *= -1
                    R_b = torch.matmul(Vh_corrected.T, U.T)
                
                R[b] = R_b

            except RuntimeError:
                # SVD 失败时使用单位矩阵
                logger.warning(f"SVD failed for batch {b}, using identity matrix")
                R[b] = torch.eye(3, device=device, dtype=dtype)

        # 3. 计算扭转角差异
        R_expanded = R[batch]
        pos_0_aligned = (
            torch.einsum("nij,nj->ni", R_expanded, pos_0_centered) + com_1[batch]
        )
        delta_torsions = torch.zeros(T, device=device, dtype=dtype)

        if T > 0 and torsion_indices is not None:
            idx0, idx1 = torsion_indices[:, 0], torsion_indices[:, 1]
            idx2, idx3 = torsion_indices[:, 2], torsion_indices[:, 3]

            angle_0 = self._calc_dihedrals(pos_0_aligned, idx0, idx1, idx2, idx3)
            angle_1 = self._calc_dihedrals(pos_1, idx0, idx1, idx2, idx3)

            # 最小角度差
            diff = angle_1 - angle_0
            delta_torsions = torch.atan2(torch.sin(diff), torch.cos(diff))

        return {
            "com_0": com_0,
            "delta_trans": delta_trans,
            "R_total": R,
            "delta_torsions": delta_torsions,
            "pos_0_centered": pos_0_centered,
            "batch": batch,
            "torsion_indices": torsion_indices,
            "torsion_moving_mask": torsion_moving_mask,
        }


    def interpolate(
        self, params: dict[str, Any], t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        计算 t 时刻的 x_t 和 v_t（有限差分）

        Args:
            params: 路径参数字典
            t: 时间参数 [B]

        Returns:
            pos_t: t 时刻的坐标 [N, 3]
            vel_t: t 时刻的速度 [N, 3]
        """

        dt = 1e-3
        t_float = t.view(-1, 1)
        pos_t = self._compute_pose_at_t(params, t_float)

        t_next = torch.clamp(t_float + dt, max=1.0)
        pos_next = self._compute_pose_at_t(params, t_next)

        return pos_t, (pos_next - pos_t) / dt

    def _compute_pose_at_t(self, params: dict[str, Any], t: Tensor) -> Tensor:
        """
        计算 t 时刻的位姿
        """

        batch = params["batch"]

        # 1. 插值平移
        com_t = params["com_0"] + t * params["delta_trans"]

        # 2. 插值旋转（轴角线性插值）
        rot_vec = self._matrix_to_axis_angle(params["R_total"])  # [B, 3]
        R_t = self._rotation_vector_to_matrix(rot_vec * t)  # [B, 3, 3]

        # 3. 应用刚体 + 扭转
        pos_torsioned = params["pos_0_centered"].clone()

        if params["delta_torsions"].numel() > 0:
            torsion_batch_idx = batch[params["torsion_indices"][:, 1]]
            current_angles = params["delta_torsions"] * t[torsion_batch_idx].squeeze()

            for i in range(current_angles.shape[0]):
                ang = current_angles[i]

                if torch.abs(ang) < PhysicsConstants.MIN_ROTATION_ANGLE:
                    continue

                idx1, idx2 = params["torsion_indices"][i, 1], params["torsion_indices"][i, 2]
                u, v = pos_torsioned[idx1], pos_torsioned[idx2]
                axis = F.normalize(v - u, dim=0, eps=PhysicsConstants.EPSILON)

                mask = params["torsion_moving_mask"][i]
                rot_mat = PoseUpdater._axis_angle_to_matrix(axis, ang)

                pts = pos_torsioned[mask] - u
                pos_torsioned[mask] = torch.matmul(pts, rot_mat.T) + u

        R_t_expanded = R_t[batch]
        return torch.einsum("nij,nj->ni", R_t_expanded, pos_torsioned) + com_t[batch]


    @staticmethod
    def _calc_dihedrals(
        pos: Tensor, idx0: Tensor, idx1: Tensor, idx2: Tensor, idx3: Tensor
    ) -> Tensor:
        """
        计算二面角
        """

        b0 = -1.0 * (pos[idx1] - pos[idx0])
        b1 = F.normalize(pos[idx2] - pos[idx1], dim=-1, eps=PhysicsConstants.EPSILON)
        b2 = pos[idx3] - pos[idx2]

        v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
        w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

        return torch.atan2(y, x)


    @staticmethod
    def _matrix_to_axis_angle(R: Tensor) -> Tensor:
        """
        Log map: SO(3) -> so(3)
        """

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        theta = torch.acos(
            torch.clamp((trace - 1) / 2, -1 + PhysicsConstants.EPSILON, 1 - PhysicsConstants.EPSILON)
        )

        axis = torch.stack(
            [
                R[:, 2, 1] - R[:, 1, 2],
                R[:, 0, 2] - R[:, 2, 0],
                R[:, 1, 0] - R[:, 0, 1],
            ],
            dim=1,
        )

        denom = 2 * torch.sin(theta).unsqueeze(1)
        mask = theta > PhysicsConstants.MIN_ROTATION_ANGLE
        result = torch.zeros_like(axis)
        result[mask] = axis[mask] / (denom[mask] + PhysicsConstants.EPSILON) * theta[mask].unsqueeze(1)

        return result


    @staticmethod
    def _rotation_vector_to_matrix(rot_vec: Tensor) -> Tensor:
        """
        Exp map: so(3) -> SO(3) (Batch version)
        """

        angle = torch.norm(rot_vec, dim=-1, keepdim=True)
        mask = angle.squeeze() < PhysicsConstants.MIN_ROTATION_ANGLE
        axis = rot_vec / (angle + PhysicsConstants.EPSILON)

        B = rot_vec.shape[0]
        K = torch.zeros(B, 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype).unsqueeze(0)
        R = (
            I
            + torch.sin(angle).unsqueeze(2) * K
            + (1 - torch.cos(angle)).unsqueeze(2) * torch.matmul(K, K)
        )

        if mask.any():
            R[mask] = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)

        return R
