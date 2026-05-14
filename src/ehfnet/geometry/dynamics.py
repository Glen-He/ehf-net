"""
动力学几何工具。

负责速度分解、位姿更新、路径插值和相关几何变换，
是训练与推理阶段的核心几何实现。
"""


import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum

logger = logging.getLogger(__name__)


def compute_center_of_mass(
    pos: Tensor,
    batch: Tensor,
    masses: Tensor,
    dim_size: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    统一质心计算工具

    Args:
        pos: 节点坐标张量。
        batch: batch。
        masses: masses。
        dim_size: 维度size。
        eps: eps。

    Returns:
        质心坐标 [B, 3]
    """
    if masses.dim() == 1:
        masses = masses.unsqueeze(-1)

    if dim_size is None:
        dim_size = int(batch.max().item()) + 1

    mass_per_mol = scatter_sum(masses, batch, dim=0, dim_size=dim_size)
    mass_per_mol = torch.clamp(mass_per_mol, min=eps)
    com = scatter_sum(pos * masses, batch, dim=0, dim_size=dim_size) / mass_per_mol
    return com


def compute_principal_frame(
    pos: Tensor,
    batch: Tensor,
    masses: Tensor,
    dim_size: int | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    计算每个分子的主惯量帧（质量加权协方差矩阵 SVD）。

    原理：
        C_b = Σ_i m_i · r_i · r_i^T  （3×3 惯量张量近似）
        C_b = U S V^T  →  R_b = U V^T
        R_b 的列向量即三个主轴方向（世界坐标系表达）。

    坐标变换约定：
        体帧坐标：x_body = R^T @ (x_world - com)
        世界帧向量：v_world = R @ v_body  ← 等变投影

    退化处理：
        - 原子数 < 2 → 单位矩阵
        - C 奇异或含 NaN → 单位矩阵
        - θ ≈ π 旋转（线性/平面分子第三轴近零）→ 翻转最小奇异向量符号

    Args:
        pos: 节点坐标张量。
        batch: batch。
        masses: masses。
        dim_size: 维度size。
        eps: eps。

    Returns:
        R: 主惯量帧旋转矩阵 [B, 3, 3]，列向量为主轴（body→world）
    """
    if masses.dim() == 1:
        masses = masses.unsqueeze(-1)

    if dim_size is None:
        dim_size = int(batch.max().item()) + 1

    B      = dim_size
    device = pos.device
    dtype  = pos.dtype

    pos_d = pos.detach()
    masses_clamped = masses.clamp(min=eps)
    mass_per_mol = scatter_sum(masses_clamped, batch, dim=0, dim_size=B).clamp(min=eps)
    com = scatter_sum(pos_d * masses_clamped, batch, dim=0, dim_size=B) / mass_per_mol

    r = pos_d - com[batch]

    atoms_per_mol = scatter_sum(
        torch.ones(pos_d.size(0), device=device, dtype=torch.long),
        batch, dim=0, dim_size=B,
    )
    degenerate_mol_mask = atoms_per_mol < 3

    weighted_r = r * masses_clamped
    atom_cov = torch.bmm(weighted_r.unsqueeze(-1), r.unsqueeze(-2))
    C = scatter_sum(
        atom_cov.reshape(-1, 9), batch, dim=0, dim_size=B
    ).reshape(B, 3, 3)

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    try:
        U, S, Vh = torch.linalg.svd(C)

        signs = torch.sign(torch.diagonal(U, dim1=-2, dim2=-1))
        signs[signs == 0] = 1.0
        U = U * signs.unsqueeze(-2)
        Vh = Vh * signs.unsqueeze(-1)

        R_b = torch.bmm(U, Vh)

        det = torch.linalg.det(R_b)
        flip_mask = det < 0
        if flip_mask.any():
            U_corr = U.clone()
            U_corr[flip_mask, :, -1] *= -1
            R_flip = torch.bmm(U_corr[flip_mask], Vh[flip_mask])
            R_b[flip_mask] = R_flip

        bad_mask = ~torch.isfinite(R_b).all(dim=-1).all(dim=-1)
        if bad_mask.any():
            R_b[bad_mask] = eye.expand(int(bad_mask.sum()), -1, -1)

        if degenerate_mol_mask.any():
            R_b[degenerate_mol_mask] = eye.expand(int(degenerate_mol_mask.sum()), -1, -1)

        small_sv_mask = (~degenerate_mol_mask) & (S[:, -1] < 1e-5)
        if small_sv_mask.any():
            for idx in small_sv_mask.nonzero(as_tuple=False).view(-1).tolist():
                u_fix = U[idx].clone()
                u_fix[:, -1] = torch.linalg.cross(u_fix[:, 0], u_fix[:, 1])
                R_b[idx] = u_fix @ Vh[idx]

        return R_b

    except RuntimeError as e:
        logger.warning(f"Batched SVD failed: {e}. Falling back to identity matrices.")
        return eye.expand(B, -1, -1).clone()


class PhysicsConstants:
    """
    动力学常量集合。

    集中声明动力学与几何更新中使用的固定常量，
    避免数值定义散落在路径插值和位姿更新逻辑中。
    """

    EPSILON = 1e-8
    MIN_NORM = 1e-7
    DAMPING_FACTOR = 1e-4
    MIN_ROTATION_ANGLE = 1e-6


class TangentTargetProjector:
    """
    切空间目标投影器

    将环境空间中的原子级笛卡尔导数投影到模型使用的
    SE(3) x T^m 切空间目标：
    - 刚体平移速度
    - 刚体旋转速度
    - 扭转角速度
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON):
        """
        初始化对象。

        Args:
            eps: eps。
        """

        self.eps = eps

    @staticmethod
    def _translation_velocity_basis(num_atoms: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_atoms, -1, -1)

    @staticmethod
    def _angular_velocity_basis(rel_pos: Tensor) -> Tensor:
        basis = torch.zeros((rel_pos.size(0), 3, 3), device=rel_pos.device, dtype=rel_pos.dtype)
        basis[:, 0, 1] = rel_pos[:, 2]
        basis[:, 0, 2] = -rel_pos[:, 1]
        basis[:, 1, 0] = -rel_pos[:, 2]
        basis[:, 1, 2] = rel_pos[:, 0]
        basis[:, 2, 0] = rel_pos[:, 1]
        basis[:, 2, 1] = -rel_pos[:, 0]
        return basis

    def _solve_local_weighted_system(
        self,
        *,
        basis_fields: list[Tensor],
        target_velocity: Tensor,
        masses: Tensor,
    ) -> Tensor:
        if not basis_fields:
            return target_velocity.new_zeros((0,))

        cols = torch.stack(
            [torch.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1) for field in basis_fields],
            dim=-1,
        )
        rhs = target_velocity.reshape(-1)
        sqrt_w = masses.clamp(min=self.eps).sqrt().repeat_interleave(3)
        cols_w = cols * sqrt_w.unsqueeze(-1)
        rhs_w = rhs * sqrt_w

        gram = cols_w.transpose(0, 1) @ cols_w
        rhs_proj = cols_w.transpose(0, 1) @ rhs_w
        damping = torch.eye(
            gram.size(0),
            device=gram.device,
            dtype=gram.dtype,
        ) * PhysicsConstants.DAMPING_FACTOR

        try:
            solution = torch.linalg.solve(gram + damping, rhs_proj)
        except RuntimeError:
            solution = torch.linalg.pinv(gram + damping) @ rhs_proj

        if not torch.isfinite(solution).all():
            return target_velocity.new_zeros((len(basis_fields),))

        max_solution = 1000.0
        return torch.clamp(solution, min=-max_solution, max=max_solution)


    def decompose(
        self,
        *,
        pos: Tensor,
        vel: Tensor,
        masses: Tensor,
        batch: Tensor,
        torsion_indices: Tensor | None,
        torsion_moving_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        联合最小二乘法将笛卡尔导数投影为切空间目标。


        Args:
            pos: 节点坐标张量。
            vel: vel。
            masses: masses。
            batch: batch。
            torsion_indices: 扭转角对应的原子索引张量。
            torsion_moving_mask: 扭转更新时需要跟随旋转的原子掩码。


        Returns:
            v_translation: 平移速度 [B, 3]
            v_rotation: 旋转速度（轴角表示）[B, 3]
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

        if masses.dim() == 2:
            masses = masses.squeeze(-1)
        masses = masses.to(device=device, dtype=dtype)

        pos_const = pos.detach()
        com = compute_center_of_mass(pos_const, batch, masses, dim_size=B)
        torsion_batch = (
            batch[torsion_indices[:, 1]]
            if torsion_indices is not None and T > 0
            else torch.zeros(0, device=device, dtype=batch.dtype)
        )

        v_translation = torch.zeros(B, 3, device=device, dtype=dtype)
        v_rotation = torch.zeros(B, 3, device=device, dtype=dtype)
        v_torsion = torch.zeros(T, device=device, dtype=dtype)

        if not torch.isfinite(pos_const).all() or not torch.isfinite(vel).all():
            logger.debug("Numerical instability in decompose: non-finite coordinates or velocities.")
            return v_translation, v_rotation, v_torsion

        for b in range(B):
            atom_ids = torch.nonzero(batch == b, as_tuple=False).squeeze(-1)
            if atom_ids.numel() == 0:
                continue

            pos_b = pos_const[atom_ids]
            rel_pos_b = pos_b - com[b]
            vel_b = vel[atom_ids]
            masses_b = masses[atom_ids].clamp(min=self.eps)

            basis_fields: list[Tensor] = []
            translation_basis = self._translation_velocity_basis(
                int(atom_ids.numel()),
                device=device,
                dtype=dtype,
            )
            rotation_basis = self._angular_velocity_basis(rel_pos_b)
            basis_fields.extend([translation_basis[:, :, k] for k in range(3)])
            basis_fields.extend([rotation_basis[:, :, k] for k in range(3)])

            local_torsion_ids = (
                torch.nonzero(torsion_batch == b, as_tuple=False).squeeze(-1)
                if T > 0
                else torch.zeros(0, device=device, dtype=torch.long)
            )

            if local_torsion_ids.numel() > 0 and torsion_moving_mask is not None and torsion_indices is not None:
                local_masks = torsion_moving_mask[local_torsion_ids]
                if local_masks.dtype != torch.bool:
                    local_masks = local_masks > 0.5
                local_masks = local_masks[:, atom_ids]

                u = pos_const[torsion_indices[local_torsion_ids, 1]]
                v = pos_const[torsion_indices[local_torsion_ids, 2]]
                axis = F.normalize(v - u, dim=-1, eps=self.eps)

                for local_idx in range(int(local_torsion_ids.numel())):
                    field = torch.zeros((atom_ids.numel(), 3), device=device, dtype=dtype)
                    moving_mask = local_masks[local_idx]
                    if bool(moving_mask.any()):
                        relative_rotation = pos_b[moving_mask] - u[local_idx]
                        axis_vec = axis[local_idx].unsqueeze(0).expand_as(relative_rotation)
                        field[moving_mask] = torch.cross(axis_vec, relative_rotation, dim=-1)
                    basis_fields.append(field)

            solution = self._solve_local_weighted_system(
                basis_fields=basis_fields,
                target_velocity=vel_b,
                masses=masses_b,
            )
            if solution.numel() < 6:
                continue

            v_translation[b] = solution[:3]
            v_rotation[b] = solution[3:6]
            if local_torsion_ids.numel() > 0:
                v_torsion[local_torsion_ids] = solution[6 : 6 + int(local_torsion_ids.numel())]

        return v_translation, v_rotation, v_torsion


class PoseUpdater:
    """
    位姿更新器。

    负责根据模型预测的平移、旋转和扭转速度更新当前分子位姿，
    是 ODE 推理轨迹和局部位姿演化的核心几何组件。
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON):
        """
        初始化对象。

        Args:
            eps: eps。
        """

        self.eps = eps


    def update(
        self,
        *,
        pos: Tensor,
        masses: Tensor,
        batch: Tensor,
        v_translation: Tensor,
        v_rotation: Tensor,
        v_torsion: Tensor | None,
        torsion_indices: Tensor | None,
        torsion_moving_mask: Tensor | None,
        dt: float = 1.0,
    ) -> Tensor:
        """
        更新位姿：先应用扭转，再应用刚体变换。


        Args:
            pos: 节点坐标张量。
            masses: masses。
            batch: batch。
            v_translation: 平移速度。
            v_rotation: 旋转速度。
            v_torsion: 扭转角速度或扭转更新量。
            torsion_indices: 扭转角对应的原子索引张量。
            torsion_moving_mask: 扭转更新时需要跟随旋转的原子掩码。
            dt: 时间步长。


        Returns:
            更新后的坐标 [N, 3]
        """

        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)

        B = v_translation.shape[0]
        T = torsion_indices.shape[0] if torsion_indices is not None else 0
        new_pos = pos.clone()

        if (
            T > 0
            and v_torsion is not None
            and torsion_indices is not None
            and torsion_moving_mask is not None
        ):
            angles = (v_torsion * dt).reshape(-1)

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

                rotation_matrix = self._axis_angle_to_matrix(axis, angle)
                rel_pts = new_pos[mask] - origin

                new_pos[mask] = torch.matmul(rel_pts, rotation_matrix.T) + origin

        com = compute_center_of_mass(new_pos, batch, masses, dim_size=B)

        delta_translation = v_translation * dt
        delta_rotation_vector = v_rotation * dt
        theta = torch.norm(delta_rotation_vector, dim=-1, keepdim=True)
        rotation_axis = delta_rotation_vector / (theta + self.eps)

        R_all = self._axis_angle_to_matrix_batched(rotation_axis, theta.squeeze(-1))

        small_angle = theta.squeeze(-1) <= PhysicsConstants.MIN_ROTATION_ANGLE
        eye_B = torch.eye(3, device=new_pos.device, dtype=new_pos.dtype).unsqueeze(0).expand(B, -1, -1)
        R_all = torch.where(small_angle[:, None, None], eye_B, R_all)

        com_per_atom = com[batch]
        R_per_atom = R_all[batch]
        delta_translation_per_atom = delta_translation[batch]

        pos_centered = new_pos - com_per_atom
        pos_rotated = torch.einsum('nij,nj->ni', R_per_atom, pos_centered)
        new_pos = pos_rotated + com_per_atom + delta_translation_per_atom

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

        identity_mat = torch.eye(3, device=axis.device, dtype=axis.dtype)
        return identity_mat + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * torch.matmul(K, K)


    @staticmethod
    def _axis_angle_to_matrix_batched(axis: Tensor, angle: Tensor) -> Tensor:
        """
        批量 Rodrigues 公式：一次为所有分子计算旋转矩阵。

        Args:
            axis: 旋转轴（单位向量）[B, 3]
            angle: 旋转角度（弧度）[B]

        Returns:
            旋转矩阵 [B, 3, 3]
        """
        x, y, z = axis.unbind(-1)
        zeros = torch.zeros_like(x)

        K = torch.stack([
            torch.stack([zeros,  -z,      y    ], dim=-1),
            torch.stack([z,       zeros,  -x   ], dim=-1),
            torch.stack([-y,      x,      zeros], dim=-1),
        ], dim=-2)

        I = torch.eye(3, device=axis.device, dtype=axis.dtype).unsqueeze(0)
        s = torch.sin(angle)[:, None, None]
        c = (1.0 - torch.cos(angle))[:, None, None]

        return I + s * K + c * torch.bmm(K, K)


class PathInterpolator:
    """
    路径插值器。

    负责在初始构象与目标构象之间构造物理上更合理的插值路径，
    为流匹配训练提供位置轨迹和速度监督目标。
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON, fd_dt: float = 0.05):
        """
        初始化对象。

        Args:
            eps: eps。
            fd_dt: 有限差分计算目标速度时使用的时间步长。
        """

        self.eps = eps
        self.fd_dt = fd_dt

    def _solve_single_kabsch(self, H: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        eye = torch.eye(3, device=device, dtype=dtype)
        try:
            H_cpu = H.to("cpu", dtype=torch.float64)
            U_cpu, _, Vh_cpu = torch.linalg.svd(H_cpu, full_matrices=False)
            R_cpu = Vh_cpu.T @ U_cpu.T
            if torch.linalg.det(R_cpu) < 0:
                Vh_cpu = Vh_cpu.clone()
                Vh_cpu[-1, :] *= -1
                R_cpu = Vh_cpu.T @ U_cpu.T
            return R_cpu.to(device=device, dtype=dtype)
        except Exception:
            return eye


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
            pos_0: pos0。
            pos_1: pos1。
            masses: masses。
            batch: batch。
            torsion_indices: 扭转角对应的原子索引张量。
            torsion_moving_mask: 扭转更新时需要跟随旋转的原子掩码。


        Returns:
            包含变换参数的字典
        """

        device = pos_0.device
        dtype = pos_0.dtype
        B = int(batch.max().item()) + 1
        T = torsion_indices.shape[0] if torsion_indices is not None else 0

        if masses.dim() == 1:
            masses = masses.unsqueeze(-1)

        com_0 = compute_center_of_mass(pos_0, batch, masses, dim_size=B)
        com_1 = compute_center_of_mass(pos_1, batch, masses, dim_size=B)
        delta_translation = com_1 - com_0

        pos_0_centered = pos_0 - com_0[batch]
        pos_1_centered = pos_1 - com_1[batch]

        if not torch.isfinite(pos_0_centered).all() or not torch.isfinite(pos_1_centered).all():
            logger.warning("NaN detected in centered positions during path parameter computation.")
            return {
                "com_0": com_0,
                "delta_translation": delta_translation,
                "R_total": torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1),
                "delta_torsions": torch.zeros(T, device=device, dtype=dtype),
                "pos_0_centered": torch.zeros_like(pos_0_centered),
                "batch": batch,
                "torsion_indices": torsion_indices,
                "torsion_moving_mask": torsion_moving_mask,
            }

        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        R = eye.clone()
        atom_counts = scatter_sum(
            torch.ones(pos_0.size(0), device=device, dtype=torch.long),
            batch,
            dim=0,
            dim_size=B,
        )
        atom_cov = torch.bmm(
            (pos_0_centered * masses).unsqueeze(-1),
            pos_1_centered.unsqueeze(-2),
        )
        H = scatter_sum(atom_cov.reshape(-1, 9), batch, dim=0, dim_size=B).reshape(B, 3, 3)
        valid_mask = (
            (atom_counts >= 2)
            & torch.isfinite(H).all(dim=-1).all(dim=-1)
            & (torch.norm(H.reshape(B, -1), dim=-1) >= self.eps)
        )

        fallback_indices = torch.zeros(0, device=device, dtype=torch.long)
        if bool(valid_mask.any()):
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            H_valid = H[valid_indices]
            try:
                U, _, Vh = torch.linalg.svd(H_valid, full_matrices=False)
                R_valid = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
                det = torch.linalg.det(R_valid)
                if bool((det < 0).any()):
                    Vh = Vh.clone()
                    Vh[det < 0, -1, :] *= -1
                    R_valid = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
                finite_rotation = torch.isfinite(R_valid).all(dim=-1).all(dim=-1)
                if bool(finite_rotation.any()):
                    R[valid_indices[finite_rotation]] = R_valid[finite_rotation].to(device=device, dtype=dtype)
                fallback_indices = valid_indices[~finite_rotation]
            except RuntimeError as exc:
                logger.warning("Batched Kabsch SVD failed: %s. Falling back to per-molecule solve.", exc)
                fallback_indices = valid_indices

        for idx in fallback_indices.tolist():
            R[idx] = self._solve_single_kabsch(H[idx], device=device, dtype=dtype)

        R_expanded = R[batch]
        pos_0_aligned = (
            torch.einsum("nij,nj->ni", R_expanded, pos_0_centered) + com_1[batch]
        )
        delta_torsions = torch.zeros(T, device=device, dtype=dtype)

        if T > 0 and torsion_indices is not None:
            idx0, idx1 = torsion_indices[:, 0], torsion_indices[:, 1]
            idx2, idx3 = torsion_indices[:, 2], torsion_indices[:, 3]

            angle_0 = self._calc_dihedrals(
                pos_0_aligned, idx0=idx0, idx1=idx1, idx2=idx2, idx3=idx3
            )
            angle_1 = self._calc_dihedrals(pos_1, idx0=idx0, idx1=idx1, idx2=idx2, idx3=idx3)

            diff = angle_1 - angle_0
            delta_torsions = torch.atan2(torch.sin(diff), torch.cos(diff))

            if not torch.isfinite(delta_torsions).all():
                logger.warning("NaN detected in delta_torsions, replacing with zeros")
                delta_torsions = torch.nan_to_num(delta_torsions, nan=0.0)

        return {
            "com_0": com_0,
            "delta_translation": delta_translation,
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
            params: params。
            t: t。

        Returns:
            pos_t: t 时刻的坐标 [N, 3]
            vel_t: t 时刻的速度 [N, 3]
        """

        t_float = t.view(-1, 1)
        pos_t = self._compute_pose_at_t(params, t_float)

        t_next = torch.clamp(t_float + self.fd_dt, max=1.0)
        pos_next = self._compute_pose_at_t(params, t_next)

        actual_dt = (t_next - t_float).clamp(min=self.eps)
        actual_dt_per_atom = actual_dt[params["batch"]]
        return pos_t, (pos_next - pos_t) / actual_dt_per_atom


    def _compute_pose_at_t(self, params: dict[str, Any], t: Tensor) -> Tensor:
        """
        计算 t 时刻的位姿

        Returns:
            Tensor: 返回计算得到的张量结果。
        """

        batch = params["batch"]

        com_t = params["com_0"] + t * params["delta_translation"]

        rotation_vector = self._matrix_to_axis_angle(params["R_total"])
        R_t = self._rotation_vector_to_matrix(rotation_vector * t)

        pos_torsioned = params["pos_0_centered"].clone()

        if params["delta_torsions"].numel() > 0:
            torsion_batch_idx = batch[params["torsion_indices"][:, 1]]
            current_angles = params["delta_torsions"] * t[torsion_batch_idx].squeeze(-1)

            for i in range(current_angles.shape[0]):
                ang = current_angles[i]

                if torch.abs(ang) < PhysicsConstants.MIN_ROTATION_ANGLE:
                    continue

                idx1, idx2 = params["torsion_indices"][i, 1], params["torsion_indices"][i, 2]
                u, v = pos_torsioned[idx1], pos_torsioned[idx2]
                axis = F.normalize(v - u, dim=0, eps=PhysicsConstants.EPSILON)

                mask = params["torsion_moving_mask"][i]
                rotation_matrix = PoseUpdater._axis_angle_to_matrix(axis, ang)

                pts = pos_torsioned[mask] - u
                pos_torsioned[mask] = torch.matmul(pts, rotation_matrix.T) + u

        R_t_expanded = R_t[batch]
        return torch.einsum("nij,nj->ni", R_t_expanded, pos_torsioned) + com_t[batch]


    @staticmethod
    def _calc_dihedrals(
        pos: Tensor,
        *,
        idx0: Tensor,
        idx1: Tensor,
        idx2: Tensor,
        idx3: Tensor,
    ) -> Tensor:
        """
        计算由四组原子索引定义的二面角（弧度）。

        Args:
            pos: 原子坐标，形状 [N, 3]。
            idx0, idx1, idx2, idx3: 四元组原子索引，形状 [T]，定义 T 个二面角。

        Returns:
            Tensor: 二面角弧度值，形状 [T]。
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
        对数映射（log map）：SO(3) -> so(3)

        Returns:
            Tensor: 返回计算得到的张量结果。
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
    def _rotation_vector_to_matrix(rotation_vector: Tensor) -> Tensor:
        """
        指数映射（exp map）：so(3) -> SO(3)（batch 版本）

        Returns:
            Tensor: 返回计算得到的张量结果。
        """

        angle = torch.norm(rotation_vector, dim=-1, keepdim=True)
        mask = angle.squeeze(-1) < PhysicsConstants.MIN_ROTATION_ANGLE
        axis = rotation_vector / (angle + PhysicsConstants.EPSILON)

        B = rotation_vector.shape[0]
        K = torch.zeros(
            B,
            3,
            3,
            device=rotation_vector.device,
            dtype=rotation_vector.dtype,
        )
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        identity_mat = torch.eye(
            3,
            device=rotation_vector.device,
            dtype=rotation_vector.dtype,
        ).unsqueeze(0)
        R = (
            identity_mat
            + torch.sin(angle).unsqueeze(2) * K
            + (1 - torch.cos(angle)).unsqueeze(2) * torch.matmul(K, K)
        )

        if mask.any():
            R[mask] = torch.eye(
                3,
                device=rotation_vector.device,
                dtype=rotation_vector.dtype,
            )

        return R
