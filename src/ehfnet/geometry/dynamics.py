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
        pos: 原子坐标 [N, 3]
        batch: 批次索引 [N]
        masses: 原子质量 [N] 或 [N, 1]
        dim_size: 批次大小 (可选)
        eps: 数值稳定性参数 (默认 1e-6)

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
        pos:      原子坐标 [N, 3]
        batch:    批次索引 [N]
        masses:   原子质量 [N] 或 [N, 1]
        dim_size: 批次大小（可选，None 则自动推断）
        eps:      数值稳定性参数

    Returns:
        R: 主惯量帧旋转矩阵 [B, 3, 3]，列向量为主轴（body→world）
    """
    if masses.dim() == 1:
        masses = masses.unsqueeze(-1)      # [N, 1]

    if dim_size is None:
        dim_size = int(batch.max().item()) + 1

    B      = dim_size
    device = pos.device
    dtype  = pos.dtype

    # 使用 detach 的坐标——主惯量帧仅作几何参考基，不需要反向传播
    pos_d = pos.detach()

    # 质心（质量加权）
    masses_clamped = masses.clamp(min=eps)
    mass_per_mol = scatter_sum(masses_clamped, batch, dim=0, dim_size=B).clamp(min=eps)
    com = scatter_sum(pos_d * masses_clamped, batch, dim=0, dim_size=B) / mass_per_mol

    r = pos_d - com[batch]                 # [N, 3] 质心系坐标

    # 单/双原子分子安全处理：协方差秩 < 3，直接退化为单位矩阵
    atoms_per_mol = scatter_sum(
        torch.ones(pos_d.size(0), device=device, dtype=torch.long),
        batch, dim=0, dim_size=B,
    )
    degenerate_mol_mask = atoms_per_mol < 3

    # ── 向量化构建协方差矩阵 [B, 3, 3] ────────────────────────────────────
    weighted_r = r * masses_clamped                                    # [N, 3]
    # 原子级外积：[N, 3, 1] @ [N, 1, 3] → [N, 3, 3]
    atom_cov = torch.bmm(weighted_r.unsqueeze(-1), r.unsqueeze(-2))
    # scatter 聚合到分子维度
    C = scatter_sum(
        atom_cov.reshape(-1, 9), batch, dim=0, dim_size=B
    ).reshape(B, 3, 3)

    # ── 批量 SVD + 符号消歧 ─────────────────────────────────────────────
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)        # [1, 3, 3]

    try:
        U, S, Vh = torch.linalg.svd(C)

        # 【关键】消除 SVD 符号歧义（Sign Ambiguity）
        # 强制 U 对角线元素为正，保证同一分子在坐标微扰时帧方向连续
        signs = torch.sign(torch.diagonal(U, dim1=-2, dim2=-1))        # [B, 3]
        signs[signs == 0] = 1.0
        U  = U * signs.unsqueeze(-2)                                   # [B, 3, 3]
        Vh = Vh * signs.unsqueeze(-1)                                  # 补偿 V 侧

        R_b = torch.bmm(U, Vh)

        # 保证右手系：det(R) = +1
        det = torch.linalg.det(R_b)                                   # [B]
        flip_mask = det < 0
        if flip_mask.any():
            U_corr = U.clone()
            U_corr[flip_mask, :, -1] *= -1
            R_flip = torch.bmm(U_corr[flip_mask], Vh[flip_mask])
            R_b[flip_mask] = R_flip

        # 逐分子 NaN/Inf 修复（仅失败的退化为单位矩阵）
        bad_mask = ~torch.isfinite(R_b).all(dim=-1).all(dim=-1)       # [B]
        if bad_mask.any():
            R_b[bad_mask] = eye.expand(int(bad_mask.sum()), -1, -1)

        # 对单/双原子分子：强制使用单位矩阵
        if degenerate_mol_mask.any():
            R_b[degenerate_mol_mask] = eye.expand(int(degenerate_mol_mask.sum()), -1, -1)

        # 对线性/近线性或平面分子：显式修复最小奇异向量，避免第三轴不稳定
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
    物理化学常量定义
    """

    # 数值稳定性
    EPSILON = 1e-8              # 通用数值保护
    MIN_NORM = 1e-7             # 最小向量模长

    # 正则化参数
    DAMPING_FACTOR = 1e-4       # Tikhonov 正则化系数

    # 旋转相关
    MIN_ROTATION_ANGLE = 1e-6   # 最小旋转角度（弧度）


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
        Args:
            eps: 数值稳定性保护参数
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
            pos: 原子坐标 [N, 3]
            vel: 原子级笛卡尔导数 [N, 3]
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

        v_trans = torch.zeros(B, 3, device=device, dtype=dtype)
        v_rot = torch.zeros(B, 3, device=device, dtype=dtype)
        v_torsion = torch.zeros(T, device=device, dtype=dtype)

        if not torch.isfinite(pos_const).all() or not torch.isfinite(vel).all():
            logger.debug("Numerical instability in decompose: non-finite coordinates or velocities.")
            return v_trans, v_rot, v_torsion

        for b in range(B):
            atom_ids = torch.nonzero(batch == b, as_tuple=False).squeeze(-1)
            if atom_ids.numel() == 0:
                continue

            pos_b = pos_const[atom_ids]
            rel_pos_b = pos_b - com[b]
            vel_b = vel[atom_ids]
            masses_b = masses[atom_ids].clamp(min=self.eps)

            basis_fields: list[Tensor] = []
            trans_basis = self._translation_velocity_basis(
                int(atom_ids.numel()),
                device=device,
                dtype=dtype,
            )
            rot_basis = self._angular_velocity_basis(rel_pos_b)
            basis_fields.extend([trans_basis[:, :, k] for k in range(3)])
            basis_fields.extend([rot_basis[:, :, k] for k in range(3)])

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
                        rel_rot = pos_b[moving_mask] - u[local_idx]
                        axis_vec = axis[local_idx].unsqueeze(0).expand_as(rel_rot)
                        field[moving_mask] = torch.cross(axis_vec, rel_rot, dim=-1)
                    basis_fields.append(field)

            solution = self._solve_local_weighted_system(
                basis_fields=basis_fields,
                target_velocity=vel_b,
                masses=masses_b,
            )
            if solution.numel() < 6:
                continue

            v_trans[b] = solution[:3]
            v_rot[b] = solution[3:6]
            if local_torsion_ids.numel() > 0:
                v_torsion[local_torsion_ids] = solution[6 : 6 + int(local_torsion_ids.numel())]

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
            angles = (v_torsion * dt).reshape(-1)  # ensure 1-D [T]

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
        com = compute_center_of_mass(new_pos, batch, masses, dim_size=B)

        d_trans = v_trans * dt
        d_rot_vec = v_rot * dt
        theta = torch.norm(d_rot_vec, dim=-1, keepdim=True)
        rot_axis = d_rot_vec / (theta + self.eps)

        # 向量化刚体更新：避免 Python for 循环，所有分子并行处理
        # 计算全体分子的旋转矩阵 [B, 3, 3]
        R_all = self._axis_angle_to_matrix_batched(rot_axis, theta.squeeze(-1))  # [B, 3, 3]

        # 小角掩码：角度过小时跳过旋转（保持原坐标），防止数值噪声
        small_angle = (theta.squeeze(-1) <= PhysicsConstants.MIN_ROTATION_ANGLE)  # [B]
        # 对小角分子，旋转矩阵退化为单位矩阵
        eye_B = torch.eye(3, device=new_pos.device, dtype=new_pos.dtype).unsqueeze(0).expand(B, -1, -1)
        R_all = torch.where(small_angle[:, None, None], eye_B, R_all)  # [B, 3, 3]

        # 展开到原子维度
        com_per_atom   = com[batch]          # [N, 3]
        R_per_atom     = R_all[batch]        # [N, 3, 3]
        d_trans_per_atom = d_trans[batch]    # [N, 3]

        # 中心化 → 旋转 → 移回质心 → 平移
        pos_centered = new_pos - com_per_atom                                  # [N, 3]
        pos_rotated  = torch.einsum('nij,nj->ni', R_per_atom, pos_centered)    # [N, 3]
        new_pos = pos_rotated + com_per_atom + d_trans_per_atom                # [N, 3]

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
        x, y, z = axis.unbind(-1)          # 各 [B]
        zeros = torch.zeros_like(x)

        # 反对称矩阵 K [B, 3, 3]
        K = torch.stack([
            torch.stack([zeros,  -z,      y    ], dim=-1),
            torch.stack([z,       zeros,  -x   ], dim=-1),
            torch.stack([-y,      x,      zeros], dim=-1),
        ], dim=-2)

        I = torch.eye(3, device=axis.device, dtype=axis.dtype).unsqueeze(0)  # [1, 3, 3]
        s = torch.sin(angle)[:, None, None]          # [B, 1, 1]
        c = (1.0 - torch.cos(angle))[:, None, None]  # [B, 1, 1]

        return I + s * K + c * torch.bmm(K, K)       # [B, 3, 3]


class PathInterpolator:
    """
    路径插值器

    计算两个构象之间的物理合理插值路径（Kabsch对齐 + 扭转角插值）。
    """

    def __init__(self, eps: float = PhysicsConstants.EPSILON, fd_dt: float = 0.05):
        """
        Args:
            eps:   数值稳定性保护参数
            fd_dt: 速度有限差分步长，v = Δpos / fd_dt（默认 0.05 Å/unit-t）
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
        com_0 = compute_center_of_mass(pos_0, batch, masses, dim_size=B)
        com_1 = compute_center_of_mass(pos_1, batch, masses, dim_size=B)
        delta_trans = com_1 - com_0

        pos_0_centered = pos_0 - com_0[batch]
        pos_1_centered = pos_1 - com_1[batch]

        if not torch.isfinite(pos_0_centered).all() or not torch.isfinite(pos_1_centered).all():
            logger.warning("NaN detected in centered positions during path parameter computation.")
            return {
                "com_0": com_0,
                "delta_trans": delta_trans,
                "R_total": torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1),
                "delta_torsions": torch.zeros(T, device=device, dtype=dtype),
                "pos_0_centered": torch.zeros_like(pos_0_centered),
                "batch": batch,
                "torsion_indices": torsion_indices,
                "torsion_moving_mask": torsion_moving_mask,
            }

        # 2. 计算最佳旋转（批量质量加权 Kabsch）
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
                finite_rot = torch.isfinite(R_valid).all(dim=-1).all(dim=-1)
                if bool(finite_rot.any()):
                    R[valid_indices[finite_rot]] = R_valid[finite_rot].to(device=device, dtype=dtype)
                fallback_indices = valid_indices[~finite_rot]
            except RuntimeError as exc:
                logger.warning("Batched Kabsch SVD failed: %s. Falling back to per-molecule solve.", exc)
                fallback_indices = valid_indices

        for idx in fallback_indices.tolist():
            R[idx] = self._solve_single_kabsch(H[idx], device=device, dtype=dtype)

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
            
            if not torch.isfinite(delta_torsions).all():
                logger.warning("NaN detected in delta_torsions, replacing with zeros")
                delta_torsions = torch.nan_to_num(delta_torsions, nan=0.0)

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

        t_float = t.view(-1, 1)
        pos_t = self._compute_pose_at_t(params, t_float)

        t_next = torch.clamp(t_float + self.fd_dt, max=1.0)
        pos_next = self._compute_pose_at_t(params, t_next)

        # 用实际步长而非固定 fd_dt 做归一化：
        # 当 t 接近 1.0 时 clamp 会压缩实际步长，若仍除以 fd_dt 会
        # 系统性低估目标速度，导致模型在 t→1 阶段学到"刹车"行为。
        # actual_dt 形状 [B, 1]，需按原子 batch 索引展开为 [N, 1]
        # 才能与 pos 差值 [N, 3] 正确广播。
        actual_dt = (t_next - t_float).clamp(min=self.eps)       # [B, 1]
        actual_dt_per_atom = actual_dt[params["batch"]]           # [N, 1]
        return pos_t, (pos_next - pos_t) / actual_dt_per_atom


    def _compute_pose_at_t(self, params: dict[str, Any], t: Tensor) -> Tensor:
        """
        计算 t 时刻的位姿
        """

        batch = params["batch"]

        # 1. 插值平移
        com_t = params["com_0"] + t * params["delta_trans"]

        # 2. 插值旋转（轴角线性插值）
        rot_vec = self._matrix_to_axis_angle(params["R_total"])     # [B, 3]
        R_t = self._rotation_vector_to_matrix(rot_vec * t)          # [B, 3, 3]

        # 3. 应用刚体 + 扭转
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
                rot_mat = PoseUpdater._axis_angle_to_matrix(axis, ang)

                pts = pos_torsioned[mask] - u
                pos_torsioned[mask] = torch.matmul(pts, rot_mat.T) + u

        R_t_expanded = R_t[batch]
        return torch.einsum("nij,nj->ni", R_t_expanded, pos_torsioned) + com_t[batch]


    @staticmethod
    def _calc_dihedrals(pos: Tensor, idx0: Tensor, idx1: Tensor, idx2: Tensor, idx3: Tensor) -> Tensor:
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
        对数映射（log map）：SO(3) -> so(3)
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
        指数映射（exp map）：so(3) -> SO(3)（batch 版本）
        """

        angle = torch.norm(rot_vec, dim=-1, keepdim=True)
        mask = angle.squeeze(-1) < PhysicsConstants.MIN_ROTATION_ANGLE
        axis = rot_vec / (angle + PhysicsConstants.EPSILON)

        B = rot_vec.shape[0]
        K = torch.zeros(B, 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        identity_mat = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype).unsqueeze(0)
        R = (
            identity_mat
            + torch.sin(angle).unsqueeze(2) * K
            + (1 - torch.cos(angle)).unsqueeze(2) * torch.matmul(K, K)
        )

        if mask.any():
            R[mask] = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)

        return R
