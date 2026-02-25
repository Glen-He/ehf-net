"""
静态几何计算

提供数据预处理阶段的几何计算功能，包括二面角计算和可旋转键识别。
"""

import logging
import numpy as np

from rdkit import Chem

logger = logging.getLogger(__name__)


# 物理化学常量
MIN_BOND_LENGTH = 0.5   # Å, 最小合理键长
MAX_BOND_LENGTH = 3.0   # Å, 最大合理键长
EPSILON = 1e-8          # 数值稳定性保护


def calculate_dihedral(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float | None:
    """
    计算由四个3D点定义的二面角 (p0-p1-p2-p3)。

    二面角定义为两个平面之间的夹角:
    - 平面1由 p0-p1-p2 定义
    - 平面2由 p1-p2-p3 定义
    - p1-p2 是旋转轴

    Args:
        p0: 第一个点的3D坐标 (shape=(3,))
        p1: 第二个点的3D坐标，定义轴线起点
        p2: 第三个点的3D坐标，定义轴线终点
        p3: 第四个点的3D坐标 (shape=(3,))

    Returns:
        二面角的弧度值，范围为 [-π, π]。
        如果点共线无法定义平面，返回 None。
    """

    # 确保输入是向量并检查维度
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))

    if not all(p.shape == (3,) for p in (p0, p1, p2, p3)):
        raise ValueError(
            f"All points must be 3D coordinates (shape=(3,)). "
            f"Got shapes: {[p.shape for p in (p0, p1, p2, p3)]}"
        )

    # 计算键向量
    b0 = p0 - p1  # 从 p1 指向 p0
    b1 = p2 - p1  # 旋转轴向量
    b2 = p3 - p2  # 从 p2 指向 p3

    # 归一化旋转轴（添加数值稳定性保护）
    b1_norm = np.linalg.norm(b1)

    if b1_norm < EPSILON:
        logger.warning("Collinear points detected, cannot define dihedral angle")
        return None

    b1 = b1 / (b1_norm + EPSILON)

    # 投影向量：计算 b0 和 b2 在垂直于 b1 的平面上的投影
    v = b0 - np.dot(b0, b1) * b1        # b0 垂直于 b1 的分量
    w = b2 - np.dot(b2, b1) * b1        # b2 垂直于 b1 的分量

    # 计算角度
    x = np.dot(v, w)  # |v||w|cos(θ)
    y = np.dot(np.cross(b1, v), w)  # |v||w|sin(θ) * 轴方向

    return float(np.arctan2(y, x))


def get_moving_atoms(mol: Chem.Mol, bond_idx: int) -> tuple[list[int], int, int]:
    """
    使用 RDKit 寻找旋转键断开后的移动片段。

    该函数通过断开指定键并分析生成的片段来确定哪些原子会随旋转移动。
    采用启发式规则：原子数较少的片段作为移动部分。

    Args:
        mol: RDKit 分子对象
        bond_idx: 旋转键索引

    Returns:
        moving_atoms: 移动部分的所有原子索引列表 (list[int])
        axis_fixed: 旋转轴上属于固定部分的原子索引 (int)
        axis_moving: 旋转轴上属于移动部分的原子索引 (int)

    Raises:
        ValueError: 如果键索引超出范围
    """

    # 边界检查
    num_bonds = mol.GetNumBonds()

    if bond_idx < 0 or bond_idx >= num_bonds:
        raise ValueError(
            f"Bond index {bond_idx} out of range for molecule with {num_bonds} bonds"
        )

    bond = mol.GetBondWithIdx(bond_idx)
    u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

    # 在分子副本上断开该键
    mol_cp = Chem.RWMol(mol)
    mol_cp.RemoveBond(u, v)

    # 获取断开后的片段 (RDKit C++ 实现)
    frags = Chem.GetMolFrags(mol_cp)

    if len(frags) != 2:
        # 断键后没有分成两部分(如环上的键)，该键不可旋转
        logger.debug(
            f"Bond {bond_idx} ({u}-{v}) is not rotatable (ring bond or other constraint)"
        )
        return [], u, v

    frag0, frag1 = frags[0], frags[1]

    # 启发式规则：原子数较少的片段作为移动部分
    if len(frag0) < len(frag1):
        moving_atoms = list(frag0)
        axis_moving, axis_fixed = (u, v) if u in moving_atoms else (v, u)
        
    else:
        moving_atoms = list(frag1)
        axis_moving, axis_fixed = (u, v) if u in moving_atoms else (v, u)

    return moving_atoms, axis_fixed, axis_moving


def validate_bond_length(length: float) -> bool:
    """
    验证键长是否在物理合理范围内。

    Args:
        length: 键长（单位：Å）

    Returns:
        是否合理
    """

    return MIN_BOND_LENGTH <= length <= MAX_BOND_LENGTH


def validate_geometry(positions: np.ndarray, bonds: np.ndarray) -> bool:
    """
    验证分子几何是否物理合理。

    Args:
        positions: 原子坐标 [N, 3]
        bonds: 键连接 [M, 2]

    Returns:
        是否所有键长都在合理范围内
    """
    
    if len(bonds) == 0:
        return True

    bond_vectors = positions[bonds[:, 0]] - positions[bonds[:, 1]]
    bond_lengths = np.linalg.norm(bond_vectors, axis=1)

    valid = np.all(
        (bond_lengths >= MIN_BOND_LENGTH) & (bond_lengths <= MAX_BOND_LENGTH)
    )

    if not valid:
        min_len, max_len = bond_lengths.min(), bond_lengths.max()
        logger.warning(
            f"Unrealistic bond lengths detected: {min_len:.3f} - {max_len:.3f} Å "
            f"(expected: {MIN_BOND_LENGTH:.1f} - {MAX_BOND_LENGTH:.1f} Å)"
        )

    return bool(valid)
