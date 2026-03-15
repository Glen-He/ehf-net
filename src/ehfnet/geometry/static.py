"""
静态几何工具。

负责二面角、旋转键和固定几何量计算，
为初始化与扭转建模提供基础能力。
"""


import logging
import numpy as np

from rdkit import Chem

logger = logging.getLogger(__name__)


MIN_BOND_LENGTH = 0.5
MAX_BOND_LENGTH = 3.0
EPSILON = 1e-8


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
        p0: 定义第一个平面的起始三维点坐标。
        p1: 同时属于两个平面的第一个旋转轴端点坐标。
        p2: 同时属于两个平面的第二个旋转轴端点坐标。
        p3: 定义第二个平面的末端三维点坐标。

    Returns:
        float | None: 返回计算得到的二面角弧度；当几何关系退化而无法定义二面角时返回 `None`。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))

    if not all(p.shape == (3,) for p in (p0, p1, p2, p3)):
        raise ValueError(
            f"All points must be 3D coordinates (shape=(3,)). "
            f"Got shapes: {[p.shape for p in (p0, p1, p2, p3)]}"
        )

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)

    if b1_norm < EPSILON:
        logger.warning("Collinear points detected, cannot define dihedral angle")
        return None

    b1 = b1 / (b1_norm + EPSILON)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return float(np.arctan2(y, x))


def _fragment_signature(
    fragment: tuple[int, ...],
    canonical_ranks: list[int],
) -> tuple[tuple[int, int], ...]:
    """
    为断键后的片段构造稳定、与输入原子顺序弱相关的签名。

    Args:
        fragment: 断键后某个片段包含的原子索引序列。
        canonical_ranks: 与原子索引对齐的规范排序结果列表。

    Returns:
        tuple[tuple[int, int], ...]: 返回描述片段组成与排序信息的稳定签名元组。

    Raises:
        ValueError: 当片段中存在越界原子索引时抛出。
    """
    invalid_indices = [idx for idx in fragment if idx < 0 or idx >= len(canonical_ranks)]
    if invalid_indices:
        raise ValueError(
            "Fragment contains atom indices outside the canonical rank range: "
            f"{invalid_indices}."
        )
    return tuple(sorted((canonical_ranks[idx], idx) for idx in fragment))


def get_moving_atoms(
    mol: Chem.Mol,
    bond_idx: int,
    *,
    canonical_ranks: list[int] | None = None,
) -> tuple[list[int], int, int]:
    """
    使用 RDKit 寻找旋转键断开后的移动片段。

    该函数通过断开指定键并分析生成的片段来确定哪些原子会随旋转移动。
    采用启发式规则：原子数较少的片段作为移动部分。


    Args:
        mol: 待读取或处理的 RDKit 分子对象。
        bond_idx: 目标键在分子中的键索引。
        canonical_ranks: RDKit 计算得到的规范原子排序。


    Returns:
        tuple[list[int], int, int]: 移动原子索引列表、轴固定原子索引、轴移动原子索引。

    Raises:
        ValueError: 如果键索引超出范围
    """

    num_bonds = mol.GetNumBonds()

    if bond_idx < 0 or bond_idx >= num_bonds:
        raise ValueError(
            f"Bond index {bond_idx} out of range for molecule with {num_bonds} bonds"
        )

    bond = mol.GetBondWithIdx(bond_idx)
    u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

    mol_cp = Chem.RWMol(mol)
    mol_cp.RemoveBond(u, v)

    frags = Chem.GetMolFrags(mol_cp)

    if len(frags) != 2:
        logger.debug(
            f"Bond {bond_idx} ({u}-{v}) is not rotatable (ring bond or other constraint)"
        )
        return [], u, v

    frag0, frag1 = frags[0], frags[1]

    if canonical_ranks is None:
        canonical_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    if len(frag0) < len(frag1):
        moving_atoms = list(frag0)
        axis_moving, axis_fixed = (u, v) if u in moving_atoms else (v, u)

    elif len(frag1) < len(frag0):
        moving_atoms = list(frag1)
        axis_moving, axis_fixed = (u, v) if u in moving_atoms else (v, u)

    else:
        frag0_sig = _fragment_signature(frag0, canonical_ranks)
        frag1_sig = _fragment_signature(frag1, canonical_ranks)
        moving_fragment = frag0 if frag0_sig <= frag1_sig else frag1
        moving_atoms = list(moving_fragment)
        axis_moving, axis_fixed = (u, v) if u in moving_atoms else (v, u)

    return moving_atoms, axis_fixed, axis_moving
