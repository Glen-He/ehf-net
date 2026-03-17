"""
初始位姿工具。

负责生成训练和推理使用的解耦初始构象，
并处理配体刚体与扭转相关的初始化逻辑。
"""


import copy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from ehfnet.data.datasets.ligand_sanitize import load_ligand_mol


def _embed_molecule(mol: Chem.Mol, *, random_seed: int) -> int:
    """
    带降级策略的 ETKDG 三维坐标嵌入。

    对稠环、多手性中心等困难分子，``enforceChirality=True`` 常因约束冲突导致嵌入失败。
    依次尝试以下策略，返回第一个成功时的状态码（0 表示成功，-1 表示失败）：

    1. ``enforceChirality=True``：优先保留立体信息；
    2. ``enforceChirality=False``：放宽手性约束，覆盖稠环多手性中心分子；
    3. 不同随机种子 × ``enforceChirality=False``：应对极端困难分子。

    Args:
        mol: 已添加氢、已移除构象的 RDKit 分子对象（原地修改）。
        random_seed: 基准随机种子。

    Returns:
        int: 最终嵌入状态码；0 表示成功，-1 表示所有策略均失败。
    """
    base_params = AllChem.ETKDGv3()
    base_params.randomSeed = int(random_seed)
    base_params.useRandomCoords = True

    # 策略一：严格立体约束
    base_params.enforceChirality = True
    if AllChem.EmbedMolecule(mol, base_params) == 0:
        return 0

    # 策略二：放宽手性约束（覆盖稠环多手性中心分子）
    base_params.enforceChirality = False
    if AllChem.EmbedMolecule(mol, base_params) == 0:
        return 0

    # 策略三：换种子重试（应对极端困难构型）
    for offset in (1, 2, 3):
        base_params.randomSeed = int(random_seed) + offset
        if AllChem.EmbedMolecule(mol, base_params) == 0:
            return 0

    return -1


def remove_hs_consistent(mol: Chem.Mol) -> Chem.Mol:
    """
    与预处理管线一致地移除氢原子。

    使用 ``Chem.RemoveHs(sanitize=False)`` 保守策略，与 ``load_ligand_mol(remove_hs=True)``
    保持相同的去氢行为：保留立体氢、极性氢等特殊显式氢，
    确保生成的重原子集合与图缓存中 ``ligand_atom.pos`` 的原子数严格一致。

    Args:
        mol: 待处理的 RDKit 分子对象（通常为 ETKDG 嵌入后含氢的构象）。

    Returns:
        Chem.Mol: 返回移除普通氢后、属性缓存已更新的分子对象。
    """
    mol_no_h = Chem.RemoveHs(mol, sanitize=False)
    mol_no_h.UpdatePropertyCache(strict=False)
    return mol_no_h


def get_positions(mol: Chem.Mol) -> np.ndarray:
    """
    提取分子构象坐标。

    从当前 conformer 中读取原子三维坐标并返回副本，
    避免后续原地修改影响原始分子对象。

    Args:
        mol: 待读取或处理的 RDKit 分子对象。

    Returns:
        np.ndarray: 返回当前分子构象的原子三维坐标副本。
    """
    conf = mol.GetConformer()
    return conf.GetPositions().copy()


def generate_decoupled_ligand_positions(
    ligand_path: str,
    *,
    random_seed: int,
    remove_hs: bool = True,
) -> np.ndarray:
    """
    生成解耦初始配体坐标。

    不复用真实结合 pose，而是仅基于分子拓扑重新嵌入三维构象，
    用于训练和推理阶段构造与目标构象解耦的起始位姿。

    Args:
        ligand_path: 配体文件路径。
        random_seed: 随机种子。
        remove_hs: 是否移除分子中的显式氢原子。

    Returns:
        np.ndarray: 返回重新嵌入后得到的解耦初始配体坐标。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    mol_file = load_ligand_mol(ligand_path, remove_hs=False, require_conformer=False)
    start_mol = copy.deepcopy(mol_file)
    start_mol.RemoveAllConformers()
    start_mol = AllChem.AddHs(start_mol, addCoords=True)

    status = _embed_molecule(start_mol, random_seed=random_seed)
    if status != 0:
        raise ValueError(f"RDKit ETKDG embedding failed for ligand: {ligand_path}")

    try:
        AllChem.MMFFOptimizeMolecule(start_mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(start_mol)
        except Exception:
            pass

    if remove_hs:
        start_mol = remove_hs_consistent(start_mol)

    return get_positions(start_mol).astype(np.float32, copy=False)


def build_start_positions(
    ligand_path: str,
    *,
    random_seed: int,
    remove_hs: bool = True,
) -> np.ndarray:
    """
    构建解耦初始配体坐标。

    当前实现固定使用 `rdkit_decoupled` 路径，
    即仅基于 ligand 文件中的分子拓扑重新嵌入三维构象。

    Args:
        ligand_path: 配体文件路径。
        random_seed: 随机种子。
        remove_hs: 是否移除分子中的显式氢原子。

    Returns:
        np.ndarray: 解耦初始配体坐标。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """
    return generate_decoupled_ligand_positions(
        ligand_path,
        random_seed=random_seed,
        remove_hs=remove_hs,
    )
