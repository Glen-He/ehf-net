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


def remove_all_hs_safe(mol: Chem.Mol) -> Chem.Mol:
    """
    安全移除分子中的氢原子。

    先执行 RDKit 的标准去氢流程，再移除剩余显式氢，
    为后续生成仅含重原子的初始坐标提供稳定分子对象。

    Args:
        mol: 待读取或处理的 RDKit 分子对象。

    Returns:
        Chem.Mol: 返回移除氢原子后的 RDKit 分子对象。
    """
    mol_no_h = Chem.RemoveHs(mol, sanitize=True)
    return AllChem.RemoveAllHs(mol_no_h)


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

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True
    params.enforceChirality = True

    status = AllChem.EmbedMolecule(start_mol, params)
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
        start_mol = remove_all_hs_safe(start_mol)

    return get_positions(start_mol).astype(np.float32, copy=False)
