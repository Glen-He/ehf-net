"""
配体初始构象生成工具。

生成与真实结合构象解耦的 RDKit 3D conformer，供训练/验证/测试的随机起点使用。
"""

from __future__ import annotations

import copy

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from ehfnet.datasets.ligand_sanitize import load_ligand_mol

def remove_all_hs_safe(mol: Chem.Mol) -> Chem.Mol:
    mol_no_h = Chem.RemoveHs(mol, sanitize=True)
    return AllChem.RemoveAllHs(mol_no_h)


def get_positions(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return conf.GetPositions().copy()


def generate_decoupled_ligand_positions(
    ligand_path: str,
    *,
    random_seed: int,
    remove_hs: bool = True,
) -> np.ndarray:
    """
    生成与真实结合 pose 解耦的 RDKit conformer。

    这里不使用真实 pose 做插值起点，只从分子拓扑重新嵌入 3D 构象。
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
