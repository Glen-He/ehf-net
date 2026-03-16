"""
RMSD 计算工具。

统一封装对称感知 RMSD 与按图拆分的批处理逻辑，
为训练评估、候选生成和论文口径对齐提供稳定接口。
"""


from __future__ import annotations

import os.path as osp
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from ehfnet.data.datasets.layout import ligand_path
from ehfnet.data.datasets.ligand_sanitize import load_ligand_mol


def resolve_sample_pdb_id(sample: Any) -> str | None:
    """
    从样本对象中解析 pdb_id。

    Args:
        sample: 单图样本对象。

    Returns:
        str | None: 解析出的样本标识；若不存在则返回 `None`。
    """
    for attr_name in ("dataset_pdb_id", "pdb_id"):
        attr_value = getattr(sample, attr_name, None)
        if attr_value is not None:
            return str(attr_value)
    return None


def resolve_sample_ligand_path(sample: Any, *, dataset_raw_dir: str) -> str | None:
    """
    根据样本解析配体文件路径。

    Args:
        sample: 单图样本对象。
        dataset_raw_dir: 数据集原始样本目录。

    Returns:
        str | None: 配体文件路径；若无法定位则返回 `None`。
    """
    pdb_id = resolve_sample_pdb_id(sample)
    if pdb_id is None:
        return None
    pdb_dir = osp.join(dataset_raw_dir, pdb_id)
    return ligand_path(pdb_id, pdb_dir)


def _clone_mol_with_positions(mol: Chem.Mol, positions: torch.Tensor) -> Chem.Mol:
    """
    用给定坐标重建分子构象。

    Args:
        mol: 参考 RDKit 分子拓扑。
        positions: 原子坐标张量。

    Returns:
        Chem.Mol: 写入新构象后的分子对象。

    Raises:
        ValueError: 当原子数与坐标数不一致时抛出。
    """
    num_atoms = mol.GetNumAtoms()
    if int(positions.size(0)) != num_atoms:
        raise ValueError(
            f"Atom count mismatch for symmetry-aware RMSD: expected {num_atoms}, got {int(positions.size(0))}."
        )
    cloned = Chem.Mol(mol)
    cloned.RemoveAllConformers()
    conf = Chem.Conformer(num_atoms)
    pos_cpu = positions.detach().cpu().to(dtype=torch.float32)
    for atom_idx in range(num_atoms):
        xyz = pos_cpu[atom_idx]
        conf.SetAtomPosition(
            atom_idx,
            Point3D(float(xyz[0].item()), float(xyz[1].item()), float(xyz[2].item())),
        )
    cloned.AddConformer(conf, assignId=True)
    return cloned


def compute_symmetry_aware_rmsd(
    *,
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    ligand_file: str,
) -> float:
    """
    计算考虑对称原子等价关系的 RMSD。

    Args:
        current_pos: 当前构象坐标。
        target_pos: 目标构象坐标。
        ligand_file: 配体文件路径。

    Returns:
        float: RDKit `GetBestRMS` 计算得到的对称感知 RMSD。

    Raises:
        ValueError: 当配体文件不可用或原子数不匹配时抛出。
    """
    mol = load_ligand_mol(
        ligand_file,
        remove_hs=True,
        require_conformer=False,
    )
    ref_mol = _clone_mol_with_positions(mol, target_pos)
    pred_mol = _clone_mol_with_positions(mol, current_pos)
    return float(rdMolAlign.GetBestRMS(pred_mol, ref_mol))


def compute_batch_symmetry_aware_rmsd(
    *,
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    samples: list[Any],
    dataset_raw_dir: str,
) -> torch.Tensor:
    """
    对一个 batch 中的每个图计算对称感知 RMSD。

    Args:
        current_pos: 当前构象坐标。
        target_pos: 目标构象坐标。
        batch_idx: 原子所属图的批索引。
        samples: 对应的逐图样本列表。
        dataset_raw_dir: 数据集原始样本目录。

    Returns:
        torch.Tensor: 每个图对应一个对称感知 RMSD 标量。

    Raises:
        ValueError: 当样本缺失配体文件或图级原子切片为空时抛出。
    """
    rmsd_values: list[float] = []
    for graph_idx, sample in enumerate(samples):
        graph_mask = batch_idx == graph_idx
        if not bool(graph_mask.any()):
            raise ValueError(
                f"Missing ligand atoms for graph index {graph_idx} when computing symmetry-aware RMSD."
            )
        ligand_file = resolve_sample_ligand_path(sample, dataset_raw_dir=dataset_raw_dir)
        if ligand_file is None:
            pdb_id = resolve_sample_pdb_id(sample)
            raise ValueError(
                f"Missing ligand file for symmetry-aware RMSD on sample {pdb_id!r}."
            )
        graph_rmsd = compute_symmetry_aware_rmsd(
            current_pos=current_pos[graph_mask],
            target_pos=target_pos[graph_mask],
            ligand_file=ligand_file,
        )
        rmsd_values.append(graph_rmsd)
    return torch.tensor(
        rmsd_values,
        device=current_pos.device,
        dtype=current_pos.dtype,
    )
