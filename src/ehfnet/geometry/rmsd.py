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

    使用分子自身的自同构映射（automorphism）枚举所有对称等价原子排列，
    对每种排列在不施加额外平移或旋转的前提下计算原始坐标 RMSD，
    返回所有排列中的最小值。

    此口径与 DiffDock 等对接论文保持一致：只允许对称原子重标记，
    不允许全局叠合，确保评估结果反映真实的三维定位精度。

    Args:
        current_pos: 当前构象坐标，shape [N, 3]。
        target_pos: 目标构象坐标，shape [N, 3]。
        ligand_file: 配体文件路径，用于读取分子拓扑与对称信息。

    Returns:
        float: 最优对称映射下的原始坐标 RMSD（Å）。

    Raises:
        ValueError: 当配体文件不可用或原子数不匹配时抛出。
    """
    mol = load_ligand_mol(
        ligand_file,
        remove_hs=True,
        require_conformer=False,
    )
    num_atoms = mol.GetNumAtoms()
    if int(current_pos.size(0)) != num_atoms or int(target_pos.size(0)) != num_atoms:
        raise ValueError(
            f"Atom count mismatch for symmetry-aware RMSD: "
            f"mol={num_atoms}, current={int(current_pos.size(0))}, target={int(target_pos.size(0))}."
        )

    # 枚举分子自同构（对称等价原子映射），不做额外平移/旋转
    mappings = mol.GetSubstructMatches(mol, useChirality=False)
    pred_cpu = current_pos.detach().cpu().to(dtype=torch.float64)
    ref_cpu = target_pos.detach().cpu().to(dtype=torch.float64)

    best_rmsd = float("inf")
    for mapping in mappings:
        reordered = pred_cpu[list(mapping)]
        diff = reordered - ref_cpu
        rmsd = float(torch.sqrt((diff ** 2).sum(dim=-1).mean()).item())
        if rmsd < best_rmsd:
            best_rmsd = rmsd
    return best_rmsd


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
