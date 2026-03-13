"""
配体读取与 RDKit 清洗工具。

统一项目内所有 ligand chemistry 入口，避免图构建、初始构象生成、
scaffold split 各自维护不同的 sanitize 语义。
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem


logger = logging.getLogger(__name__)


FALLBACK_SANITIZE_OPS = (
    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
)


def read_ligand_file(
    ligand_path: str | Path,
    *,
    remove_hs: bool = False,
) -> Chem.Mol | None:
    """
    从 SDF/MOL2 读取配体文件，不在这里做 sanitize。
    """

    path = str(ligand_path)

    if path.endswith(".mol2"):
        return Chem.MolFromMol2File(path, sanitize=False, removeHs=remove_hs)

    supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=remove_hs)
    return supplier[0] if len(supplier) > 0 else None


def sanitize_ligand_mol(mol: Chem.Mol, ligand_path: str | Path) -> Chem.Mol:
    """
    统一的 ligand sanitize 策略。

    full sanitize 失败后，保留环/芳香性/共轭/杂化等关键语义做 partial sanitize。
    若仍失败，直接拒绝该分子，避免把脏 chemistry 带入下游。
    """

    path = str(ligand_path)
    full_result = Chem.SanitizeMol(mol, catchErrors=True)
    if full_result == Chem.SanitizeFlags.SANITIZE_NONE:
        return mol

    logger.warning(
        "Full RDKit sanitization failed for %s with flag=%d. Falling back to partial sanitization.",
        path,
        int(full_result),
    )
    mol.UpdatePropertyCache(strict=False)
    fallback_result = Chem.SanitizeMol(
        mol,
        sanitizeOps=FALLBACK_SANITIZE_OPS,
        catchErrors=True,
    )
    if fallback_result != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError(
            "RDKit sanitization failed for "
            f"{path}: full_flag={int(full_result)}, partial_flag={int(fallback_result)}"
        )
    return mol


def load_ligand_mol(
    ligand_path: str | Path,
    *,
    remove_hs: bool = False,
    require_conformer: bool = False,
) -> Chem.Mol:
    """
    统一读取并清洗 ligand。
    """

    path = str(ligand_path)
    mol = read_ligand_file(path, remove_hs=remove_hs)
    if mol is None:
        raise ValueError(f"Failed to load ligand: {path}")

    mol = sanitize_ligand_mol(mol, path)

    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol.UpdatePropertyCache(strict=False)

    if require_conformer and mol.GetNumConformers() == 0:
        raise ValueError(f"Ligand has no conformer: {path}")

    return mol
