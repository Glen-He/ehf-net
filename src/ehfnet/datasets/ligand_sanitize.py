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


class LigandSanitizationError(ValueError):
    """
    RDKit sanitize 失败时抛出的结构化异常。
    """

    def __init__(
        self,
        ligand_path: str | Path,
        *,
        full_flag: int,
        partial_flag: int | None = None,
        allow_partial_fallback: bool,
    ) -> None:
        self.ligand_path = str(ligand_path)
        self.full_flag = int(full_flag)
        self.partial_flag = None if partial_flag is None else int(partial_flag)
        self.allow_partial_fallback = bool(allow_partial_fallback)

        detail = f"full_flag={self.full_flag}"
        if self.partial_flag is not None:
            detail = f"{detail}, partial_flag={self.partial_flag}"
        if not self.allow_partial_fallback:
            detail = f"{detail}. Partial fallback is disabled."

        super().__init__(f"RDKit sanitization rejected ligand {self.ligand_path}: {detail}")


def _set_sanitize_props(
    mol: Chem.Mol,
    *,
    mode: str,
    full_flag: int,
    partial_flag: int | None = None,
) -> Chem.Mol:
    mol.SetProp("_ehfnet_sanitize_mode", mode)
    mol.SetProp("_ehfnet_full_sanitize_flag", str(int(full_flag)))
    if partial_flag is not None:
        mol.SetProp("_ehfnet_partial_sanitize_flag", str(int(partial_flag)))
    return mol


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


def sanitize_ligand_mol_with_mode(
    mol: Chem.Mol,
    ligand_path: str | Path,
    *,
    allow_partial_fallback: bool = False,
) -> Chem.Mol:
    """
    统一的 ligand sanitize 策略。

    默认只接受 full sanitize 成功的分子；partial fallback 仅作为显式 opt-in。
    """
    path = str(ligand_path)
    full_result = Chem.SanitizeMol(mol, catchErrors=True)
    if full_result == Chem.SanitizeFlags.SANITIZE_NONE:
        return _set_sanitize_props(
            mol,
            mode="full",
            full_flag=int(full_result),
        )

    if not allow_partial_fallback:
        raise LigandSanitizationError(
            path,
            full_flag=int(full_result),
            allow_partial_fallback=False,
        )

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
        raise LigandSanitizationError(
            path,
            full_flag=int(full_result),
            partial_flag=int(fallback_result),
            allow_partial_fallback=True,
        )
    return _set_sanitize_props(
        mol,
        mode="partial",
        full_flag=int(full_result),
        partial_flag=int(fallback_result),
    )


def load_ligand_mol(
    ligand_path: str | Path,
    *,
    remove_hs: bool = False,
    require_conformer: bool = False,
    allow_partial_fallback: bool = False,
) -> Chem.Mol:
    """
    统一读取并清洗 ligand。

    默认仅接受 full sanitize 成功的分子；如果确有需要，可显式开启 partial fallback。
    """

    path = str(ligand_path)
    mol = read_ligand_file(path, remove_hs=remove_hs)
    if mol is None:
        raise ValueError(f"Failed to load ligand: {path}")

    mol = sanitize_ligand_mol_with_mode(
        mol,
        path,
        allow_partial_fallback=allow_partial_fallback,
    )

    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol.UpdatePropertyCache(strict=False)

    if require_conformer and mol.GetNumConformers() == 0:
        raise ValueError(f"Ligand has no conformer: {path}")

    return mol
