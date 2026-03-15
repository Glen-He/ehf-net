"""
配体清洗工具。

负责读取配体文件、执行 RDKit 标准化与清洗，
并产出后续特征化和构图可直接使用的分子对象。
"""


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
    配体清洗异常。

    用于封装 RDKit 清洗阶段的结构化失败信息，
    便于上层在记录日志或写入元数据时区分具体错误来源。
    """

    def __init__(
        self,
        ligand_path: str | Path,
        *,
        full_flag: int,
        partial_flag: int | None = None,
        allow_partial_fallback: bool,
    ) -> None:
        """
        初始化配体清洗异常。

        记录清洗阶段的 full/partial 结果与回退策略，
        便于上层统一报告配体读取失败原因。

        Args:
            ligand_path: 配体文件路径。
            full_flag: full sanitize 分支是否成功。
            partial_flag: partial sanitize 分支是否成功。
            allow_partial_fallback: 是否允许在 full sanitize 失败时回退到 partial 策略。
        """
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
    读取原始配体文件。

    负责从 SDF 或 MOL2 中加载分子对象，
    但不在这里执行标准化或 sanitize，便于上层自行控制清洗策略。

    Args:
        ligand_path: 配体文件路径。
        remove_hs: 是否移除分子中的显式氢原子。

    Returns:
        Chem.Mol | None: 返回原始读取到的分子对象；若文件内容为空或无法解析则返回 `None`。
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
    按指定模式清洗配体分子。

    统一封装 full sanitize 与可选 partial fallback 的策略选择，
    让上层读取逻辑能够在严格模式和显式降级模式之间切换。

    Args:
        mol: 待读取或处理的 RDKit 分子对象。
        ligand_path: 配体文件路径。
        allow_partial_fallback: 是否允许在 full sanitize 失败时回退到 partial 策略。

    Returns:
        Chem.Mol: 返回按指定 sanitize 策略处理后的配体分子对象。

    Raises:
        LigandSanitizationError: 函数内部在对应异常条件下会抛出该异常。
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
    读取并清洗配体分子。

    负责串联文件读取、sanitize 策略选择和 conformer 校验，
    为预处理和初始化流程返回可直接使用的配体对象。

    Args:
        ligand_path: 配体文件路径。
        remove_hs: 是否移除分子中的显式氢原子。
        require_conformer: 是否要求输入分子必须带有三维构象。
        allow_partial_fallback: 是否允许在 full sanitize 失败时回退到 partial 策略。

    Returns:
        Chem.Mol: 返回完成读取、清洗和可选 conformer 校验后的配体分子对象。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
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
