"""
化学特征常量。

集中维护元素、残基和离散类别映射，
供蛋白与配体编码流程共享使用。
"""


from dataclasses import dataclass
from enum import Enum
from typing import Any


class Element(Enum):
    """
    元素信息枚举。

    集中描述常见元素的编号、符号和基础理化属性，
    供蛋白与配体特征编码共享使用。
    """

    def __init__(
        self,
        idx: int,
        atomic_number: int,
        vdw_radius_mm3: float,
        atomic_weight: float,
        en_pauling: float,
        electron_affinity: float,
        first_ionization_energy: float,
    ):
        """
        初始化元素枚举项。

        为每个元素写入原子序数、半径和理化属性，
        供后续特征编码直接读取。

        Args:
            idx: 当前访问的样本索引。
            atomic_number: 元素的原子序数。
            vdw_radius_mm3: MM3 力场使用的范德华半径。
            atomic_weight: 元素的原子量。
            en_pauling: Pauling 电负性。
            electron_affinity: 电子亲和能。
            first_ionization_energy: 第一电离能。
        """
        self.symbol = self.name
        self.idx = idx
        self.atomic_number = atomic_number
        self.vdw_radius_mm3 = vdw_radius_mm3
        self.atomic_weight = atomic_weight
        self.en_pauling = en_pauling
        self.electron_affinity = electron_affinity
        self.first_ionization_energy = first_ionization_energy

    @classmethod
    def _missing_(cls, value: Any) -> "Element":
        """
        回退逻辑：支持通过原子序数或符号查找元素。

        Returns:
            Element: 返回按输入值解析得到的元素枚举；无法识别时回退为 `Element.UNK`。
        """

        if not hasattr(cls, "_lookup_cache"):
            cls._int_lookup: dict[int, Element] = {e.atomic_number: e for e in cls}
            cls._str_lookup: dict[str, Element] = {e.symbol.upper(): e for e in cls}
            cls._lookup_cache = True

        if isinstance(value, int):
            return cls._int_lookup.get(value, cls.UNK)
        elif isinstance(value, str):
            return cls._str_lookup.get(value.upper(), cls.UNK)

        return super()._missing_(value)

    @classmethod
    def safe_get(cls, value: Any) -> "Element":
        """
        安全解析元素，失败时返回 UNK

        Args:
            value: 待处理或校验的输入值。

        Returns:
            Element: 返回安全解析后的元素枚举；异常输入统一回退为 `Element.UNK`。
        """

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNK

        try:
            return cls(value)
        except Exception:
            return cls.UNK

    H = (0, 1, 162.0, 1.008, 2.20, 0.7542, 13.5984)
    B = (1, 5, 215.0, 10.81, 2.04, 0.2797, 8.2980)
    C = (2, 6, 204.0, 12.011, 2.55, 1.2621, 11.2603)
    N = (3, 7, 193.0, 14.007, 3.04, -1.4000, 14.5341)
    O = (4, 8, 182.0, 15.999, 3.44, 1.4611, 13.6181)
    F = (5, 9, 171.0, 18.998, 3.98, 3.4012, 17.4228)
    Si = (6, 14, 229.0, 28.086, 1.90, 1.3895, 8.1517)
    P = (7, 15, 222.0, 30.974, 2.19, 0.7466, 10.4867)
    S = (8, 16, 215.0, 32.06, 2.58, 2.0771, 10.3600)
    Cl = (9, 17, 207.0, 35.45, 3.16, 3.6127, 12.9676)
    Br = (10, 35, 222.0, 79.904, 2.96, 3.3636, 11.8138)
    I = (11, 53, 236.0, 126.904, 2.66, 3.0590, 10.4512)
    Mg = (12, 12, 173.0, 24.305, 1.31, 0.0, 7.646)
    Ca = (13, 20, 231.0, 40.078, 1.00, 0.02455, 6.113)
    Mn = (14, 25, 205.0, 54.938, 1.55, 0.0, 7.434)
    Fe = (15, 26, 204.0, 55.845, 1.83, 0.151, 7.902)
    Zn = (16, 30, 210.0, 65.38, 1.65, 0.0, 9.394)
    Co = (17, 27, 200.0, 58.933, 1.88, 0.662, 7.881)
    Ni = (18, 28, 163.0, 58.693, 1.91, 1.157, 7.640)
    Cu = (19, 29, 140.0, 63.546, 1.90, 1.236, 7.726)
    UNK = (20, 0, 204.5, 33.755, 2.115, 0.9556, 9.877)


class ResidueType(Enum):
    """
    残基类型定义。

    描述标准氨基酸残基的内部编码、原子布局和扭转信息，
    为蛋白编码和几何计算提供统一语义。
    """

    def __init__(
        self,
        one_letter: str,
        index: int,
        atom_swap: dict[str, str],
        atom14: tuple[str, ...],
        chi_angle: tuple[tuple[str, str, str, str], ...],
        chi_pi_periodic: tuple[float, float, float, float],
    ):
        """
        初始化残基类型定义。

        写入残基的编码、原子布局和扭转角信息，
        供蛋白编码与几何模块共享使用。

        Args:
            one_letter: 氨基酸的一字母缩写。
            index: 索引。
            atom_swap: 原子swap。
            atom14: 残基的 Atom14 原子命名布局。
            chi_angle: 残基可用的 chi 扭转角定义。
            chi_pi_periodic: 各 chi 扭转角是否具有 pi 周期性。
        """
        self.three_letter = self.name
        self.one_letter = one_letter
        self.index = index
        self.atom_swap = atom_swap
        self.atom14 = atom14
        self.chi_angle = chi_angle
        self.chi_pi_periodic = chi_pi_periodic

    @classmethod
    def _missing_(cls, value: Any) -> "ResidueType":
        """
        回退逻辑：支持单字母或三字母码查找

        Returns:
            ResidueType: 返回按输入值解析得到的残基类型；无法识别时回退为 `ResidueType.UNK`。
        """

        if not hasattr(cls, "_lookup_cache"):
            lookup_cache: dict[str, ResidueType] = {}

            for member in cls:
                lookup_cache[member.one_letter.upper()] = member
                lookup_cache[member.three_letter.upper()] = member

            cls._lookup_cache = lookup_cache

        if isinstance(value, str):
            return cls._lookup_cache.get(value.upper(), ResidueType.UNK)

        return super()._missing_(value)

    @classmethod
    def safe_get(cls, value: Any) -> "ResidueType":
        """
        安全解析残基，失败时返回 UNK

        Args:
            value: 待处理或校验的输入值。

        Returns:
            ResidueType: 返回安全解析后的残基类型；异常输入统一回退为 `ResidueType.UNK`。
        """

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNK

        try:
            return cls(value)
        except Exception:
            return cls.UNK

    ALA = ("A", 0, {}, ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""), (), (0.0, 0.0, 0.0, 0.0))
    ARG = ("R", 1, {}, ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")), (0.0, 0.0, 0.0, 0.0))
    ASN = ("N", 2, {}, ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")), (0.0, 0.0, 0.0, 0.0))
    ASP = ("D", 3, {"OD1": "OD2"}, ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")), (0.0, 1.0, 0.0, 0.0))
    CYS = ("C", 4, {}, ("N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""), (("N", "CA", "CB", "SG"),), (0.0, 0.0, 0.0, 0.0))
    GLN = ("Q", 5, {}, ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")), (0.0, 0.0, 0.0, 0.0))
    GLU = ("E", 6, {"OE1": "OE2"}, ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")), (0.0, 0.0, 1.0, 0.0))
    GLY = ("G", 7, {}, ("N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""), (), (0.0, 0.0, 0.0, 0.0))
    HIS = ("H", 8, {}, ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")), (0.0, 0.0, 0.0, 0.0))
    ILE = ("I", 9, {}, ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""), (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")), (0.0, 0.0, 0.0, 0.0))
    LEU = ("L", 10, {}, ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")), (0.0, 0.0, 0.0, 0.0))
    LYS = ("K", 11, {}, ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")), (0.0, 0.0, 0.0, 0.0))
    MET = ("M", 12, {}, ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")), (0.0, 0.0, 0.0, 0.0))
    PHE = ("F", 13, {}, ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")), (0.0, 0.0, 0.0, 0.0))
    PRO = ("P", 14, {}, ("N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")), (0.0, 0.0, 0.0, 0.0))
    SER = ("S", 15, {}, ("N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""), (("N", "CA", "CB", "OG"),), (0.0, 0.0, 0.0, 0.0))
    THR = ("T", 16, {}, ("N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""), (("N", "CA", "CB", "OG1"),), (0.0, 0.0, 0.0, 0.0))
    TRP = ("W", 17, {}, ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")), (0.0, 0.0, 0.0, 0.0))
    TYR = ("Y", 18, {}, ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""), (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")), (0.0, 0.0, 0.0, 0.0))
    VAL = ("V", 19, {}, ("N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""), (("N", "CA", "CB", "CG1"),), (0.0, 0.0, 0.0, 0.0))
    UNK = ("X", 20, {}, ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""), (), (0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class ResidueResolution:
    """
    残基解析结果。

    封装残基名映射后的目标类型和解析来源，
    便于上层同时使用解析结果与诊断信息。
    """

    residue_type: ResidueType
    original_resname: str
    normalized_resname: str
    source: str


_RESIDUE_ALIASES: dict[str, str] = {
    "ASH": "ASP",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "MSE": "MET",
    "SEC": "CYS",
    "PYL": "LYS",
    "CYX": "CYS",
}


def _normalize_residue_name(resname: str | None) -> str:
    """
    规范化残基三字母名称。

    将原始残基名统一转为去空白、去数字后缀、全大写的三字母形式，
    供蛋白编码与 ESM 序列构建共享同一套残基语义入口。

    Args:
        resname: 原始残基名称，可能来自 PDB 或解析库对象。

    Returns:
        str: 规范化后的残基名称；无法解析时返回 `UNK`。
    """

    if resname is None:
        return "UNK"

    normalized = str(resname).strip().upper()
    if not normalized:
        return "UNK"

    normalized = "".join(ch for ch in normalized if ch.isalpha())
    if not normalized:
        return "UNK"

    return normalized[:3]


def resolve_protein_residue_type(resname: str | None) -> ResidueResolution:
    """
    解析蛋白残基类型。

    将结构文件中的残基名映射到内部 `ResidueType`，
    并保留映射来源以支持编码和调试。

    Args:
        resname: 待解析的残基名称。

    Returns:
        ResidueResolution: 返回解析后的残基类型、规范化名称和来源信息。
    """

    normalized = _normalize_residue_name(resname)
    if normalized in ResidueType.__members__:
        residue_type = ResidueType[normalized]
        return ResidueResolution(
            residue_type=residue_type,
            original_resname=normalized,
            normalized_resname=normalized,
            source="native",
        )

    alias_target = _RESIDUE_ALIASES.get(normalized)
    if alias_target is not None:
        return ResidueResolution(
            residue_type=ResidueType[alias_target],
            original_resname=normalized,
            normalized_resname=alias_target,
            source="alias",
        )
    return ResidueResolution(
        residue_type=ResidueType.UNK,
        original_resname=normalized,
        normalized_resname="UNK",
        source="unknown",
    )


def resolve_esm_residue_type(resname: str | None) -> ResidueResolution:
    """
    解析 ESM 用残基类型。

    提供更适合序列构建场景的残基解析入口，
    用于在提取蛋白序列时保持与编码侧一致的残基语义。

    Args:
        resname: 待解析的残基名称。

    Returns:
        ResidueResolution: 返回适用于 ESM 序列构建的残基解析结果。
    """

    return resolve_protein_residue_type(resname)
