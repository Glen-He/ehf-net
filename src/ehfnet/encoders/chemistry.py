"""
化学基础定义

包含元素和氨基酸残基的枚举定义
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Element(Enum):
    """
    元素枚举，包含常见元素的物理化学性质
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
        回退逻辑：支持通过原子序数或符号查找元素
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
        """

        if isinstance(value, cls):
            return value

        if value is None:
            return cls.UNK

        try:
            return cls(value)

        except Exception:
            return cls.UNK


    # 元素定义: (idx, 原子序数, 范德华半径, 原子量, 电负性, 电子亲和能, 第一电离能)
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
    # Metals commonly found in PDBBind
    Mg = (12, 12, 173.0, 24.305, 1.31, 0.0, 7.646)
    Ca = (13, 20, 231.0, 40.078, 1.00, 0.02455, 6.113)
    Mn = (14, 25, 205.0, 54.938, 1.55, 0.0, 7.434)
    Fe = (15, 26, 204.0, 55.845, 1.83, 0.151, 7.902)
    Zn = (16, 30, 210.0, 65.38, 1.65, 0.0, 9.394)
    Co = (17, 27, 200.0, 58.933, 1.88, 0.662, 7.881)
    Ni = (18, 28, 163.0, 58.693, 1.91, 1.157, 7.640)
    Cu = (19, 29, 140.0, 63.546, 1.90, 1.236, 7.726)
    # UNK 连续特征使用已知元素的中位量级，避免与真实元素分布出现百倍量纲跳变。
    UNK = (20, 0, 204.5, 33.755, 2.115, 0.9556, 9.877)


class ResidueType(Enum):
    """
    氨基酸残基类型，包含原子排列和扭转角信息
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
        """

        if isinstance(value, cls):
            return value

        if value is None:
            return cls.UNK

        try:
            return cls(value)

        except Exception:
            return cls.UNK


    # 残基定义: (单字母码, 索引, 对称原子映射, 原子14表示, 侧链扭转角, 周期性掩码)
    ALA = (
        "A",
        0,
        {},
        ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""),
        (),
        (0.0, 0.0, 0.0, 0.0),
    )
    ARG = (
        "R",
        1,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""),
        (
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"),
            ("CG", "CD", "NE", "CZ"),
        ),
        (0.0, 0.0, 0.0, 0.0),
    )
    ASN = (
        "N",
        2,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    ASP = (
        "D",
        3,
        {"OD1": "OD2"},
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        (0.0, 1.0, 0.0, 0.0),
    )
    CYS = (
        "C",
        4,
        {},
        ("N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""),
        (("N", "CA", "CB", "SG"),),
        (0.0, 0.0, 0.0, 0.0),
    )
    GLN = (
        "Q",
        5,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    GLU = (
        "E",
        6,
        {"OE1": "OE2"},
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
        (0.0, 0.0, 1.0, 0.0),
    )
    GLY = (
        "G",
        7,
        {},
        ("N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""),
        (),
        (0.0, 0.0, 0.0, 0.0),
    )
    HIS = (
        "H",
        8,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    ILE = (
        "I",
        9,
        {},
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    LEU = (
        "L",
        10,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    LYS = (
        "K",
        11,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""),
        (
            ("N", "CA", "CB", "CG"),
            ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"),
            ("CG", "CD", "CE", "NZ"),
        ),
        (0.0, 0.0, 0.0, 0.0),
    )
    MET = (
        "M",
        12,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")),
        (0.0, 0.0, 0.0, 0.0),
    )
    PHE = (
        "F",
        13,
        {"CD1": "CD2", "CE1": "CE2"},
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        (0.0, 1.0, 0.0, 0.0),
    )
    PRO = (
        "P",
        14,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
        (0.0, 0.0, 0.0, 0.0),
    )
    SER = (
        "S",
        15,
        {},
        ("N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""),
        (("N", "CA", "CB", "OG"),),
        (0.0, 0.0, 0.0, 0.0),
    )
    THR = (
        "T",
        16,
        {},
        ("N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""),
        (("N", "CA", "CB", "OG1"),),
        (0.0, 0.0, 0.0, 0.0),
    )
    TRP = (
        "W",
        17,
        {},
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        (0.0, 0.0, 0.0, 0.0),
    )
    TYR = (
        "Y",
        18,
        {"CD1": "CD2", "CE1": "CE2"},
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""),
        (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
        (0.0, 1.0, 0.0, 0.0),
    )
    VAL = (
        "V",
        19,
        {},
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""),
        (("N", "CA", "CB", "CG1"),),
        (0.0, 0.0, 0.0, 0.0),
    )
    UNK = (
        "X",
        20,
        {},
        ("", "", "", "", "", "", "", "", "", "", "", "", "", ""),
        (),
        (0.0, 0.0, 0.0, 0.0),
    )


@dataclass(frozen=True)
class ResidueTypeResolution:
    """
    残基名解析结果。

    `source` 取值：
    - "canonical": 原始残基名已是标准残基
    - "alias": 通过别名映射到了标准残基
    - "unknown": 仍无法解析，最终会落到 UNK / X
    """

    original_resname: str
    normalized_resname: str
    residue_type: ResidueType
    source: str


_STRUCTURE_RESIDUE_ALIASES: dict[str, str] = {
    # 常见质子化 / 力场命名差异，heavy-atom 拓扑与标准残基一致。
    "ASH": "ASP",
    "CYM": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "LYN": "LYS",
}

_ESM_SEQUENCE_RESIDUE_ALIASES: dict[str, str] = {
    **_STRUCTURE_RESIDUE_ALIASES,
    # 常见修饰残基，序列上退回到母体氨基酸，避免直接变成 X。
    "ALY": "LYS",
    "CME": "CYS",
    "CSD": "CYS",
    "CSO": "CYS",
    "CSS": "CYS",
    "CSX": "CYS",
    "FME": "MET",
    "HYP": "PRO",
    "KCX": "LYS",
    "LLP": "LYS",
    "MLY": "LYS",
    "MSE": "MET",
    "OCS": "CYS",
    "PTR": "TYR",
    "PYL": "LYS",
    "SEC": "CYS",
    "SEP": "SER",
    "TPO": "THR",
}


def _normalize_residue_name(value: Any) -> str:
    if isinstance(value, ResidueType):
        return value.three_letter.upper()
    return str(value or "").strip().upper()


def _resolve_residue_type(
    value: Any,
    *,
    alias_map: dict[str, str],
) -> ResidueTypeResolution:
    original = _normalize_residue_name(value)

    direct_match = ResidueType.safe_get(original)
    if direct_match != ResidueType.UNK:
        return ResidueTypeResolution(
            original_resname=original,
            normalized_resname=direct_match.three_letter,
            residue_type=direct_match,
            source="canonical",
        )

    alias_name = alias_map.get(original, "")
    alias_match = ResidueType.safe_get(alias_name)
    if alias_name and alias_match != ResidueType.UNK:
        return ResidueTypeResolution(
            original_resname=original,
            normalized_resname=alias_match.three_letter,
            residue_type=alias_match,
            source="alias",
        )

    return ResidueTypeResolution(
        original_resname=original,
        normalized_resname=ResidueType.UNK.three_letter,
        residue_type=ResidueType.UNK,
        source="unknown",
    )


def resolve_protein_residue_type(value: Any) -> ResidueTypeResolution:
    """
    解析结构特征使用的残基类型。

    这里只接受对 heavy-atom 拓扑基本兼容的命名别名，避免把修饰残基
    强行当成标准残基后，扭转角 / atom14 掩码出现更隐蔽的错配。
    """

    return _resolve_residue_type(value, alias_map=_STRUCTURE_RESIDUE_ALIASES)


def resolve_esm_residue_type(value: Any) -> ResidueTypeResolution:
    """
    解析 ESM 序列使用的残基类型。

    对常见修饰残基先退回母体氨基酸，尽量避免直接生成 X。
    """

    return _resolve_residue_type(value, alias_map=_ESM_SEQUENCE_RESIDUE_ALIASES)
