"""
特征规格定义。

集中声明分类特征与连续特征的 schema，
保证编码维度和字段顺序在全项目内保持一致。
"""


from typing import NamedTuple

from ehfnet.data.featurizers.chemistry import ResidueType


class CatFeature(NamedTuple):
    """
    分类特征配置。

    描述离散特征的名称、词表大小和嵌入维度，
    用于统一声明图节点的分类输入字段。
    """

    name: str
    num_embeddings: int
    embed_dim: int
    description: str


class ContFeature(NamedTuple):
    """
    连续特征配置。

    描述连续特征的名称和归一化相关约定，
    用于统一声明图节点的连续输入字段。
    """

    name: str
    description: str


LIGAND_ATOM_CAT_SCHEMA = [
    CatFeature("num_hs", 6, 4, "氢原子数量 [0-5]"),
    CatFeature("degree", 8, 8, "原子度 (连接数) [0-7]"),
    CatFeature("implicit_valence", 7, 4, "隐式化合价 [0-6] (硫/磷可达6价)"),
    CatFeature("atomic_idx", 24, 16, "原子类型索引 (基于 Element 枚举)"),
    CatFeature("formal_charge", 9, 8, "形式电荷 [-4, +4] 映射到 [0, 8]"),
    CatFeature("hybridization", 9, 8, "杂化类型 (SP, SP2, SP3等)"),
    CatFeature("num_radical_electrons", 4, 4, "自由基电子数 [0-3]"),
    CatFeature("chirality", 4, 4, "手性标签 (None, R, S, Other)"),
    CatFeature("is_chiral_center", 2, 2, "是否为手性中心 [0, 1]"),
    CatFeature("distance_to_nearest_chiral", 11, 8, "距离最近手性中心的拓扑距离桶：0=分子内无手性中心，1=自身为手性中心，2-10 表示距离 1-9+]"),
    CatFeature("is_aromatic", 2, 2, "是否具有芳香性 [0, 1]"),
    CatFeature("is_in_ring", 2, 2, "是否在环上 [0, 1]"),
    CatFeature("smallest_ring_size", 11, 8, "所属最小环大小桶：0=非环原子，3-10 表示环大小 3-10+]"),
    CatFeature("is_in_murcko_scaffold", 2, 2, "是否属于 Murcko 骨架 [0, 1]"),
    CatFeature("num_single_bonds", 5, 4, "连接的单键数量 [0-4]"),
    CatFeature("num_double_bonds", 3, 4, "连接的双键数量 [0-2]"),
    CatFeature("num_triple_bonds", 2, 2, "连接的三键数量 [0-1]"),
    CatFeature("num_aromatic_bonds", 4, 4, "连接的芳香键数量 [0-3]"),
    CatFeature("is_acceptor", 2, 2, "是否为氢键受体 [0, 1]"),
    CatFeature("is_donor", 2, 2, "是否为氢键供体 [0, 1]"),
    CatFeature("is_hydrophobe", 2, 2, "是否具有疏水性 [0, 1]"),
    CatFeature("is_positive", 2, 2, "是否带正电 [0, 1]"),
    CatFeature("is_negative", 2, 2, "是否带负电 [0, 1]"),
]

LIGAND_ATOM_CONT_SCHEMA = [
    ContFeature("vdw_radius_mm3", "范德华半径 (MM3力场)"),
    ContFeature("atomic_weight", "相对原子质量"),
    ContFeature("en_pauling", "鲍林电负性"),
    ContFeature("electron_affinity", "电子亲和能"),
    ContFeature("first_ionization_energy", "第一电离能"),
    ContFeature("gasteiger_charge", "Gasteiger 偏电荷"),
    ContFeature("sasa_contribution", "对分子 SASA 的贡献值"),
    ContFeature("max_topological_distance", "到分子内最远原子的拓扑距离"),
    ContFeature("mean_topological_distance", "到分子内所有原子的平均拓扑距离"),
]

LIGAND_MOLECULE_CONT_SCHEMA = [
    ContFeature("wt", "分子量 (Molecular Weight)"),
    ContFeature("tpsa", "拓扑极性表面积 (TPSA)"),
    ContFeature("logp", "油水分配系数 (LogP)"),
    ContFeature("qed", "药物相似性定量评估 (QED)"),
    ContFeature("molar_refractivity", "摩尔折射率"),
    ContFeature("hallKier_alpha", "Hall-Kier Alpha 形状索引"),
    ContFeature("kappa2", "Kappa2 分子形状指数"),
    ContFeature("chi1v", "Chi1v 拓扑指数"),
    ContFeature("chi4v", "Chi4v 拓扑指数"),
]


def _build_protein_atom_name_vocab() -> tuple[str, ...]:
    names = {"UNK", "OXT"}

    for res_type in ResidueType:
        for atom_name in res_type.atom14:
            if atom_name:
                names.add(atom_name)

    backbone_first = [
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CG1", "CG2",
        "CD", "CD1", "CD2",
        "CE", "CE1", "CE2", "CE3",
        "CZ", "CZ2", "CZ3", "CH2",
        "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
        "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "SD", "SG",
    ]
    ordered = [name for name in backbone_first if name in names]
    ordered.extend(sorted(name for name in names if name not in ordered))
    return tuple(ordered)


PROTEIN_ATOM_NAME_VOCAB = _build_protein_atom_name_vocab()
PROTEIN_ATOM_NAME_TO_CLASS = {
    name: idx for idx, name in enumerate(PROTEIN_ATOM_NAME_VOCAB)
}

PROTEIN_RESIDUE_TORSION_NAMES = (
    "phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4",
)
PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

PROTEIN_ATOM_CAT_SCHEMA = [
    CatFeature("atomic_idx", 24, 16, "原子类型索引 (Element 枚举)"),
    CatFeature("atom_name_class", len(PROTEIN_ATOM_NAME_VOCAB), 12, "蛋白原子名类别"),
]

PROTEIN_ATOM_CONT_SCHEMA = [
    ContFeature("vdw_radius_mm3", "范德华半径"),
    ContFeature("atomic_weight", "相对原子质量"),
    ContFeature("en_pauling", "鲍林电负性"),
    ContFeature("electron_affinity", "电子亲和能"),
    ContFeature("first_ionization_energy", "第一电离能"),
    ContFeature("is_backbone", "是否为主链原子"),
    ContFeature("is_sidechain", "是否为侧链原子"),
    ContFeature("is_alpha_carbon", "是否为 CA 原子"),
    ContFeature("is_donor_like", "是否具有供体倾向"),
    ContFeature("is_acceptor_like", "是否具有受体倾向"),
    ContFeature("is_aromatic_like", "是否属于芳香体系"),
]

PROTEIN_ATOM_SCALAR_FIELDS = {
    "vdw_radius_mm3",
    "atomic_weight",
    "en_pauling",
    "electron_affinity",
    "first_ionization_energy",
}
PROTEIN_ATOM_SCALAR_DIM = sum(
    1 for f in PROTEIN_ATOM_CONT_SCHEMA if f.name in PROTEIN_ATOM_SCALAR_FIELDS
)

PROTEIN_RESIDUE_CAT_SCHEMA = [
    CatFeature("residue_type", 21, 64, "残基类型 (20种氨基酸 + UNK)"),
]

PROTEIN_RESIDUE_TORSION_CONT_SCHEMA = [
    item
    for name in PROTEIN_RESIDUE_TORSION_NAMES
    for item in (
        ContFeature(f"{name}_sin", f"{name.upper()} 角的正弦值"),
        ContFeature(f"{name}_cos", f"{name.upper()} 角的余弦值"),
    )
]

PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA = [
    *[
        ContFeature(f"{name}_valid", f"{name.upper()} 角是否可观测")
        for name in PROTEIN_RESIDUE_TORSION_NAMES
    ],
    *[
        ContFeature(
            f"backbone_{name.lower()}_observed",
            f"主链原子 {name} 是否被观测到",
        )
        for name in PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES
    ],
    ContFeature("has_prev_contiguous", "是否存在连续的前驱残基"),
    ContFeature("has_next_contiguous", "是否存在连续的后继残基"),
    ContFeature("segment_rel_pos", "在连续 segment 内的相对位置 [0, 1]"),
    ContFeature("segment_centrality", "在连续 segment 内的中心化位置 [-1, 1]"),
    ContFeature("segment_length_norm", "连续 segment 长度的归一化值"),
]

PROTEIN_RESIDUE_CONT_SCHEMA = [
    *PROTEIN_RESIDUE_TORSION_CONT_SCHEMA,
    *PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA,
]

PROTEIN_RESIDUE_TORSION_DIM = len(PROTEIN_RESIDUE_TORSION_CONT_SCHEMA)
PROTEIN_RESIDUE_CONTEXT_DIM = len(PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA)
PROTEIN_RESIDUE_TORSION_VALID_START = PROTEIN_RESIDUE_TORSION_DIM
PROTEIN_RESIDUE_TORSION_VALID_DIM = len(PROTEIN_RESIDUE_TORSION_NAMES)
PROTEIN_RESIDUE_BACKBONE_OBSERVED_START = (
    PROTEIN_RESIDUE_TORSION_VALID_START + PROTEIN_RESIDUE_TORSION_VALID_DIM
)
PROTEIN_RESIDUE_BACKBONE_OBSERVED_DIM = len(PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES)
