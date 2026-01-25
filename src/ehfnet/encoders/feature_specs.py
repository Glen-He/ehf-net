"""
特征规格定义

包含分类特征和连续特征的配置，以及配体和蛋白质的特征 Schema
"""

from typing import NamedTuple


class CatFeature(NamedTuple):
    """
    分类特征配置
    """

    name: str
    num_embeddings: int
    embed_dim: int
    description: str


class ContFeature(NamedTuple):
    """
    连续特征配置
    """

    name: str
    description: str


# --- 配体特征 Schema ---
LIGAND_ATOM_CAT_SCHEMA = [

    # 基础属性
    CatFeature("num_hs",                    6,  4,  "氢原子数量 [0-5]"),
    CatFeature("degree",                    8,  8,  "原子度 (连接数) [0-7]"),
    CatFeature("implicit_valence",          7,  4,  "隐式化合价 [0-6] (硫/磷可达6价)"),
    CatFeature("atomic_idx",                13, 16, "原子类型索引 (基于 Element 枚举)"),
    CatFeature("formal_charge",             9,  8,  "形式电荷 [-4, +4] 映射到 [0, 8]"),
    CatFeature("hybridization",             9,  8,  "杂化类型 (SP, SP2, SP3等)"),
    CatFeature("num_radical_electrons",     4,  4,  "自由基电子数 [0-3]"),

    # 手性与立体化学
    CatFeature("chirality",                 4,  4,  "手性标签 (None, R, S, Other)"),
    CatFeature("is_chiral_center",          2,  2,  "是否为手性中心 [0, 1]"),
    CatFeature("distance_to_nearest_chiral",10, 8,  "距离最近手性中心的拓扑距离 [0-9]"),

    # 环与拓扑结构
    CatFeature("is_aromatic",               2,  2,  "是否具有芳香性 [0, 1]"),
    CatFeature("is_in_ring",                2,  2,  "是否在环上 [0, 1]"),
    CatFeature("smallest_ring_size",        10, 8,  "所属最小环的大小 [0-9]"),
    CatFeature("is_in_murcko_scaffold",     2,  2,  "是否属于 Murcko 骨架 [0, 1]"),

    # 键连接性统计
    CatFeature("num_single_bonds",          5,  4,  "连接的单键数量 [0-4]"),
    CatFeature("num_double_bonds",          3,  4,  "连接的双键数量 [0-2]"),
    CatFeature("num_triple_bonds",          2,  2,  "连接的三键数量 [0-1]"),
    CatFeature("num_aromatic_bonds",        4,  4,  "连接的芳香键数量 [0-3]"),

    # 药效团特征
    CatFeature("is_acceptor",               2,  2,  "是否为氢键受体 [0, 1]"),
    CatFeature("is_donor",                  2,  2,  "是否为氢键供体 [0, 1]"),
    CatFeature("is_hydrophobe",             2,  2,  "是否具有疏水性 [0, 1]"),
    CatFeature("is_positive",               2,  2,  "是否带正电 [0, 1]"),
    CatFeature("is_negative",               2,  2,  "是否带负电 [0, 1]"),
]


LIGAND_ATOM_CONT_SCHEMA = [

    # 物理化学性质
    ContFeature("vdw_radius_mm3",           "范德华半径 (MM3力场)"),
    ContFeature("atomic_weight",            "相对原子质量"),
    ContFeature("en_pauling",               "鲍林电负性"),
    ContFeature("electron_affinity",        "电子亲和能"),
    ContFeature("first_ionization_energy",  "第一电离能"),
    
    # 电荷与溶剂化
    ContFeature("gasteiger_charge",         "Gasteiger 偏电荷"),
    ContFeature("sasa_contribution",        "对分子 SASA 的贡献值"),
    
    # 拓扑距离
    ContFeature("max_topological_distance", "到分子内最远原子的拓扑距离"),
    ContFeature("mean_topological_distance","到分子内所有原子的平均拓扑距离"),
]


LIGAND_MOLECULE_CONT_SCHEMA = [

    # 全局理化性质
    ContFeature("wt",                       "分子量 (Molecular Weight)"),
    ContFeature("tpsa",                     "拓扑极性表面积 (TPSA)"),
    ContFeature("logp",                     "油水分配系数 (LogP)"),
    ContFeature("qed",                      "药物相似性定量评估 (QED)"),
    ContFeature("molar_refractivity",       "摩尔折射率"),
    
    # 结构描述符
    ContFeature("hallKier_alpha",           "Hall-Kier Alpha 形状索引"),
    ContFeature("kappa2",                   "Kappa2 分子形状指数"),
    ContFeature("chi1v",                    "Chi1v 拓扑指数"),
    ContFeature("chi4v",                    "Chi4v 拓扑指数"),
]


# --- 蛋白质特征 Schema ---
PROTEIN_ATOM_CAT_SCHEMA = [
    CatFeature("atomic_idx",                13, 16, "原子类型索引 (Element 枚举)"),
]


PROTEIN_ATOM_CONT_SCHEMA = [
    ContFeature("vdw_radius_mm3",           "范德华半径"),
    ContFeature("atomic_weight",            "相对原子质量"),
    ContFeature("en_pauling",               "鲍林电负性"),
    ContFeature("electron_affinity",        "电子亲和能"),
    ContFeature("first_ionization_energy",  "第一电离能"),
]


PROTEIN_RESIDUE_CAT_SCHEMA = [
    CatFeature("residue_type",              21,   64, "残基类型 (20种氨基酸 + UNK)"),
    CatFeature("residue_id",                4096, 32, "残基序列号 (截断到 4096 或相对位置)"),
]


PROTEIN_RESIDUE_CONT_SCHEMA = [

    # 骨架扭转角 (正弦/余弦编码)
    ContFeature("phi_sin",                  "Phi 角的正弦值"),
    ContFeature("phi_cos",                  "Phi 角的余弦值"),
    ContFeature("psi_sin",                  "Psi 角的正弦值"),
    ContFeature("psi_cos",                  "Psi 角的余弦值"),
    ContFeature("omega_sin",                "Omega 角的正弦值"),
    ContFeature("omega_cos",                "Omega 角的余弦值"),
    
    # 侧链扭转角
    ContFeature("chi1_sin",                 "Chi1 角的正弦值"),
    ContFeature("chi1_cos",                 "Chi1 角的余弦值"),
    ContFeature("chi2_sin",                 "Chi2 角的正弦值"),
    ContFeature("chi2_cos",                 "Chi2 角的余弦值"),
    ContFeature("chi3_sin",                 "Chi3 角的正弦值"),
    ContFeature("chi3_cos",                 "Chi3 角的余弦值"),
    ContFeature("chi4_sin",                 "Chi4 角的正弦值"),
    ContFeature("chi4_cos",                 "Chi4 角的余弦值"),
]
