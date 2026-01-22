"""
分子编码器模块

提供蛋白质-配体复合物的特征编码功能
"""

from ehfnet.encoders.chemistry import Element, ResidueType
from ehfnet.encoders.feature_specs import (
    CatFeature,
    ContFeature,
    LIGAND_ATOM_CAT_SCHEMA,
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.encoders.esm_embedding import (
    compute_esm_embeddings,
    save_esm_embeddings,
    load_esm_embeddings,
    cache_esm_embeddings,
    load_or_compute_esm_embeddings,
)
from ehfnet.encoders.ligand_encoder import LigandEncoder
from ehfnet.encoders.protein_encoder import ProteinEncoder

__all__ = [
    # 化学基础
    "Element",
    "ResidueType",

    # 特征配置
    "CatFeature",
    "ContFeature",

    # 配体 Schema
    "LIGAND_ATOM_CAT_SCHEMA",
    "LIGAND_ATOM_CONT_SCHEMA",
    "LIGAND_MOLECULE_CONT_SCHEMA",

    # 蛋白质 Schema
    "PROTEIN_ATOM_CAT_SCHEMA",
    "PROTEIN_ATOM_CONT_SCHEMA",
    "PROTEIN_RESIDUE_CAT_SCHEMA",
    "PROTEIN_RESIDUE_CONT_SCHEMA",

    # ESM 嵌入
    "compute_esm_embeddings",
    "save_esm_embeddings",
    "load_esm_embeddings",
    "cache_esm_embeddings",
    "load_or_compute_esm_embeddings",

    # 编码器
    "LigandEncoder",
    "ProteinEncoder",
]
