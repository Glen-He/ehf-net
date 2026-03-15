"""
特征化模块入口。

导出特征规格、蛋白配体编码器和 ESM 工具，
统一特征侧组件的公开导入路径。
"""


from importlib import import_module

__all__ = [
    "Element",
    "ResidueType",
    "CatFeature",
    "ContFeature",
    "LIGAND_ATOM_CAT_SCHEMA",
    "LIGAND_ATOM_CONT_SCHEMA",
    "LIGAND_MOLECULE_CONT_SCHEMA",
    "PROTEIN_ATOM_NAME_TO_CLASS",
    "PROTEIN_ATOM_NAME_VOCAB",
    "PROTEIN_ATOM_CAT_SCHEMA",
    "PROTEIN_ATOM_CONT_SCHEMA",
    "PROTEIN_ATOM_SCALAR_DIM",
    "PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES",
    "PROTEIN_RESIDUE_CAT_SCHEMA",
    "PROTEIN_RESIDUE_CONT_SCHEMA",
    "PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA",
    "PROTEIN_RESIDUE_CONTEXT_DIM",
    "PROTEIN_RESIDUE_TORSION_CONT_SCHEMA",
    "PROTEIN_RESIDUE_TORSION_DIM",
    "PROTEIN_RESIDUE_TORSION_NAMES",
    "PROTEIN_RESIDUE_TORSION_VALID_DIM",
    "PROTEIN_RESIDUE_TORSION_VALID_START",
    "compute_esm_embeddings",
    "load_esm_embeddings",
    "load_or_compute_esm_embeddings",
    "save_esm_embeddings",
    "LigandEncoder",
    "LigandEncodingResult",
    "ProteinEncoder",
    "ProteinEncodingResult",
    "ProteinResidueSegment",
    "continuity_break_reason",
    "resolve_esm_residue_type",
    "segment_residues_by_continuity",
]

_EXPORT_MAP = {
    "Element": ("ehfnet.data.featurizers.chemistry", "Element"),
    "ResidueType": ("ehfnet.data.featurizers.chemistry", "ResidueType"),
    "CatFeature": ("ehfnet.data.featurizers.feature_specs", "CatFeature"),
    "ContFeature": ("ehfnet.data.featurizers.feature_specs", "ContFeature"),
    "LIGAND_ATOM_CAT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "LIGAND_ATOM_CAT_SCHEMA",
    ),
    "LIGAND_ATOM_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "LIGAND_ATOM_CONT_SCHEMA",
    ),
    "LIGAND_MOLECULE_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "LIGAND_MOLECULE_CONT_SCHEMA",
    ),
    "PROTEIN_ATOM_NAME_TO_CLASS": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_ATOM_NAME_TO_CLASS",
    ),
    "PROTEIN_ATOM_NAME_VOCAB": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_ATOM_NAME_VOCAB",
    ),
    "PROTEIN_ATOM_CAT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_ATOM_CAT_SCHEMA",
    ),
    "PROTEIN_ATOM_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_ATOM_CONT_SCHEMA",
    ),
    "PROTEIN_ATOM_SCALAR_DIM": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_ATOM_SCALAR_DIM",
    ),
    "PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES",
    ),
    "PROTEIN_RESIDUE_CAT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_CAT_SCHEMA",
    ),
    "PROTEIN_RESIDUE_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_CONT_SCHEMA",
    ),
    "PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_CONTEXT_CONT_SCHEMA",
    ),
    "PROTEIN_RESIDUE_CONTEXT_DIM": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_CONTEXT_DIM",
    ),
    "PROTEIN_RESIDUE_TORSION_CONT_SCHEMA": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_TORSION_CONT_SCHEMA",
    ),
    "PROTEIN_RESIDUE_TORSION_DIM": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_TORSION_DIM",
    ),
    "PROTEIN_RESIDUE_TORSION_NAMES": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_TORSION_NAMES",
    ),
    "PROTEIN_RESIDUE_TORSION_VALID_DIM": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_TORSION_VALID_DIM",
    ),
    "PROTEIN_RESIDUE_TORSION_VALID_START": (
        "ehfnet.data.featurizers.feature_specs",
        "PROTEIN_RESIDUE_TORSION_VALID_START",
    ),
    "compute_esm_embeddings": (
        "ehfnet.data.featurizers.esm_embedding",
        "compute_esm_embeddings",
    ),
    "load_esm_embeddings": ("ehfnet.data.featurizers.esm_embedding", "load_esm_embeddings"),
    "load_or_compute_esm_embeddings": (
        "ehfnet.data.featurizers.esm_embedding",
        "load_or_compute_esm_embeddings",
    ),
    "save_esm_embeddings": ("ehfnet.data.featurizers.esm_embedding", "save_esm_embeddings"),
    "LigandEncoder": ("ehfnet.data.featurizers.ligand_encoder", "LigandEncoder"),
    "LigandEncodingResult": (
        "ehfnet.data.featurizers.ligand_encoder",
        "LigandEncodingResult",
    ),
    "ProteinEncoder": ("ehfnet.data.featurizers.protein_encoder", "ProteinEncoder"),
    "ProteinEncodingResult": (
        "ehfnet.data.featurizers.protein_encoder",
        "ProteinEncodingResult",
    ),
    "ProteinResidueSegment": (
        "ehfnet.data.featurizers.protein_segments",
        "ProteinResidueSegment",
    ),
    "continuity_break_reason": (
        "ehfnet.data.featurizers.protein_segments",
        "continuity_break_reason",
    ),
    "resolve_esm_residue_type": (
        "ehfnet.data.featurizers.chemistry",
        "resolve_esm_residue_type",
    ),
    "segment_residues_by_continuity": (
        "ehfnet.data.featurizers.protein_segments",
        "segment_residues_by_continuity",
    ),
}


def __getattr__(name: str):
    """
    按名称返回公开对象。

    仅在首次访问时执行真实导入，
    用于避免包初始化阶段触发重模块加载或循环依赖。

    Args:
        name: 请求访问或解析的公开对象名称。

    Returns:
        object: 返回与名称对应的惰性导出对象。

    Raises:
        AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
    """

    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
