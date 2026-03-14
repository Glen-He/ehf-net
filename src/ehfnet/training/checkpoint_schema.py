"""
Checkpoint schema helpers.

集中定义模型输入签名与 checkpoint 兼容性校验，避免旧特征链被静默复用。
"""

from __future__ import annotations

from typing import Any, Mapping

from ehfnet.datasets.pdbbind import GRAPH_CACHE_VERSION_TAG
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


CHECKPOINT_FEATURE_SCHEMA_VERSION = "ehfnet_feature_signature_v2"


def _serialize_cat_schema(schema: list[CatFeature]) -> list[dict[str, int | str]]:
    return [
        {
            "name": feat.name,
            "num_embeddings": int(feat.num_embeddings),
            "embed_dim": int(feat.embed_dim),
        }
        for feat in schema
    ]


def _serialize_cont_schema(schema: list[ContFeature]) -> list[str]:
    return [feat.name for feat in schema]


def build_feature_signature(*, esm_dim: int) -> dict[str, Any]:
    esm_dim = int(esm_dim)
    if esm_dim <= 0:
        raise ValueError(f"esm_dim must be positive, got {esm_dim}.")

    return {
        "schema_version": CHECKPOINT_FEATURE_SCHEMA_VERSION,
        "graph_cache_version": GRAPH_CACHE_VERSION_TAG,
        "ligand_atom_cat_schema": _serialize_cat_schema(LIGAND_ATOM_CAT_SCHEMA),
        "ligand_atom_cont_schema": _serialize_cont_schema(LIGAND_ATOM_CONT_SCHEMA),
        "ligand_molecule_cont_schema": _serialize_cont_schema(LIGAND_MOLECULE_CONT_SCHEMA),
        "protein_atom_cat_schema": _serialize_cat_schema(PROTEIN_ATOM_CAT_SCHEMA),
        "protein_atom_cont_schema": _serialize_cont_schema(PROTEIN_ATOM_CONT_SCHEMA),
        "protein_residue_cat_schema": _serialize_cat_schema(PROTEIN_RESIDUE_CAT_SCHEMA),
        "protein_residue_cont_schema": _serialize_cont_schema(PROTEIN_RESIDUE_CONT_SCHEMA),
        "lig_atom_cont_count": len(LIGAND_ATOM_CONT_SCHEMA),
        "lig_mol_cont_count": len(LIGAND_MOLECULE_CONT_SCHEMA),
        "pro_atom_cont_count": len(PROTEIN_ATOM_CONT_SCHEMA),
        "pro_res_base_cont_count": len(PROTEIN_RESIDUE_CONT_SCHEMA),
        "pro_res_cont_count": len(PROTEIN_RESIDUE_CONT_SCHEMA) + esm_dim,
        "esm_dim": esm_dim,
    }


def build_model_config(
    *,
    hidden_dim: int,
    time_dim: int,
    num_gnn_blocks: int,
    lig_atom_cont_count: int,
    lig_mol_cont_count: int,
    pro_atom_cont_count: int,
    pro_res_cont_count: int,
    esm_dim: int,
    interaction_profile: str,
    m_dim_scalar: int = 16,
    dropout_rate: float = 0.0,
    num_rbf: int = 50,
    r_cutoff: float = 10.0,
    fix_protein: bool = True,
) -> dict[str, Any]:
    return {
        "hidden_dim": int(hidden_dim),
        "time_dim": int(time_dim),
        "num_gnn_blocks": int(num_gnn_blocks),
        "lig_atom_cont_count": int(lig_atom_cont_count),
        "lig_mol_cont_count": int(lig_mol_cont_count),
        "pro_atom_cont_count": int(pro_atom_cont_count),
        "pro_res_cont_count": int(pro_res_cont_count),
        "esm_dim": int(esm_dim),
        "interaction_profile": str(interaction_profile),
        "m_dim_scalar": int(m_dim_scalar),
        "dropout_rate": float(dropout_rate),
        "num_rbf": int(num_rbf),
        "r_cutoff": float(r_cutoff),
        "fix_protein": bool(fix_protein),
    }


def _format_mismatch_lines(
    actual_signature: Mapping[str, Any],
    expected_signature: Mapping[str, Any],
) -> list[str]:
    lines: list[str] = []
    for key, expected_value in expected_signature.items():
        actual_value = actual_signature.get(key)
        if actual_value != expected_value:
            lines.append(
                f"{key}: checkpoint={actual_value!r}, current={expected_value!r}"
            )
    return lines


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError(
            "Checkpoint is missing model_config and is treated as legacy/incompatible. "
            "Please regenerate a checkpoint with the current feature pipeline."
        )

    feature_signature = checkpoint.get("feature_signature")
    if not isinstance(feature_signature, Mapping):
        raise ValueError(
            "Checkpoint is missing feature_signature and is treated as legacy/incompatible. "
            "Please regenerate a checkpoint with the current feature pipeline."
        )

    esm_dim = model_config.get("esm_dim", feature_signature.get("esm_dim"))
    if esm_dim is None:
        raise ValueError("Checkpoint model_config is missing esm_dim.")

    expected_signature = build_feature_signature(esm_dim=int(esm_dim))
    mismatches = _format_mismatch_lines(feature_signature, expected_signature)
    if mismatches:
        mismatch_text = "\n".join(f"  - {line}" for line in mismatches)
        raise ValueError(
            "Checkpoint feature signature does not match the current codebase.\n"
            f"{mismatch_text}\n"
            "Do not reuse this checkpoint with the current feature pipeline; retrain or export a new compatible checkpoint."
        )

    required_keys = [
        "hidden_dim",
        "time_dim",
        "num_gnn_blocks",
        "lig_atom_cont_count",
        "lig_mol_cont_count",
        "pro_atom_cont_count",
        "pro_res_cont_count",
        "esm_dim",
        "interaction_profile",
        "m_dim_scalar",
        "dropout_rate",
        "num_rbf",
        "r_cutoff",
        "fix_protein",
    ]
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise ValueError(
            f"Checkpoint model_config is incomplete; missing keys: {missing_keys}."
        )

    current_counts = {
        "lig_atom_cont_count": expected_signature["lig_atom_cont_count"],
        "lig_mol_cont_count": expected_signature["lig_mol_cont_count"],
        "pro_atom_cont_count": expected_signature["pro_atom_cont_count"],
        "pro_res_cont_count": expected_signature["pro_res_cont_count"],
        "esm_dim": expected_signature["esm_dim"],
    }
    count_mismatches = [
        f"{key}: checkpoint={model_config.get(key)!r}, current={expected_value!r}"
        for key, expected_value in current_counts.items()
        if model_config.get(key) != expected_value
    ]
    if count_mismatches:
        mismatch_text = "\n".join(f"  - {line}" for line in count_mismatches)
        raise ValueError(
            "Checkpoint model_config is inconsistent with the current feature counts.\n"
            f"{mismatch_text}"
        )

    return dict(model_config)
