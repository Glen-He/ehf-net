"""
Checkpoint 契约定义。

负责构建、校验和比较模型配置签名，
保证 checkpoint 内容与当前模型结构和特征约定匹配。
"""


from collections.abc import Mapping
from typing import Any

from ehfnet.contracts.cache import GRAPH_CACHE_SCHEMA_TAG
from ehfnet.data.featurizers import (
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


CHECKPOINT_FEATURE_SCHEMA_TAG = "ehfnet_feature_signature_context_rigid"


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
    """
    构建特征签名。

    汇总当前特征 schema、上下文节点类型和 ESM 维度信息，
    用于将 checkpoint 与当前特征配置做一致性比对。

    Args:
        esm_dim: ESM 残基嵌入维度。

    Returns:
        dict[str, Any]: 当前特征 schema、context 节点约定与 ESM 维度组成的签名字典。

    Raises:
        ValueError: 当 `esm_dim` 不是正数时抛出。
    """
    esm_dim = int(esm_dim)
    if esm_dim <= 0:
        raise ValueError(f"esm_dim must be positive, got {esm_dim}.")

    return {
        "feature_schema": CHECKPOINT_FEATURE_SCHEMA_TAG,
        "graph_cache_schema": GRAPH_CACHE_SCHEMA_TAG,
        "ligand_atom_cat_schema": _serialize_cat_schema(LIGAND_ATOM_CAT_SCHEMA),
        "ligand_atom_cont_schema": _serialize_cont_schema(LIGAND_ATOM_CONT_SCHEMA),
        "ligand_molecule_cont_schema": _serialize_cont_schema(LIGAND_MOLECULE_CONT_SCHEMA),
        "protein_atom_cat_schema": _serialize_cat_schema(PROTEIN_ATOM_CAT_SCHEMA),
        "protein_atom_cont_schema": _serialize_cont_schema(PROTEIN_ATOM_CONT_SCHEMA),
        "protein_residue_cat_schema": _serialize_cat_schema(PROTEIN_RESIDUE_CAT_SCHEMA),
        "protein_residue_cont_schema": _serialize_cont_schema(PROTEIN_RESIDUE_CONT_SCHEMA),
        "context_node_type": "protein_context",
        "lig_atom_cont_count": len(LIGAND_ATOM_CONT_SCHEMA),
        "lig_mol_cont_count": len(LIGAND_MOLECULE_CONT_SCHEMA),
        "pro_atom_cont_count": len(PROTEIN_ATOM_CONT_SCHEMA),
        "pro_res_base_cont_count": len(PROTEIN_RESIDUE_CONT_SCHEMA),
        "pro_res_cont_count": len(PROTEIN_RESIDUE_CONT_SCHEMA) + esm_dim,
        "esm_dim": esm_dim,
    }


def _require_model_value(value: Any, *, name: str) -> Any:
    """
    校验 checkpoint 中的模型配置值已显式提供。

    Returns:
        Any: 返回已确认非空、可安全写入模型配置签名的值。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if value is None:
        raise ValueError(f"Missing required model configuration value: {name}.")
    return value


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
    m_dim_scalar: int,
    dropout_rate: float,
    num_rbf: int,
    r_cutoff: float,
    force_cutoff: float,
    frame_refine_threshold: float,
    frame_refine_temperature: float,
    energy_guide_threshold: float,
    energy_guide_temperature: float,
    clash_threshold: float,
    clash_push_threshold: float,
    clash_push_force: float,
    score_clamp_min: float,
    score_clamp_max: float,
    force_limit: float,
    max_neighbors: int,
    min_max_neighbors: int,
    knn_fallback_k: int,
    r_cutoff_intra: float,
    max_neighbors_intra: int,
    atom_neighbor_cap: int,
    residue_neighbor_cap: int,
    residue_radius_scale: float,
    residue_radius_bias: float,
    ligand_atom_fallback_k: int,
    protein_atom_fallback_k: int,
    protein_residue_fallback_k: int,
    dynamic_inter_cutoff: float,
    dynamic_inter_knn_k: int,
    dynamic_residue_cutoff: float,
    dynamic_residue_knn_k: int,
) -> dict[str, Any]:
    """
    构建模型配置签名。

    将模型结构、图构建参数和交互拓扑相关配置整理为标准字典，
    作为 checkpoint 中描述模型结构的核心配置片段。

    Args:
        hidden_dim: 隐藏层维度。
        time_dim: 时间嵌入维度。
        num_gnn_blocks: 主干 GNN 块数量。
        lig_atom_cont_count: 配体原子连续特征维度。
        lig_mol_cont_count: 配体分子连续特征维度。
        pro_atom_cont_count: 蛋白原子连续特征维度。
        pro_res_cont_count: 蛋白残基连续特征维度。
        esm_dim: ESM 残基嵌入维度。
        interaction_profile: 跨图交互拓扑配置。
        m_dim_scalar: 消息传递分支的标量维度。
        dropout_rate: Dropout 比例。
        num_rbf: RBF 基函数数量。
        r_cutoff: 几何邻域构建的距离截断半径。
        force_cutoff: 力相关分支使用的局部截断半径。
        frame_refine_threshold: 主惯量帧细化门控阈值。
        frame_refine_temperature: 主惯量帧细化门控温度。
        energy_guide_threshold: 能量引导门控阈值。
        energy_guide_temperature: 能量引导门控温度。
        clash_threshold: 位阻判定阈值。
        clash_push_threshold: 位阻推开分支使用的距离阈值。
        clash_push_force: 位阻推开分支的力缩放系数。
        score_clamp_min: 分数裁剪下界。
        score_clamp_max: 分数裁剪上界。
        force_limit: 力大小的软限制。
        max_neighbors: 预测头阶段每个节点保留的最大邻居数。
        min_max_neighbors: 预测头动态邻居数的下限。
        knn_fallback_k: 回退到 kNN 时使用的邻居数。
        r_cutoff_intra: 图内边构建的距离截断半径。
        max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
        atom_neighbor_cap: 原子层图内边的邻居上限。
        residue_neighbor_cap: 残基层图内边的邻居上限。
        residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
        residue_radius_bias: 残基层邻域半径的额外偏置。
        ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
        protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
        protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
        dynamic_inter_cutoff: 动态跨图原子边的半径阈值。
        dynamic_inter_knn_k: 动态跨图原子边回退到 kNN 时的邻居数。
        dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
        dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。

    Returns:
        dict[str, Any]: 规范化后的模型配置签名字典，可直接写入 checkpoint。
    """
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
        "m_dim_scalar": int(_require_model_value(m_dim_scalar, name="m_dim_scalar")),
        "dropout_rate": float(
            _require_model_value(dropout_rate, name="dropout_rate")
        ),
        "num_rbf": int(_require_model_value(num_rbf, name="num_rbf")),
        "r_cutoff": float(_require_model_value(r_cutoff, name="r_cutoff")),
        "force_cutoff": float(
            _require_model_value(force_cutoff, name="force_cutoff")
        ),
        "frame_refine_threshold": float(
            _require_model_value(
                frame_refine_threshold,
                name="frame_refine_threshold",
            )
        ),
        "frame_refine_temperature": float(
            _require_model_value(
                frame_refine_temperature,
                name="frame_refine_temperature",
            )
        ),
        "energy_guide_threshold": float(
            _require_model_value(
                energy_guide_threshold,
                name="energy_guide_threshold",
            )
        ),
        "energy_guide_temperature": float(
            _require_model_value(
                energy_guide_temperature,
                name="energy_guide_temperature",
            )
        ),
        "clash_threshold": float(
            _require_model_value(clash_threshold, name="clash_threshold")
        ),
        "clash_push_threshold": float(
            _require_model_value(
                clash_push_threshold,
                name="clash_push_threshold",
            )
        ),
        "clash_push_force": float(
            _require_model_value(clash_push_force, name="clash_push_force")
        ),
        "score_clamp_min": float(
            _require_model_value(score_clamp_min, name="score_clamp_min")
        ),
        "score_clamp_max": float(
            _require_model_value(score_clamp_max, name="score_clamp_max")
        ),
        "force_limit": float(_require_model_value(force_limit, name="force_limit")),
        "max_neighbors": int(
            _require_model_value(max_neighbors, name="max_neighbors")
        ),
        "min_max_neighbors": int(
            _require_model_value(min_max_neighbors, name="min_max_neighbors")
        ),
        "knn_fallback_k": int(
            _require_model_value(knn_fallback_k, name="knn_fallback_k")
        ),
        "r_cutoff_intra": float(
            _require_model_value(r_cutoff_intra, name="r_cutoff_intra")
        ),
        "max_neighbors_intra": int(
            _require_model_value(max_neighbors_intra, name="max_neighbors_intra")
        ),
        "atom_neighbor_cap": int(
            _require_model_value(atom_neighbor_cap, name="atom_neighbor_cap")
        ),
        "residue_neighbor_cap": int(
            _require_model_value(residue_neighbor_cap, name="residue_neighbor_cap")
        ),
        "residue_radius_scale": float(
            _require_model_value(
                residue_radius_scale,
                name="residue_radius_scale",
            )
        ),
        "residue_radius_bias": float(
            _require_model_value(residue_radius_bias, name="residue_radius_bias")
        ),
        "ligand_atom_fallback_k": int(
            _require_model_value(
                ligand_atom_fallback_k,
                name="ligand_atom_fallback_k",
            )
        ),
        "protein_atom_fallback_k": int(
            _require_model_value(
                protein_atom_fallback_k,
                name="protein_atom_fallback_k",
            )
        ),
        "protein_residue_fallback_k": int(
            _require_model_value(
                protein_residue_fallback_k,
                name="protein_residue_fallback_k",
            )
        ),
        "dynamic_inter_cutoff": float(
            _require_model_value(
                dynamic_inter_cutoff,
                name="dynamic_inter_cutoff",
            )
        ),
        "dynamic_inter_knn_k": int(
            _require_model_value(dynamic_inter_knn_k, name="dynamic_inter_knn_k")
        ),
        "dynamic_residue_cutoff": float(
            _require_model_value(
                dynamic_residue_cutoff,
                name="dynamic_residue_cutoff",
            )
        ),
        "dynamic_residue_knn_k": int(
            _require_model_value(
                dynamic_residue_knn_k,
                name="dynamic_residue_knn_k",
            )
        ),
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


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """
    校验 checkpoint 契约。

    对比 checkpoint 中保存的特征签名和模型配置签名，
    在加载前尽早发现结构不一致或配置缺失的问题。

    Args:
        checkpoint: 待校验的 checkpoint 内容字典。

    Returns:
        dict[str, Any]: 返回通过契约校验后的模型配置字典。

    Raises:
        ValueError: 当 checkpoint 缺少关键字段或与当前配置不兼容时抛出。
    """
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError(
            "Checkpoint is missing model_config and does not satisfy the current checkpoint contract. "
            "Please regenerate a checkpoint with the current feature pipeline."
        )

    feature_signature = checkpoint.get("feature_signature")
    if not isinstance(feature_signature, Mapping):
        raise ValueError(
            "Checkpoint is missing feature_signature and does not satisfy the current checkpoint contract. "
            "Please regenerate a checkpoint with the current feature pipeline."
        )

    esm_dim = model_config.get("esm_dim")
    if esm_dim is None:
        raise ValueError("Checkpoint model_config is missing esm_dim.")

    expected_signature = build_feature_signature(esm_dim=int(esm_dim))
    mismatches = _format_mismatch_lines(feature_signature, expected_signature)
    if mismatches:
        mismatch_text = "\n".join(f"  - {line}" for line in mismatches)
        raise ValueError(
            "Checkpoint feature signature does not match the current codebase.\n"
            f"{mismatch_text}\n"
            "Do not reuse this checkpoint with the current feature pipeline; retrain or export a new checkpoint that matches the current contract."
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
        "force_cutoff",
        "frame_refine_threshold",
        "frame_refine_temperature",
        "energy_guide_threshold",
        "energy_guide_temperature",
        "clash_threshold",
        "clash_push_threshold",
        "clash_push_force",
        "score_clamp_min",
        "score_clamp_max",
        "force_limit",
        "max_neighbors",
        "min_max_neighbors",
        "knn_fallback_k",
        "r_cutoff_intra",
        "max_neighbors_intra",
        "atom_neighbor_cap",
        "residue_neighbor_cap",
        "residue_radius_scale",
        "residue_radius_bias",
        "ligand_atom_fallback_k",
        "protein_atom_fallback_k",
        "protein_residue_fallback_k",
        "dynamic_inter_cutoff",
        "dynamic_inter_knn_k",
        "dynamic_residue_cutoff",
        "dynamic_residue_knn_k",
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
