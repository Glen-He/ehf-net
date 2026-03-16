"""
运行时工厂。

负责按统一配置构造数据集、模型及其依赖对象，
减少入口脚本中的装配逻辑和重复代码。
"""


from collections.abc import Mapping
from typing import Any, TypeVar

import torch

from ehfnet.data.datasets import ProteinLigandDataset
from ehfnet.models import EHFNet


T = TypeVar("T")


def _require_value(value: T | None, *, name: str) -> T:
    """
    校验运行时配置值已显式提供。

    Returns:
        T: 返回已确认非空、可继续用于运行时装配的配置值。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if value is None:
        raise ValueError(f"Missing required configuration value: {name}.")
    return value


def build_dataset(
    *,
    root: str,
    index_file: str,
    esm_root: str | None = None,
    esm: str,
    esm_model_name: str,
    esm_device: str | None = None,
    esm_dim: int,
    r_cutoff_intra: float,
    max_neighbors_intra: int,
    atom_neighbor_cap: int,
    residue_neighbor_cap: int,
    residue_radius_scale: float,
    residue_radius_bias: float,
    ligand_atom_fallback_k: int,
    protein_atom_fallback_k: int,
    protein_residue_fallback_k: int,
    interaction_profile: str,
    force_reprocess: bool = False,
    pre_filter: Any = None,
) -> ProteinLigandDataset:
    """
    构造数据集对象。

    按统一参数约定实例化 `ProteinLigandDataset`，
    让训练和推理侧共享同一套数据集装配逻辑。

    Args:
        root: 数据集根目录。
        index_file: 数据索引文件路径。
        esm_root: ESM 缓存根目录。
        esm: ESM 处理模式或缓存策略。
        esm_model_name: ESM 主干模型名称。
        esm_device: 执行 ESM 推理时使用的设备。
        esm_dim: ESM 残基嵌入维度。
        r_cutoff_intra: 图内边构建的距离截断半径。
        max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
        atom_neighbor_cap: 原子层图内边的邻居上限。
        residue_neighbor_cap: 残基层图内边的邻居上限。
        residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
        residue_radius_bias: 残基层邻域半径的额外偏置。
        ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
        protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
        protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
        interaction_profile: 跨图交互拓扑配置。
        force_reprocess: 是否忽略已有缓存并强制重新预处理。
        pre_filter: 样本写入缓存前执行的过滤回调。

    Returns:
        ProteinLigandDataset: 按当前运行参数构造好的数据集对象。
    """

    return ProteinLigandDataset(
        root=root,
        index_file=index_file,
        esm_root=esm_root,
        esm=esm,
        esm_model_name=str(_require_value(esm_model_name, name="esm_model_name")),
        esm_device=esm_device,
        esm_dim=int(_require_value(esm_dim, name="esm_dim")),
        r_cutoff_intra=float(_require_value(r_cutoff_intra, name="r_cutoff_intra")),
        max_neighbors_intra=int(
            _require_value(max_neighbors_intra, name="max_neighbors_intra")
        ),
        atom_neighbor_cap=int(
            _require_value(atom_neighbor_cap, name="atom_neighbor_cap")
        ),
        residue_neighbor_cap=int(
            _require_value(residue_neighbor_cap, name="residue_neighbor_cap")
        ),
        residue_radius_scale=float(
            _require_value(residue_radius_scale, name="residue_radius_scale")
        ),
        residue_radius_bias=float(
            _require_value(residue_radius_bias, name="residue_radius_bias")
        ),
        ligand_atom_fallback_k=int(
            _require_value(ligand_atom_fallback_k, name="ligand_atom_fallback_k")
        ),
        protein_atom_fallback_k=int(
            _require_value(protein_atom_fallback_k, name="protein_atom_fallback_k")
        ),
        protein_residue_fallback_k=int(
            _require_value(
                protein_residue_fallback_k,
                name="protein_residue_fallback_k",
            )
        ),
        interaction_profile=str(
            _require_value(interaction_profile, name="interaction_profile")
        ),
        force_reprocess=force_reprocess,
        pre_filter=pre_filter,
    )


def build_dataset_from_model_config(
    *,
    root: str,
    index_file: str,
    model_config: Mapping[str, Any],
    esm_root: str | None = None,
    esm: str,
    esm_model_name: str,
    esm_device: str | None = None,
    force_reprocess: bool = False,
    pre_filter: Any = None,
) -> ProteinLigandDataset:
    """
    按模型配置构造数据集。

    从 checkpoint 或模型配置中提取必要字段创建数据集，
    确保缓存读取与模型特征约定保持一致。

    Args:
        root: 数据集根目录。
        index_file: 数据索引文件路径。
        model_config: 模型配置字典。
        esm_root: ESM 缓存根目录。
        esm: ESM 处理模式或缓存策略。
        esm_model_name: ESM 主干模型名称。
        esm_device: 执行 ESM 推理时使用的设备。
        force_reprocess: 是否忽略已有缓存并强制重新预处理。
        pre_filter: 样本写入缓存前执行的过滤回调。

    Returns:
        ProteinLigandDataset: 与给定模型配置保持一致的数据集对象。
    """

    return build_dataset(
        root=root,
        index_file=index_file,
        esm_root=esm_root,
        esm=esm,
        esm_model_name=esm_model_name,
        esm_device=esm_device,
        esm_dim=int(model_config["esm_dim"]),
        r_cutoff_intra=float(model_config["r_cutoff_intra"]),
        max_neighbors_intra=int(model_config["max_neighbors_intra"]),
        atom_neighbor_cap=int(model_config["atom_neighbor_cap"]),
        residue_neighbor_cap=int(model_config["residue_neighbor_cap"]),
        residue_radius_scale=float(model_config["residue_radius_scale"]),
        residue_radius_bias=float(model_config["residue_radius_bias"]),
        ligand_atom_fallback_k=int(model_config["ligand_atom_fallback_k"]),
        protein_atom_fallback_k=int(model_config["protein_atom_fallback_k"]),
        protein_residue_fallback_k=int(model_config["protein_residue_fallback_k"]),
        interaction_profile=str(model_config["interaction_profile"]),
        force_reprocess=force_reprocess,
        pre_filter=pre_filter,
    )


def build_model(
    *,
    hidden_dim: int,
    time_dim: int,
    num_gnn_blocks: int,
    lig_atom_cont_count: int,
    lig_mol_cont_count: int,
    pro_atom_cont_count: int,
    pro_res_cont_count: int,
    interaction_profile: str,
    normalization_stats: dict[str, Any] | None = None,
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
    dynamic_inter_cutoff: float,
    dynamic_inter_knn_k: int,
    dynamic_inter_max_neighbors: int,
    dynamic_residue_cutoff: float,
    dynamic_residue_knn_k: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
) -> EHFNet:
    """
    构造 EHFNet 模型。

    按统一参数约定实例化主模型及其依赖配置，
    避免入口脚本中重复堆叠模型装配细节。

    Args:
        hidden_dim: 隐藏层维度。
        time_dim: 时间嵌入维度。
        num_gnn_blocks: 主干 GNN 块数量。
        lig_atom_cont_count: 配体原子连续特征维度。
        lig_mol_cont_count: 配体分子连续特征维度。
        pro_atom_cont_count: 蛋白原子连续特征维度。
        pro_res_cont_count: 蛋白残基连续特征维度。
        interaction_profile: 跨图交互拓扑配置。
        normalization_stats: 输入特征归一化统计量。
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
        dynamic_inter_cutoff: 动态跨图原子边的半径阈值。
        dynamic_inter_knn_k: 动态跨图原子边回退到 kNN 时的邻居数。
        dynamic_inter_max_neighbors: 动态跨图原子边的单源邻居上限。
        dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
        dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。
        dynamic_residue_max_neighbors: 动态配体-残基边的单源邻居上限。
        dynamic_residue_candidate_topk: 动态配体-残基边每个复合物保留的候选残基数。

    Returns:
        EHFNet: 按给定参数实例化完成的主模型。
    """

    return EHFNet(
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        interaction_profile=interaction_profile,
        normalization_stats=normalization_stats,
        m_dim_scalar=int(_require_value(m_dim_scalar, name="m_dim_scalar")),
        dropout_rate=float(_require_value(dropout_rate, name="dropout_rate")),
        num_rbf=int(_require_value(num_rbf, name="num_rbf")),
        r_cutoff=float(_require_value(r_cutoff, name="r_cutoff")),
        force_cutoff=float(_require_value(force_cutoff, name="force_cutoff")),
        frame_refine_threshold=float(
            _require_value(frame_refine_threshold, name="frame_refine_threshold")
        ),
        frame_refine_temperature=float(
            _require_value(
                frame_refine_temperature,
                name="frame_refine_temperature",
            )
        ),
        energy_guide_threshold=float(
            _require_value(energy_guide_threshold, name="energy_guide_threshold")
        ),
        energy_guide_temperature=float(
            _require_value(
                energy_guide_temperature,
                name="energy_guide_temperature",
            )
        ),
        clash_threshold=float(_require_value(clash_threshold, name="clash_threshold")),
        clash_push_threshold=float(
            _require_value(clash_push_threshold, name="clash_push_threshold")
        ),
        clash_push_force=float(
            _require_value(clash_push_force, name="clash_push_force")
        ),
        score_clamp_min=float(
            _require_value(score_clamp_min, name="score_clamp_min")
        ),
        score_clamp_max=float(
            _require_value(score_clamp_max, name="score_clamp_max")
        ),
        force_limit=float(_require_value(force_limit, name="force_limit")),
        max_neighbors=int(_require_value(max_neighbors, name="max_neighbors")),
        min_max_neighbors=int(
            _require_value(min_max_neighbors, name="min_max_neighbors")
        ),
        knn_fallback_k=int(_require_value(knn_fallback_k, name="knn_fallback_k")),
        dynamic_inter_cutoff=float(
            _require_value(dynamic_inter_cutoff, name="dynamic_inter_cutoff")
        ),
        dynamic_inter_knn_k=int(
            _require_value(dynamic_inter_knn_k, name="dynamic_inter_knn_k")
        ),
        dynamic_inter_max_neighbors=int(
            _require_value(dynamic_inter_max_neighbors, name="dynamic_inter_max_neighbors")
        ),
        dynamic_residue_cutoff=float(
            _require_value(dynamic_residue_cutoff, name="dynamic_residue_cutoff")
        ),
        dynamic_residue_knn_k=int(
            _require_value(dynamic_residue_knn_k, name="dynamic_residue_knn_k")
        ),
        dynamic_residue_max_neighbors=int(
            _require_value(dynamic_residue_max_neighbors, name="dynamic_residue_max_neighbors")
        ),
        dynamic_residue_candidate_topk=int(
            _require_value(
                dynamic_residue_candidate_topk,
                name="dynamic_residue_candidate_topk",
            )
        ),
    )


def build_model_from_config(
    model_config: Mapping[str, Any],
    *,
    normalization_stats: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> EHFNet:
    """
    按配置恢复模型。

    从 checkpoint 或配置字典中提取模型参数并创建实例，
    用于评估、继续训练和推理时的统一模型恢复流程。

    Args:
        model_config: 模型配置字典。
        normalization_stats: 输入特征归一化统计量。
        device: 运行所用设备，如 CPU 或 CUDA 设备。

    Returns:
        EHFNet: 根据配置字典恢复出的模型实例。
    """

    model = build_model(
        hidden_dim=int(model_config["hidden_dim"]),
        time_dim=int(model_config["time_dim"]),
        num_gnn_blocks=int(model_config["num_gnn_blocks"]),
        lig_atom_cont_count=int(model_config["lig_atom_cont_count"]),
        lig_mol_cont_count=int(model_config["lig_mol_cont_count"]),
        pro_atom_cont_count=int(model_config["pro_atom_cont_count"]),
        pro_res_cont_count=int(model_config["pro_res_cont_count"]),
        interaction_profile=str(model_config["interaction_profile"]),
        normalization_stats=normalization_stats,
        m_dim_scalar=int(model_config["m_dim_scalar"]),
        dropout_rate=float(model_config["dropout_rate"]),
        num_rbf=int(model_config["num_rbf"]),
        r_cutoff=float(model_config["r_cutoff"]),
        force_cutoff=float(model_config["force_cutoff"]),
        frame_refine_threshold=float(model_config["frame_refine_threshold"]),
        frame_refine_temperature=float(model_config["frame_refine_temperature"]),
        energy_guide_threshold=float(model_config["energy_guide_threshold"]),
        energy_guide_temperature=float(model_config["energy_guide_temperature"]),
        clash_threshold=float(model_config["clash_threshold"]),
        clash_push_threshold=float(model_config["clash_push_threshold"]),
        clash_push_force=float(model_config["clash_push_force"]),
        score_clamp_min=float(model_config["score_clamp_min"]),
        score_clamp_max=float(model_config["score_clamp_max"]),
        force_limit=float(model_config["force_limit"]),
        max_neighbors=int(model_config["max_neighbors"]),
        min_max_neighbors=int(model_config["min_max_neighbors"]),
        knn_fallback_k=int(model_config["knn_fallback_k"]),
        dynamic_inter_cutoff=float(model_config["dynamic_inter_cutoff"]),
        dynamic_inter_knn_k=int(model_config["dynamic_inter_knn_k"]),
        dynamic_inter_max_neighbors=int(model_config["dynamic_inter_max_neighbors"]),
        dynamic_residue_cutoff=float(model_config["dynamic_residue_cutoff"]),
        dynamic_residue_knn_k=int(model_config["dynamic_residue_knn_k"]),
        dynamic_residue_max_neighbors=int(model_config["dynamic_residue_max_neighbors"]),
        dynamic_residue_candidate_topk=int(model_config["dynamic_residue_candidate_topk"]),
    )
    return model.to(device) if device is not None else model
