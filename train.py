"""
训练命令入口。

负责解析训练配置、补全运行参数、初始化日志与输出目录，
并调用训练主流程启动一次完整训练。
"""


import argparse
import json
import logging
import os
import sys
import tomllib

from pathlib import Path

# 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

def _resolve_initial_config_path(argv: list[str], default_path: Path) -> Path:
    """
    在导入 torch 前解析早期配置文件路径。

    Args:
        argv: 当前进程接收到的命令行参数列表（不含程序名）。
        default_path: 未显式传入 `--config` 时使用的默认配置路径。

    Returns:
        Path: 返回本次启动应优先读取的训练配置文件路径。
    """
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return default_path


def _load_cuda_allocator_config(config_path: Path) -> str | None:
    """
    从训练配置读取 CUDA allocator 配置字符串。

    Args:
        config_path: 待读取的训练配置文件路径。

    Returns:
        str | None: 返回配置文件中声明的 CUDA allocator 配置；若缺失则返回 `None`。
    """
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        return None
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    runtime_config = raw_config.get("runtime", {})
    value = runtime_config.get("torch_cuda_alloc_conf")
    if value is None:
        return None
    configured = str(value).strip()
    return configured or None


_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.toml"
_initial_config_path = _resolve_initial_config_path(sys.argv[1:], _DEFAULT_CONFIG_PATH)
_cuda_allocator_config = _load_cuda_allocator_config(_initial_config_path)

# 必须在导入 `torch` 之前设置 CUDA 显存分配配置，避免初始化后配置失效。
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and _cuda_allocator_config is not None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _cuda_allocator_config

from ehfnet.training import train
from ehfnet.data.featurizers import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.runtime import (
    build_run_suffix,
    configure_text_logging,
    load_flattened_toml_config,
    load_train_defaults,
)
import torch

# 全局开启 TF32 以加速矩阵运算。
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def _resolve_auto_cutoffs(
    config: dict[str, object],
    data_root: str,
    *,
    project_root: Path,
) -> None:
    """
    将 config 中值为 'auto' 的 cutoff 解析为 dataset_profile.json 中的建议值。

    Args:
        config: 待原地更新的模型配置字典。
        data_root: 当前训练使用的数据目录。
        project_root: 项目根目录路径。
    """
    profile_path = Path(data_root) / "dataset_profile.json"
    if not profile_path.exists():
        return
    try:
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        return
    suggested = profile.get("suggested_cutoffs", {})
    # 将配置键映射到统计文件中的建议截断半径键。
    key_map = {
        "dynamic_inter_cutoff": "ligand_atom-protein_atom",
        "force_cutoff": "ligand_atom-protein_atom",
    }
    for cfg_key, profile_key in key_map.items():
        if cfg_key in config and config[cfg_key] == "auto":
            val = suggested.get(profile_key)
            if val is not None:
                config[cfg_key] = float(val)


def _require_config_keys(
    config: dict[str, object],
    *,
    keys: list[str],
) -> None:
    """
    校验训练默认配置已显式包含所需键。

    Args:
        config: 已加载并合并完成的训练配置字典。
        keys: 必须在配置中显式提供的键名列表。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    missing_keys = [key for key in keys if key not in config]
    if missing_keys:
        raise ValueError(
            "Training config is incomplete. Missing keys: "
            f"{missing_keys}. Please complete configs/train.toml and configs/model.toml."
        )


def main():
    """
    训练入口函数。

    负责解析命令行参数、加载配置默认值并补全运行参数，
    随后初始化日志和输出目录并启动完整训练流程。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """
    default_config_path = PROJECT_ROOT / "configs" / "train.toml"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path), help="Path to TOML config file")
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_train_defaults(
        config_path=pre_args.config,
        project_root=PROJECT_ROOT,
    )
    # 解析 `auto` 截断半径：从 `dataset_profile.json` 读取建议值。
    data_root_for_auto = config_defaults.get("data_root") or None
    if data_root_for_auto:
        model_config_path = config_defaults.get("model_config")
        model_defaults = load_flattened_toml_config(
            str(model_config_path) if model_config_path is not None else None,
            project_root=PROJECT_ROOT,
        )
        _resolve_auto_cutoffs(
            model_defaults,
            str(data_root_for_auto),
            project_root=PROJECT_ROOT,
        )
        config_defaults.update(model_defaults)
    _require_config_keys(
        config_defaults,
        keys=[
            "data_root",
            "model_config",
            "save_dir",
            "device",
            "esm",
            "hidden_dim",
            "num_gnn_blocks",
            "m_dim_scalar",
            "dropout_rate",
            "esm_dim",
            "esm_model_name",
            "epochs",
            "lr",
            "weight_decay",
            "clip_grad",
            "warmup_epochs",
            "accumulation_steps",
            "ema_decay",
            "val_subset_ratio",
            "val_full_every",
            "val_full_last_epochs",
            "geometry_min_atom_distance",
            "ode_method",
            "min_checkpoint_selection_coverage",
            "max_val_non_oom_failures",
            "max_val_oom_failures",
            "final_topn_min_coverage",
            "split_train_frac",
            "split_val_frac",
            "split_test_frac",
            "split_seed",
            "train_cost_budget",
            "val_cost_budget",
            "blind_pool_cost_budget",
            "final_topn_cost_budget",
            "eval_cost_guard_headroom",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
            "max_oom_retry_splits",
            "enable_train_budget_callback",
            "oom_reduce_threshold",
            "oom_reduce_factor",
            "min_train_cost_budget",
            "enable_val_budget_callback",
            "val_oom_reduce_threshold",
            "val_oom_reduce_factor",
            "min_val_cost_budget",
            "train_budget_window_size",
            "train_budget_recover_window_count",
            "train_budget_recover_step",
            "train_offender_cooldown",
            "val_budget_window_size",
            "val_budget_recover_window_count",
            "val_budget_recover_step",
            "val_offender_cooldown",
            "split_cache_file",
            "force_resplit",
            "test_topk",
            "val_ode_steps",
            "center_proposal_topk",
            "center_refine_topk",
            "center_nms_radius",
            "stage1_pose_samples",
            "stage2_pose_samples",
            "center_proposal_weight",
            "center_positive_radius",
            "center_guidance_learned_start",
            "crop_candidate_topk",
            "crop_proposal_start",
            "crop_near_miss_start",
            "crop_hard_negative_start",
            "crop_min_residues",
            "crop_atom_margin",
            "disable_jitter_crop",
            "disable_hard_negative_crop",
            "pose_ranking_pair_weight",
            "pose_ranking_margin",
            "ranking_same_center_start",
            "ranking_wrong_center_start",
            "pose_bootstrap_weight",
            "pose_bootstrap_start",
            "pose_bootstrap_frequency",
            "pose_bootstrap_ode_steps",
            "blind_pool_refresh_every",
            "blind_pool_start_epoch",
            "blind_pool_refresh_on_best_update",
            "blind_pool_max_complexes",
            "blind_pool_cache_bce_weight",
            "blind_pool_cache_rank_weight",
            "blind_pool_pairs_per_complex",
            "replay_start_ratio",
            "same_center_micro_batch_size",
            "same_center_budget_window_size",
            "same_center_budget_recover_window_count",
            "same_center_budget_recover_step",
            "same_center_offender_cooldown",
            "ranking_budget_window_size",
            "ranking_budget_recover_window_count",
            "ranking_offender_cooldown",
            "ranking_wrong_center_cap",
            "replay_micro_batch_size",
            "replay_budget_window_size",
            "replay_budget_recover_window_count",
            "replay_candidate_cooldown",
            "replay_max_candidates_per_complex",
            "checkpoint_selection_mode",
            "ablation_mode",
            "run_test_after_training",
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
            "dynamic_inter_cutoff",
            "dynamic_inter_knn_k",
            "dynamic_inter_max_neighbors",
            "dynamic_residue_cutoff",
            "dynamic_residue_knn_k",
            "dynamic_residue_max_neighbors",
            "dynamic_residue_candidate_topk",
            "r_cutoff_intra",
            "max_neighbors_intra",
            "atom_neighbor_cap",
            "residue_neighbor_cap",
            "residue_radius_scale",
            "residue_radius_bias",
            "ligand_atom_fallback_k",
            "protein_atom_fallback_k",
            "protein_residue_fallback_k",
            "loss_characteristic_scale",
            "loss_weight_translation",
            "loss_weight_rotation",
            "loss_weight_torsion",
            "loss_weight_energy",
            "loss_weight_clash",
            "loss_weight_pose_rank",
            "loss_coarse_translation",
            "loss_coarse_rotation",
            "loss_coarse_torsion",
            "loss_coarse_energy",
            "loss_coarse_clash",
            "loss_coarse_pose_rank",
            "loss_transition_translation",
            "loss_transition_rotation",
            "loss_transition_torsion",
            "loss_transition_energy",
            "loss_transition_clash",
            "loss_transition_pose_rank",
            "loss_refine_translation",
            "loss_refine_rotation",
            "loss_refine_torsion",
            "loss_refine_energy",
            "loss_refine_clash",
            "loss_refine_pose_rank",
            "loss_refine_start",
            "loss_pose_gate_epoch_start",
            "loss_pose_gate_epoch_end",
            "loss_pose_gate_tau_start",
            "loss_pose_gate_tau_end",
            "loss_pose_gate_temperature",
        ],
    )

    parser = argparse.ArgumentParser(
        description="Train EHFNet for molecular docking prediction",
        parents=[pre_parser],
    )

    # 数据相关参数：只接受一个包含 `index.csv` 的数据目录，不允许自定义索引路径。
    parser.add_argument(
        "--data_root",
        type=str,
        default=config_defaults["data_root"],
        help="Processed data root containing index.csv, e.g. data/processed/hiqbind",
    )
    parser.add_argument(
        "--esm",
        type=str,
        default=config_defaults["esm"],
        help="ESM processing mode: auto, file, or off",
    )
    parser.add_argument("--save_dir", type=str, default=config_defaults["save_dir"], help="Directory to save checkpoints")
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint path used to resume training state",
    )
    parser.add_argument(
        "--resume_blind_pool_dir",
        type=str,
        default=None,
        help="Optional blind pool cache directory used when resuming into a new run directory",
    )
    parser.add_argument(
        "--stop_after_epoch",
        type=int,
        default=None,
        help="Stop after the specified absolute epoch number (1-based, inclusive)",
    )
    parser.add_argument("--esm_path", type=str, default=config_defaults.get("esm_path"), help="Path to precomputed ESM embeddings (optional)")

    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=config_defaults["epochs"], help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=config_defaults["lr"], help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config_defaults["weight_decay"], help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=config_defaults["clip_grad"], help="Gradient clipping value")

    # 模型相关参数
    parser.add_argument("--hidden_dim", type=int, default=config_defaults["hidden_dim"], help="Hidden dimension size")
    parser.add_argument("--num_gnn_blocks", type=int, default=config_defaults["num_gnn_blocks"], help="Number of GNN blocks")
    parser.add_argument(
        "--m_dim_scalar",
        type=int,
        default=config_defaults["m_dim_scalar"],
        help="EGNN message dimension (from model.toml)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=config_defaults["dropout_rate"],
        help="Model dropout rate (from model.toml)",
    )

    # 特征相关参数（通常固定，但为灵活性暴露）
    parser.add_argument("--lig_atom_cont_count", type=int, default=len(LIGAND_ATOM_CONT_SCHEMA), help="Ligand atom continuous feature count")
    parser.add_argument("--lig_mol_cont_count", type=int, default=len(LIGAND_MOLECULE_CONT_SCHEMA), help="Ligand molecule continuous feature count")
    parser.add_argument("--pro_atom_cont_count", type=int, default=len(PROTEIN_ATOM_CONT_SCHEMA), help="Protein atom continuous feature count")
    parser.add_argument("--esm_model_name", type=str, default=config_defaults["esm_model_name"], help="ESM backbone model name used for residue embedding preprocessing")
    parser.add_argument("--esm_dim", type=int, default=config_defaults["esm_dim"], help="ESM embedding dimension (default: 960 for ESMC-300M)")
    parser.add_argument("--num_rbf", type=int, default=config_defaults["num_rbf"], help="RBF basis count (from model.toml)")
    parser.add_argument("--r_cutoff", type=float, default=config_defaults["r_cutoff"], help="Distance cutoff in Å (from model.toml)")
    parser.add_argument("--force_cutoff", type=float, default=config_defaults["force_cutoff"], help="Force branch local radius in Å (from model.toml)")
    parser.add_argument("--dynamic_inter_cutoff", type=float, default=config_defaults["dynamic_inter_cutoff"], help="Dynamic inter-atom edge radius (from model.toml)")
    parser.add_argument("--dynamic_inter_knn_k", type=int, default=config_defaults["dynamic_inter_knn_k"], help="kNN fallback for inter-atom edges (from model.toml)")
    parser.add_argument("--dynamic_inter_max_neighbors", type=int, default=config_defaults["dynamic_inter_max_neighbors"], help="Per-source neighbor cap for dynamic inter-atom edges (from model.toml)")
    parser.add_argument("--dynamic_residue_cutoff", type=float, default=config_defaults["dynamic_residue_cutoff"], help="Dynamic ligand-residue edge radius (from model.toml)")
    parser.add_argument("--dynamic_residue_knn_k", type=int, default=config_defaults["dynamic_residue_knn_k"], help="kNN fallback for ligand-residue edges (from model.toml)")
    parser.add_argument("--dynamic_residue_max_neighbors", type=int, default=config_defaults["dynamic_residue_max_neighbors"], help="Per-source neighbor cap for dynamic ligand-residue edges (from model.toml)")
    parser.add_argument("--dynamic_residue_candidate_topk", type=int, default=config_defaults["dynamic_residue_candidate_topk"], help="Graph-level top-k residue candidates kept before building ligand-residue dynamic edges (from model.toml)")
    parser.add_argument("--frame_refine_threshold", type=float, default=config_defaults["frame_refine_threshold"], help="Frame refinement gate threshold (from model.toml)")
    parser.add_argument("--frame_refine_temperature", type=float, default=config_defaults["frame_refine_temperature"], help="Frame refinement gate temperature (from model.toml)")
    parser.add_argument("--energy_guide_threshold", type=float, default=config_defaults["energy_guide_threshold"], help="Energy guidance gate threshold (from model.toml)")
    parser.add_argument("--energy_guide_temperature", type=float, default=config_defaults["energy_guide_temperature"], help="Energy guidance gate temperature (from model.toml)")
    parser.add_argument("--clash_threshold", type=float, default=config_defaults["clash_threshold"], help="Clash penalty threshold in Å (from model.toml)")
    parser.add_argument("--clash_push_threshold", type=float, default=config_defaults["clash_push_threshold"], help="Clash repulsion threshold in Å (from model.toml)")
    parser.add_argument("--clash_push_force", type=float, default=config_defaults["clash_push_force"], help="Clash repulsion force scale (from model.toml)")
    parser.add_argument("--score_clamp_min", type=float, default=config_defaults["score_clamp_min"], help="Score clamp minimum (from model.toml)")
    parser.add_argument("--score_clamp_max", type=float, default=config_defaults["score_clamp_max"], help="Score clamp maximum (from model.toml)")
    parser.add_argument("--force_limit", type=float, default=config_defaults["force_limit"], help="Force magnitude soft limit (from model.toml)")
    parser.add_argument("--max_neighbors", type=int, default=config_defaults["max_neighbors"], help="Prediction head neighbor cap (from model.toml)")
    parser.add_argument("--min_max_neighbors", type=int, default=config_defaults["min_max_neighbors"], help="Prediction head adaptive neighbor floor (from model.toml)")
    parser.add_argument("--knn_fallback_k", type=int, default=config_defaults["knn_fallback_k"], help="Prediction head kNN fallback neighbors (from model.toml)")
    parser.add_argument("--r_cutoff_intra", type=float, default=config_defaults["r_cutoff_intra"], help="Intra-graph atom radius cutoff (from model.toml)")
    parser.add_argument("--max_neighbors_intra", type=int, default=config_defaults["max_neighbors_intra"], help="Intra-graph neighbor cap before typed clipping (from model.toml)")
    parser.add_argument("--atom_neighbor_cap", type=int, default=config_defaults["atom_neighbor_cap"], help="Typed cap for ligand/protein atom intra edges (from model.toml)")
    parser.add_argument("--residue_neighbor_cap", type=int, default=config_defaults["residue_neighbor_cap"], help="Typed cap for residue intra edges (from model.toml)")
    parser.add_argument("--residue_radius_scale", type=float, default=config_defaults["residue_radius_scale"], help="Residue radius scale over atom radius (from model.toml)")
    parser.add_argument("--residue_radius_bias", type=float, default=config_defaults["residue_radius_bias"], help="Residue radius additive bias in Å (from model.toml)")
    parser.add_argument("--ligand_atom_fallback_k", type=int, default=config_defaults["ligand_atom_fallback_k"], help="Ligand atom intra-edge kNN fallback (from model.toml)")
    parser.add_argument("--protein_atom_fallback_k", type=int, default=config_defaults["protein_atom_fallback_k"], help="Protein atom intra-edge kNN fallback (from model.toml)")
    parser.add_argument("--protein_residue_fallback_k", type=int, default=config_defaults["protein_residue_fallback_k"], help="Protein residue intra-edge kNN fallback (from model.toml)")
    parser.add_argument("--flow_sigma_min", type=float, default=config_defaults["flow_sigma_min"], help="Flow matching minimum time margin")
    parser.add_argument("--flow_spatial_sigma_min", type=float, default=config_defaults["flow_spatial_sigma_min"], help="Flow matching minimum translation scale")
    parser.add_argument("--flow_spatial_sigma_max", type=float, default=config_defaults["flow_spatial_sigma_max"], help="Flow matching maximum translation scale")
    parser.add_argument("--flow_fd_dt", type=float, default=config_defaults["flow_fd_dt"], help="Finite-difference step for flow target generation")
    parser.add_argument("--flow_rotation_angle_min", type=float, default=config_defaults["flow_rotation_angle_min"], help="Flow curriculum minimum rotation angle in radians")
    parser.add_argument("--flow_rotation_angle_max", type=float, default=config_defaults["flow_rotation_angle_max"], help="Flow curriculum maximum rotation angle in radians")
    parser.add_argument("--flow_torsion_scale_min", type=float, default=config_defaults["flow_torsion_scale_min"], help="Flow curriculum minimum torsion scale")
    parser.add_argument("--flow_torsion_scale_max", type=float, default=config_defaults["flow_torsion_scale_max"], help="Flow curriculum maximum torsion scale")
    parser.add_argument("--loss_characteristic_scale", type=float, default=config_defaults["loss_characteristic_scale"], help="Characteristic length scale used to balance translation and rotation losses")
    parser.add_argument("--loss_weight_translation", type=float, default=config_defaults["loss_weight_translation"], help="Global multiplier for translation loss")
    parser.add_argument("--loss_weight_rotation", type=float, default=config_defaults["loss_weight_rotation"], help="Global multiplier for rotation loss")
    parser.add_argument("--loss_weight_torsion", type=float, default=config_defaults["loss_weight_torsion"], help="Global multiplier for torsion loss")
    parser.add_argument("--loss_weight_energy", type=float, default=config_defaults["loss_weight_energy"], help="Global multiplier for affinity loss")
    parser.add_argument("--loss_weight_clash", type=float, default=config_defaults["loss_weight_clash"], help="Global multiplier for clash loss")
    parser.add_argument("--loss_weight_pose_rank", type=float, default=config_defaults["loss_weight_pose_rank"], help="Global multiplier for pose-rank BCE loss")
    parser.add_argument("--loss_coarse_translation", type=float, default=config_defaults["loss_coarse_translation"], help="Coarse curriculum translation weight")
    parser.add_argument("--loss_coarse_rotation", type=float, default=config_defaults["loss_coarse_rotation"], help="Coarse curriculum rotation weight")
    parser.add_argument("--loss_coarse_torsion", type=float, default=config_defaults["loss_coarse_torsion"], help="Coarse curriculum torsion weight")
    parser.add_argument("--loss_coarse_energy", type=float, default=config_defaults["loss_coarse_energy"], help="Coarse curriculum affinity weight")
    parser.add_argument("--loss_coarse_clash", type=float, default=config_defaults["loss_coarse_clash"], help="Coarse curriculum clash weight")
    parser.add_argument("--loss_coarse_pose_rank", type=float, default=config_defaults["loss_coarse_pose_rank"], help="Coarse curriculum pose-rank BCE weight")
    parser.add_argument("--loss_transition_translation", type=float, default=config_defaults["loss_transition_translation"], help="Transition curriculum translation weight")
    parser.add_argument("--loss_transition_rotation", type=float, default=config_defaults["loss_transition_rotation"], help="Transition curriculum rotation weight")
    parser.add_argument("--loss_transition_torsion", type=float, default=config_defaults["loss_transition_torsion"], help="Transition curriculum torsion weight")
    parser.add_argument("--loss_transition_energy", type=float, default=config_defaults["loss_transition_energy"], help="Transition curriculum affinity weight")
    parser.add_argument("--loss_transition_clash", type=float, default=config_defaults["loss_transition_clash"], help="Transition curriculum clash weight")
    parser.add_argument("--loss_transition_pose_rank", type=float, default=config_defaults["loss_transition_pose_rank"], help="Transition curriculum pose-rank BCE weight")
    parser.add_argument("--loss_refine_translation", type=float, default=config_defaults["loss_refine_translation"], help="Refine curriculum translation weight")
    parser.add_argument("--loss_refine_rotation", type=float, default=config_defaults["loss_refine_rotation"], help="Refine curriculum rotation weight")
    parser.add_argument("--loss_refine_torsion", type=float, default=config_defaults["loss_refine_torsion"], help="Refine curriculum torsion weight")
    parser.add_argument("--loss_refine_energy", type=float, default=config_defaults["loss_refine_energy"], help="Refine curriculum affinity weight")
    parser.add_argument("--loss_refine_clash", type=float, default=config_defaults["loss_refine_clash"], help="Refine curriculum clash weight")
    parser.add_argument("--loss_refine_pose_rank", type=float, default=config_defaults["loss_refine_pose_rank"], help="Refine curriculum pose-rank BCE weight")
    parser.add_argument("--loss_refine_start", type=float, default=config_defaults["loss_refine_start"], help="Progress threshold where refine curriculum starts")
    parser.add_argument("--loss_pose_gate_epoch_start", type=float, default=config_defaults["loss_pose_gate_epoch_start"], help="Pose gate smoothstep start over training progress")
    parser.add_argument("--loss_pose_gate_epoch_end", type=float, default=config_defaults["loss_pose_gate_epoch_end"], help="Pose gate smoothstep end over training progress")
    parser.add_argument("--loss_pose_gate_tau_start", type=float, default=config_defaults["loss_pose_gate_tau_start"], help="Initial t-threshold for pose-focused losses")
    parser.add_argument("--loss_pose_gate_tau_end", type=float, default=config_defaults["loss_pose_gate_tau_end"], help="Final t-threshold for pose-focused losses")
    parser.add_argument("--loss_pose_gate_temperature", type=float, default=config_defaults["loss_pose_gate_temperature"], help="Temperature for pose-focused time gating")
    parser.add_argument(
        "--device",
        type=str,
        default=str(config_defaults["device"]),
        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu')",
    )
    parser.add_argument(
        "--run_suffix",
        type=str,
        default=None,
        help="Optional run suffix used to align train logs, artifacts, and external nohup logs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=bool(config_defaults["smoke"]),
        help="Write text logs under logs/smoke/... for easier smoke-run cleanup.",
    )
    parser.add_argument(
        "--no-smoke",
        dest="smoke",
        action="store_false",
        help="Disable smoke log grouping and use the default logs/... directory.",
    )
    parser.add_argument("--crop_radius", type=float, default=config_defaults["crop_radius"], help="Runtime local crop radius in angstroms (default: 10.0)")
    parser.add_argument("--warmup_epochs", type=int, default=config_defaults["warmup_epochs"], help="Number of warmup epochs for spatial curriculum learning (default: 20)")
    parser.add_argument(
        "--val_subset_ratio",
        type=float,
        default=config_defaults["val_subset_ratio"],
        help="Fraction of validation graphs used by the default partial validation schedule",
    )
    parser.add_argument(
        "--val_full_every",
        type=int,
        default=config_defaults["val_full_every"],
        help="Run full lightweight validation every N epochs (0 disables periodic full validation)",
    )
    parser.add_argument(
        "--val_full_last_epochs",
        type=int,
        default=config_defaults["val_full_last_epochs"],
        help="Always run full lightweight validation during the last N epochs",
    )
    parser.add_argument(
        "--geometry_min_atom_distance",
        type=float,
        default=config_defaults["geometry_min_atom_distance"],
        help="Minimum ligand atom distance used by the preprocessing filter",
    )
    parser.add_argument(
        "--min_checkpoint_selection_coverage",
        type=float,
        default=config_defaults["min_checkpoint_selection_coverage"],
        help="Minimum validation coverage required before updating best checkpoints",
    )
    parser.add_argument(
        "--max_val_non_oom_failures",
        type=int,
        default=config_defaults["max_val_non_oom_failures"],
        help="Maximum non-OOM validation failures allowed before validation fails",
    )
    parser.add_argument(
        "--max_val_oom_failures",
        type=int,
        default=config_defaults["max_val_oom_failures"],
        help="Maximum irreducible validation OOM failures allowed before validation fails",
    )
    parser.add_argument(
        "--final_topn_min_coverage",
        type=float,
        default=config_defaults["final_topn_min_coverage"],
        help="Minimum final Top-N evaluation coverage required for official metrics",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default=config_defaults["ode_method"],
        choices=["euler", "rk4"],
        help="ODE solver used by validation, bootstrap, blind-pool refresh, and final Top-N evaluation",
    )
    parser.add_argument("--accumulation_steps", type=int, default=config_defaults["accumulation_steps"], help="Gradient accumulation steps")
    parser.add_argument("--train_cost_budget", type=int, default=config_defaults["train_cost_budget"], help="Static cost budget used by training; it only shrinks after real OOM")
    parser.add_argument("--val_cost_budget", type=int, default=config_defaults["val_cost_budget"], help="Static cost budget used by training-time validation")
    parser.add_argument("--blind_pool_cost_budget", type=int, default=config_defaults["blind_pool_cost_budget"], help="Static cost budget used by blind-pool refresh")
    parser.add_argument("--final_topn_cost_budget", type=int, default=config_defaults["final_topn_cost_budget"], help="Static cost budget used by final blind Top-N evaluation")
    parser.add_argument("--eval_cost_guard_headroom", type=float, default=config_defaults["eval_cost_guard_headroom"], help="Extra headroom multiplier applied to evaluation cost guards")
    parser.add_argument("--ema_decay", type=float, default=config_defaults["ema_decay"], help="EMA decay rate (default: 0.999; use 0.99 for quick smoke tests)")
    parser.add_argument("--dataloader_num_workers", type=int, default=config_defaults["dataloader_num_workers"], help="DataLoader worker count")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=bool(config_defaults["dataloader_pin_memory"]), help="Enable DataLoader pin_memory")
    parser.add_argument("--no_dataloader_pin_memory", dest="dataloader_pin_memory", action="store_false", help="Disable DataLoader pin_memory")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=bool(config_defaults["dataloader_persistent_workers"]), help="Enable DataLoader persistent_workers")
    parser.add_argument("--no_dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store_false", help="Disable DataLoader persistent_workers")
    parser.add_argument("--max_oom_retry_splits", type=int, default=config_defaults["max_oom_retry_splits"], help="Maximum recursive split depth used to retry an OOM batch")
    parser.add_argument("--enable_train_budget_callback", action="store_true", default=bool(config_defaults["enable_train_budget_callback"]), help="Enable window-based train budget callback after real CUDA OOM events")
    parser.add_argument("--disable_train_budget_callback", dest="enable_train_budget_callback", action="store_false", help="Disable the train budget callback and keep train_cost_budget fixed")
    parser.add_argument("--oom_reduce_threshold", type=int, default=config_defaults["oom_reduce_threshold"], help="Reduce train cost budget when OOM batches in an epoch reach this threshold")
    parser.add_argument("--oom_reduce_factor", type=float, default=config_defaults["oom_reduce_factor"], help="Factor to shrink train_cost_budget after OOM threshold (0-1)")
    parser.add_argument("--min_train_cost_budget", type=int, default=config_defaults["min_train_cost_budget"], help="Lower bound for adaptive train_cost_budget")
    parser.add_argument("--enable_val_budget_callback", action="store_true", default=bool(config_defaults["enable_val_budget_callback"]), help="Enable window-based validation budget callback after validation OOM events")
    parser.add_argument("--disable_val_budget_callback", dest="enable_val_budget_callback", action="store_false", help="Disable the validation budget callback and keep val_cost_budget fixed")
    parser.add_argument("--val_oom_reduce_threshold", type=int, default=config_defaults["val_oom_reduce_threshold"], help="Reduce validation cost budget when validation OOM batches reach this threshold")
    parser.add_argument("--val_oom_reduce_factor", type=float, default=config_defaults["val_oom_reduce_factor"], help="Factor to shrink val_cost_budget after OOM threshold (0-1)")
    parser.add_argument("--min_val_cost_budget", type=int, default=config_defaults["min_val_cost_budget"], help="Lower bound for adaptive val_cost_budget")
    parser.add_argument("--train_budget_window_size", type=int, default=config_defaults["train_budget_window_size"], help="Root-batch window size used by the train cost-budget callback")
    parser.add_argument("--train_budget_recover_window_count", type=int, default=config_defaults["train_budget_recover_window_count"], help="Number of clean train windows required before additive budget recovery")
    parser.add_argument("--train_budget_recover_step", type=int, default=config_defaults["train_budget_recover_step"], help="Additive recovery step for train_cost_budget after clean windows")
    parser.add_argument("--train_offender_cooldown", type=int, default=config_defaults["train_offender_cooldown"], help="Cooldown length for train offenders, measured in root-batch events")
    parser.add_argument("--val_budget_window_size", type=int, default=config_defaults["val_budget_window_size"], help="Validation callback window size for partial/full budget recovery")
    parser.add_argument("--val_budget_recover_window_count", type=int, default=config_defaults["val_budget_recover_window_count"], help="Number of clean validation windows required before additive budget recovery")
    parser.add_argument("--val_budget_recover_step", type=int, default=config_defaults["val_budget_recover_step"], help="Additive recovery step for validation budgets after clean windows")
    parser.add_argument("--val_offender_cooldown", type=int, default=config_defaults["val_offender_cooldown"], help="Cooldown length for validation offenders, measured in validation events")
    parser.add_argument("--split_train_frac", type=float, default=config_defaults["split_train_frac"], help="Train split fraction")
    parser.add_argument("--split_val_frac", type=float, default=config_defaults["split_val_frac"], help="Validation split fraction")
    parser.add_argument("--split_test_frac", type=float, default=config_defaults["split_test_frac"], help="Test split fraction")
    parser.add_argument("--split_seed", type=int, default=config_defaults["split_seed"], help="Seed for scaffold split")
    parser.add_argument("--split_cache_file", type=str, default=config_defaults["split_cache_file"], help="Path to persisted split JSON")
    parser.add_argument("--force_resplit", action="store_true", help="Force regenerate split JSON")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=config_defaults["ablation_mode"],
        choices=["none", "inter_multiscale_off"],
        help="Ablation mode: none (full model), inter_multiscale_off (atom-atom only)",
    )
    parser.add_argument("--run_test_after_training", action="store_true", default=bool(config_defaults["run_test_after_training"]), help="Run final test-set evaluation after training")
    parser.add_argument("--skip_test_after_training", dest="run_test_after_training", action="store_false", help="Skip final test-set evaluation")
    parser.add_argument("--test_topk", type=str, default=config_defaults["test_topk"], help="Comma-separated top-k values for final test evaluation")
    parser.add_argument("--center_proposal_weight", type=float, default=config_defaults["center_proposal_weight"], help="Shared loss weight for online center supervision and replay-based center value supervision")
    parser.add_argument("--center_positive_radius", type=float, default=config_defaults["center_positive_radius"], help="Hit radius in angstroms used by crop curriculum and blind center metrics")
    parser.add_argument("--center_guidance_learned_start", type=float, default=config_defaults["center_guidance_learned_start"], help="Training progress where crop-center scoring starts blending heuristic priors with learned proposal logits")
    parser.add_argument("--center_proposal_topk", type=int, default=config_defaults["center_proposal_topk"], help="Number of diverse residue centers kept in stage-1 proposal")
    parser.add_argument("--center_refine_topk", type=int, default=config_defaults["center_refine_topk"], help="Number of centers refined in stage-2 local docking")
    parser.add_argument("--center_nms_radius", type=float, default=config_defaults["center_nms_radius"], help="Diversity radius in angstroms for center NMS")
    parser.add_argument("--stage1_pose_samples", type=int, default=config_defaults["stage1_pose_samples"], help="Number of local docking samples per center in stage-1")
    parser.add_argument("--stage2_pose_samples", type=int, default=config_defaults["stage2_pose_samples"], help="Number of local docking samples per center in stage-2")
    parser.add_argument("--crop_candidate_topk", type=int, default=config_defaults["crop_candidate_topk"], help="Top-k proposal candidates used for weighted crop sampling in each curriculum bucket")
    parser.add_argument("--crop_proposal_start", type=float, default=config_defaults["crop_proposal_start"], help="Training progress where proposal-positive crop centers enter the curriculum")
    parser.add_argument("--crop_near_miss_start", type=float, default=config_defaults["crop_near_miss_start"], help="Training progress where near-miss crop centers enter the curriculum")
    parser.add_argument("--crop_hard_negative_start", type=float, default=config_defaults["crop_hard_negative_start"], help="Training progress where hard-negative crop centers enter the curriculum")
    parser.add_argument("--crop_min_residues", type=int, default=config_defaults["crop_min_residues"], help="Minimum protein residues kept in runtime local crop")
    parser.add_argument("--crop_atom_margin", type=float, default=config_defaults["crop_atom_margin"], help="Extra atom-distance margin used by runtime local crop")
    parser.add_argument("--disable_jitter_crop", action="store_true", default=bool(config_defaults["disable_jitter_crop"]), help="Disable jitter crop branch for ablation")
    parser.add_argument("--disable_hard_negative_crop", action="store_true", default=bool(config_defaults["disable_hard_negative_crop"]), help="Disable hard-negative crop branch for ablation")
    parser.add_argument("--pose_ranking_pair_weight", type=float, default=config_defaults["pose_ranking_pair_weight"], help="Pairwise ranking loss weight for pose confidence")
    parser.add_argument("--pose_ranking_margin", type=float, default=config_defaults["pose_ranking_margin"], help="Margin used by the pairwise ranking loss")
    parser.add_argument("--ranking_same_center_start", type=float, default=config_defaults["ranking_same_center_start"], help="Training progress where same-center pairwise ranking becomes active")
    parser.add_argument("--ranking_wrong_center_start", type=float, default=config_defaults["ranking_wrong_center_start"], help="Training progress where wrong-center pairwise ranking becomes active")
    parser.add_argument("--same_center_micro_batch_size", type=int, default=config_defaults["same_center_micro_batch_size"], help="Initial micro-batch size used by the online same-center ranking branch")
    parser.add_argument("--same_center_budget_window_size", type=int, default=config_defaults["same_center_budget_window_size"], help="Window size used to recover same-center ranking after OOM")
    parser.add_argument("--same_center_budget_recover_window_count", type=int, default=config_defaults["same_center_budget_recover_window_count"], help="Number of clean same-center windows required before growing the same-center micro-batch")
    parser.add_argument("--same_center_budget_recover_step", type=int, default=config_defaults["same_center_budget_recover_step"], help="Additive recovery step for same-center ranking micro-batches")
    parser.add_argument("--same_center_offender_cooldown", type=int, default=config_defaults["same_center_offender_cooldown"], help="Cooldown length for root samples that repeatedly trigger same-center ranking OOM")
    parser.add_argument("--ranking_budget_window_size", type=int, default=config_defaults["ranking_budget_window_size"], help="Window size used to recover wrong-center ranking after OOM")
    parser.add_argument("--ranking_budget_recover_window_count", type=int, default=config_defaults["ranking_budget_recover_window_count"], help="Number of clean ranking windows required before re-enabling wrong-center ranking")
    parser.add_argument("--ranking_offender_cooldown", type=int, default=config_defaults["ranking_offender_cooldown"], help="Cooldown length for ranking offenders that repeatedly trigger wrong-center OOM")
    parser.add_argument("--ranking_wrong_center_cap", type=int, default=config_defaults["ranking_wrong_center_cap"], help="Maximum wrong-center ranking branch level; 1 means enabled, 0 means same-center only")
    parser.add_argument("--pose_bootstrap_weight", type=float, default=config_defaults["pose_bootstrap_weight"], help="Bootstrap loss weight for model-generated poses")
    parser.add_argument("--pose_bootstrap_start", type=float, default=config_defaults["pose_bootstrap_start"], help="Training progress where bootstrap ranking supervision becomes active")
    parser.add_argument("--pose_bootstrap_frequency", type=int, default=config_defaults["pose_bootstrap_frequency"], help="Run bootstrap scoring every N training batches (0 disables)")
    parser.add_argument("--pose_bootstrap_ode_steps", type=int, default=config_defaults["pose_bootstrap_ode_steps"], help="ODE steps used to generate bootstrap poses")
    parser.add_argument("--val_ode_steps", type=int, default=config_defaults["val_ode_steps"], help="Shared ODE integration steps for training-time validation and final blind test evaluation (default: 50)")
    parser.add_argument(
        "--checkpoint_selection_mode",
        type=str,
        default=config_defaults["checkpoint_selection_mode"],
        choices=[
            "composite",
            "mean_rmsd",
            "val_loss",
            "single_shot_success_2a",
            "single_shot_success_5a",
            "rmsd_priority",
        ],
        help="Primary lightweight-validation metric used for best_selected_model checkpoint selection",
    )
    parser.add_argument("--blind_pool_refresh_every", type=int, default=config_defaults["blind_pool_refresh_every"], help="Refresh blind candidate pool every N epochs")
    parser.add_argument("--blind_pool_start_epoch", type=int, default=config_defaults["blind_pool_start_epoch"], help="Earliest epoch to start pool refresh")
    parser.add_argument("--blind_pool_refresh_on_best_update", action="store_true", default=bool(config_defaults["blind_pool_refresh_on_best_update"]), help="Also refresh the blind candidate pool immediately when a new best checkpoint is selected")
    parser.add_argument("--blind_pool_max_complexes", type=int, default=config_defaults["blind_pool_max_complexes"], help="Max complexes per pool refresh")
    parser.add_argument("--blind_pool_cache_bce_weight", type=float, default=config_defaults["blind_pool_cache_bce_weight"], help="Cache BCE loss weight")
    parser.add_argument("--blind_pool_cache_rank_weight", type=float, default=config_defaults["blind_pool_cache_rank_weight"], help="Cache pairwise ranking loss weight")
    parser.add_argument("--blind_pool_pairs_per_complex", type=int, default=config_defaults["blind_pool_pairs_per_complex"], help="Hard pairs sampled per complex from cache")
    parser.add_argument("--replay_start_ratio", type=float, default=config_defaults["replay_start_ratio"], help="Training progress where replay-based reranking becomes active")
    parser.add_argument("--replay_micro_batch_size", type=int, default=config_defaults["replay_micro_batch_size"], help="Initial replay micro-batch size for candidate scoring")
    parser.add_argument("--replay_budget_window_size", type=int, default=config_defaults["replay_budget_window_size"], help="Window size used by replay micro-batch recovery")
    parser.add_argument("--replay_budget_recover_window_count", type=int, default=config_defaults["replay_budget_recover_window_count"], help="Number of clean replay windows required before growing replay micro-batches")
    parser.add_argument("--replay_candidate_cooldown", type=int, default=config_defaults["replay_candidate_cooldown"], help="Cooldown length for replay complexes that repeatedly trigger OOM")
    parser.add_argument("--replay_max_candidates_per_complex", type=int, default=config_defaults["replay_max_candidates_per_complex"], help="Maximum replay candidates kept per complex before micro-batching")

    parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if not (args.data_root and str(args.data_root).strip()):
        parser.error("You must provide --data_root or set data_root in the config, e.g. data/processed/hiqbind")
    args.data_root = str(args.data_root).strip()
    if args.resume_ckpt is not None and not str(args.resume_ckpt).strip():
        args.resume_ckpt = None
    if args.resume_blind_pool_dir is not None and not str(args.resume_blind_pool_dir).strip():
        args.resume_blind_pool_dir = None
    args.index_file = os.path.join(args.data_root, "index.csv")

    try:
        parsed_topk = tuple(int(x.strip()) for x in args.test_topk.split(",") if x.strip())
        if not parsed_topk:
            raise ValueError("empty top-k list")
        args.test_topk_values = parsed_topk
    except Exception as e:
        raise ValueError(f"Invalid --test_topk='{args.test_topk}': {e}") from e

    # 动态计算 `pro_res_cont_count`：残基连续特征维度加上 `esm_dim`。
    args.pro_res_cont_count = len(PROTEIN_RESIDUE_CONT_SCHEMA) + args.esm_dim

    run_suffix = build_run_suffix(args.run_suffix)
    run_name = f"train_{run_suffix}"
    log_file, _ = configure_text_logging(
        category="train",
        file_stem="train",
        smoke=bool(args.smoke),
        run_suffix=run_suffix,
    )

    logging.info("Logging to %s", log_file)

    # 为当前运行创建独立输出目录，避免覆盖历史模型与报告文件。
    base_save_dir = args.save_dir
    args.save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Smoke log grouping: %s", args.smoke)
    logger.info("Run suffix: %s", run_suffix)
    logger.info("Run artifacts will be saved to %s", args.save_dir)
    if args.resume_ckpt is not None:
        logger.info("Resume checkpoint: %s", args.resume_ckpt)
    if args.resume_blind_pool_dir is not None:
        logger.info("Resume blind pool dir: %s", args.resume_blind_pool_dir)
    if args.stop_after_epoch is not None:
        logger.info("Stop after epoch: %d", args.stop_after_epoch)
    logger.info("Starting training with arguments: %s", args)

    try:
        train(
            data_root=args.data_root,
            index_file=args.index_file,
            save_dir=args.save_dir,
            esm_path=args.esm_path,
            epochs=args.epochs,

            lr=args.lr,
            weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,
            hidden_dim=args.hidden_dim,
            num_gnn_blocks=args.num_gnn_blocks,
            m_dim_scalar=args.m_dim_scalar,
            dropout_rate=args.dropout_rate,
            lig_atom_cont_count=args.lig_atom_cont_count,
            lig_mol_cont_count=args.lig_mol_cont_count,
            pro_atom_cont_count=args.pro_atom_cont_count,
            pro_res_cont_count=args.pro_res_cont_count,
            esm=args.esm,
            esm_model_name=args.esm_model_name,
            esm_dim=args.esm_dim,
            num_rbf=args.num_rbf,
            r_cutoff=args.r_cutoff,
            force_cutoff=args.force_cutoff,
            frame_refine_threshold=args.frame_refine_threshold,
            frame_refine_temperature=args.frame_refine_temperature,
            energy_guide_threshold=args.energy_guide_threshold,
            energy_guide_temperature=args.energy_guide_temperature,
            clash_threshold=args.clash_threshold,
            clash_push_threshold=args.clash_push_threshold,
            clash_push_force=args.clash_push_force,
            score_clamp_min=args.score_clamp_min,
            score_clamp_max=args.score_clamp_max,
            force_limit=args.force_limit,
            prediction_max_neighbors=args.max_neighbors,
            prediction_min_max_neighbors=args.min_max_neighbors,
            prediction_knn_fallback_k=args.knn_fallback_k,
            r_cutoff_intra=args.r_cutoff_intra,
            max_neighbors_intra=args.max_neighbors_intra,
            atom_neighbor_cap=args.atom_neighbor_cap,
            residue_neighbor_cap=args.residue_neighbor_cap,
            residue_radius_scale=args.residue_radius_scale,
            residue_radius_bias=args.residue_radius_bias,
            ligand_atom_fallback_k=args.ligand_atom_fallback_k,
            protein_atom_fallback_k=args.protein_atom_fallback_k,
            protein_residue_fallback_k=args.protein_residue_fallback_k,
            dynamic_inter_cutoff=args.dynamic_inter_cutoff,
            dynamic_inter_knn_k=args.dynamic_inter_knn_k,
            dynamic_inter_max_neighbors=args.dynamic_inter_max_neighbors,
            dynamic_residue_cutoff=args.dynamic_residue_cutoff,
            dynamic_residue_knn_k=args.dynamic_residue_knn_k,
            dynamic_residue_max_neighbors=args.dynamic_residue_max_neighbors,
            dynamic_residue_candidate_topk=args.dynamic_residue_candidate_topk,
            flow_sigma_min=args.flow_sigma_min,
            flow_spatial_sigma_min=args.flow_spatial_sigma_min,
            flow_spatial_sigma_max=args.flow_spatial_sigma_max,
            flow_fd_dt=args.flow_fd_dt,
            flow_rotation_angle_min=args.flow_rotation_angle_min,
            flow_rotation_angle_max=args.flow_rotation_angle_max,
            flow_torsion_scale_min=args.flow_torsion_scale_min,
            flow_torsion_scale_max=args.flow_torsion_scale_max,
            loss_characteristic_scale=args.loss_characteristic_scale,
            loss_weight_translation=args.loss_weight_translation,
            loss_weight_rotation=args.loss_weight_rotation,
            loss_weight_torsion=args.loss_weight_torsion,
            loss_weight_energy=args.loss_weight_energy,
            loss_weight_clash=args.loss_weight_clash,
            loss_weight_pose_rank=args.loss_weight_pose_rank,
            loss_coarse_translation=args.loss_coarse_translation,
            loss_coarse_rotation=args.loss_coarse_rotation,
            loss_coarse_torsion=args.loss_coarse_torsion,
            loss_coarse_energy=args.loss_coarse_energy,
            loss_coarse_clash=args.loss_coarse_clash,
            loss_coarse_pose_rank=args.loss_coarse_pose_rank,
            loss_transition_translation=args.loss_transition_translation,
            loss_transition_rotation=args.loss_transition_rotation,
            loss_transition_torsion=args.loss_transition_torsion,
            loss_transition_energy=args.loss_transition_energy,
            loss_transition_clash=args.loss_transition_clash,
            loss_transition_pose_rank=args.loss_transition_pose_rank,
            loss_refine_translation=args.loss_refine_translation,
            loss_refine_rotation=args.loss_refine_rotation,
            loss_refine_torsion=args.loss_refine_torsion,
            loss_refine_energy=args.loss_refine_energy,
            loss_refine_clash=args.loss_refine_clash,
            loss_refine_pose_rank=args.loss_refine_pose_rank,
            loss_refine_start=args.loss_refine_start,
            loss_pose_gate_epoch_start=args.loss_pose_gate_epoch_start,
            loss_pose_gate_epoch_end=args.loss_pose_gate_epoch_end,
            loss_pose_gate_tau_start=args.loss_pose_gate_tau_start,
            loss_pose_gate_tau_end=args.loss_pose_gate_tau_end,
            loss_pose_gate_temperature=args.loss_pose_gate_temperature,
            device=args.device,
            crop_radius=args.crop_radius,
            warmup_epochs=args.warmup_epochs,
            val_subset_ratio=args.val_subset_ratio,
            val_full_every=args.val_full_every,
            val_full_last_epochs=args.val_full_last_epochs,
            min_checkpoint_selection_coverage=args.min_checkpoint_selection_coverage,
            max_val_non_oom_failures=args.max_val_non_oom_failures,
            max_val_oom_failures=args.max_val_oom_failures,
            final_topn_min_coverage=args.final_topn_min_coverage,
            ode_method=args.ode_method,
            accumulation_steps=args.accumulation_steps,
            train_cost_budget=args.train_cost_budget,
            val_cost_budget=args.val_cost_budget,
            blind_pool_cost_budget=args.blind_pool_cost_budget,
            final_topn_cost_budget=args.final_topn_cost_budget,
            eval_cost_guard_headroom=args.eval_cost_guard_headroom,
            ema_decay=args.ema_decay,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            max_oom_retry_splits=args.max_oom_retry_splits,
            split_train_frac=args.split_train_frac,
            split_val_frac=args.split_val_frac,
            split_test_frac=args.split_test_frac,
            split_seed=args.split_seed,
            split_cache_file=args.split_cache_file,
            force_resplit=args.force_resplit,
            ablation_mode=args.ablation_mode,
            run_test_after_training=args.run_test_after_training,
            test_topk_values=args.test_topk_values,
            enable_train_budget_callback=args.enable_train_budget_callback,
            oom_reduce_threshold=args.oom_reduce_threshold,
            oom_reduce_factor=args.oom_reduce_factor,
            min_train_cost_budget=args.min_train_cost_budget,
            enable_val_budget_callback=args.enable_val_budget_callback,
            val_oom_reduce_threshold=args.val_oom_reduce_threshold,
            val_oom_reduce_factor=args.val_oom_reduce_factor,
            min_val_cost_budget=args.min_val_cost_budget,
            train_budget_window_size=args.train_budget_window_size,
            train_budget_recover_window_count=args.train_budget_recover_window_count,
            train_budget_recover_step=args.train_budget_recover_step,
            train_offender_cooldown=args.train_offender_cooldown,
            val_budget_window_size=args.val_budget_window_size,
            val_budget_recover_window_count=args.val_budget_recover_window_count,
            val_budget_recover_step=args.val_budget_recover_step,
            val_offender_cooldown=args.val_offender_cooldown,
            center_proposal_weight=args.center_proposal_weight,
            center_positive_radius=args.center_positive_radius,
            center_guidance_learned_start=args.center_guidance_learned_start,
            center_proposal_topk=args.center_proposal_topk,
            center_refine_topk=args.center_refine_topk,
            center_nms_radius=args.center_nms_radius,
            stage1_pose_samples=args.stage1_pose_samples,
            stage2_pose_samples=args.stage2_pose_samples,
            crop_candidate_topk=args.crop_candidate_topk,
            crop_proposal_start=args.crop_proposal_start,
            crop_near_miss_start=args.crop_near_miss_start,
            crop_hard_negative_start=args.crop_hard_negative_start,
            crop_min_residues=args.crop_min_residues,
            crop_atom_margin=args.crop_atom_margin,
            disable_jitter_crop=args.disable_jitter_crop,
            disable_hard_negative_crop=args.disable_hard_negative_crop,
            pose_ranking_pair_weight=args.pose_ranking_pair_weight,
            pose_ranking_margin=args.pose_ranking_margin,
            ranking_same_center_start=args.ranking_same_center_start,
            ranking_wrong_center_start=args.ranking_wrong_center_start,
            pose_bootstrap_weight=args.pose_bootstrap_weight,
            pose_bootstrap_start=args.pose_bootstrap_start,
            pose_bootstrap_frequency=args.pose_bootstrap_frequency,
            pose_bootstrap_ode_steps=args.pose_bootstrap_ode_steps,
            val_ode_steps=args.val_ode_steps,
            checkpoint_selection_mode=args.checkpoint_selection_mode,
            blind_pool_refresh_every=args.blind_pool_refresh_every,
            blind_pool_start_epoch=args.blind_pool_start_epoch,
            blind_pool_refresh_on_best_update=args.blind_pool_refresh_on_best_update,
            blind_pool_max_complexes=args.blind_pool_max_complexes,
            blind_pool_cache_bce_weight=args.blind_pool_cache_bce_weight,
            blind_pool_cache_rank_weight=args.blind_pool_cache_rank_weight,
            blind_pool_pairs_per_complex=args.blind_pool_pairs_per_complex,
            replay_start_ratio=args.replay_start_ratio,
            same_center_micro_batch_size=args.same_center_micro_batch_size,
            same_center_budget_window_size=args.same_center_budget_window_size,
            same_center_budget_recover_window_count=args.same_center_budget_recover_window_count,
            same_center_budget_recover_step=args.same_center_budget_recover_step,
            same_center_offender_cooldown=args.same_center_offender_cooldown,
            ranking_budget_window_size=args.ranking_budget_window_size,
            ranking_budget_recover_window_count=args.ranking_budget_recover_window_count,
            ranking_offender_cooldown=args.ranking_offender_cooldown,
            ranking_wrong_center_cap=args.ranking_wrong_center_cap,
            replay_micro_batch_size=args.replay_micro_batch_size,
            replay_budget_window_size=args.replay_budget_window_size,
            replay_budget_recover_window_count=args.replay_budget_recover_window_count,
            replay_candidate_cooldown=args.replay_candidate_cooldown,
            replay_max_candidates_per_complex=args.replay_max_candidates_per_complex,
            geometry_min_atom_distance=args.geometry_min_atom_distance,
            resume_ckpt=args.resume_ckpt,
            resume_blind_pool_dir=args.resume_blind_pool_dir,
            stop_after_epoch=args.stop_after_epoch,
            run_name=run_name,
            run_log_file=log_file,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
