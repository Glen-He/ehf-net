"""
训练入口脚本

该脚本用于解析命令行参数，配置日志记录，并启动 EHFNet 模型的训练过程。
"""

import argparse
import os
import sys
import logging
import tomllib

from datetime import datetime
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

# [修复] 必须在 import torch 之前设置 CUDA allocator 配置，否则可能不生效
# expandable_segments:True -> 允许分配器动态扩展显存段，缓解碎片化
# max_split_size_mb:128 -> 避免大块显存被切得太碎
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from ehfnet.training.trainer import train
from ehfnet.encoders.feature_specs import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
import torch

# [新增] 全局开启 TF32 (提速神器)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _flatten_config(config: dict, *, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _config_to_arg_defaults(config: dict) -> dict[str, object]:
    flat = _flatten_config(config)
    defaults: dict[str, object] = {}
    for key, value in flat.items():
        defaults[key.split(".")[-1]] = value
    return defaults


def load_train_config(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return _config_to_arg_defaults(raw)


def load_model_config(model_config_path: str | None, project_root: Path) -> dict[str, object]:
    """加载 model.toml，返回扁平化的参数字典。"""
    if not model_config_path:
        return {}
    path = Path(model_config_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return _config_to_arg_defaults(raw)


def _resolve_auto_cutoffs(
    config: dict[str, object],
    data_root: str,
    project_root: Path,
) -> None:
    """将 config 中值为 'auto' 的 cutoff 解析为 dataset_profile.json 中的建议值。"""
    profile_path = Path(data_root) / "dataset_profile.json"
    if not profile_path.exists():
        return
    try:
        with open(profile_path, encoding="utf-8") as f:
            import json
            profile = json.load(f)
    except Exception:
        return
    suggested = profile.get("suggested_cutoffs", {})
    # 映射 config 键到 profile 键
    key_map = {
        "dynamic_inter_cutoff": "ligand_atom-protein_atom",
        "force_cutoff": "ligand_atom-protein_atom",
    }
    for cfg_key, profile_key in key_map.items():
        if cfg_key in config and config[cfg_key] == "auto":
            val = suggested.get(profile_key)
            if val is not None:
                config[cfg_key] = float(val)


def main():
    default_config_path = PROJECT_ROOT / "configs" / "train.toml"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path), help="Path to TOML config file")
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = load_train_config(pre_args.config)
    model_config_path = config_defaults.get("model_config")
    model_defaults = load_model_config(model_config_path, PROJECT_ROOT)
    config_defaults.update(model_defaults)
    # 解析 "auto" cutoff：从 dataset_profile.json 读取建议值
    data_root_for_auto = config_defaults.get("data_root") or None
    if data_root_for_auto:
        _resolve_auto_cutoffs(model_defaults, str(data_root_for_auto), PROJECT_ROOT)
    config_defaults.update(model_defaults)

    parser = argparse.ArgumentParser(
        description="Train EHFNet for molecular docking prediction",
        parents=[pre_parser],
    )
    
    # 数据相关参数：只接受一个文件夹，该文件夹内必须包含 index.csv（不允许自定义 index 路径）
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录，须含 index.csv，如 data/processed/hiqbind（必填）")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--esm_path", type=str, default=None, help="Path to precomputed ESM embeddings (optional)")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    
    # 模型相关参数
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--num_gnn_blocks", type=int, default=4, help="Number of GNN blocks")
    
    # 特征相关参数（通常固定，但为灵活性暴露）
    parser.add_argument("--lig_atom_cont_count", type=int, default=len(LIGAND_ATOM_CONT_SCHEMA), help="Ligand atom continuous feature count")
    parser.add_argument("--lig_mol_cont_count", type=int, default=len(LIGAND_MOLECULE_CONT_SCHEMA), help="Ligand molecule continuous feature count")
    parser.add_argument("--pro_atom_cont_count", type=int, default=len(PROTEIN_ATOM_CONT_SCHEMA), help="Protein atom continuous feature count")
    parser.add_argument("--esm_dim", type=int, default=960, help="ESM embedding dimension (default: 960 for ESMC-300M)")
    parser.add_argument("--num_rbf", type=int, default=50, help="RBF basis count (from model.toml)")
    parser.add_argument("--r_cutoff", type=float, default=10.0, help="Distance cutoff in Å (from model.toml)")
    parser.add_argument("--force_cutoff", type=float, default=6.0, help="Force branch local radius in Å (from model.toml)")
    parser.add_argument("--dynamic_inter_cutoff", type=float, default=10.0, help="Dynamic inter-atom edge radius (from model.toml)")
    parser.add_argument("--dynamic_inter_knn_k", type=int, default=8, help="kNN fallback for inter-atom edges (from model.toml)")
    parser.add_argument("--dynamic_residue_cutoff", type=float, default=14.0, help="Dynamic ligand-residue edge radius (from model.toml)")
    parser.add_argument("--dynamic_residue_knn_k", type=int, default=6, help="kNN fallback for ligand-residue edges (from model.toml)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--crop_radius", type=float, default=10.0, help="Runtime local crop radius in angstroms (default: 10.0)")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs for spatial curriculum learning (default: 20)")
    parser.add_argument("--rmsd_ratio", type=float, default=0.2, help="Ratio of validation set to compute RMSD (0.0-1.0)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_nodes_per_batch", type=int, default=20000, help="Max nodes per batch for DynamicBatchSampler.")
    parser.add_argument("--val_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for validation loader (default: min(train_budget, 6000))")
    parser.add_argument("--test_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for final test loader (default: same as val budget)")
    parser.add_argument("--topn_max_nodes_per_batch", type=int, default=None, help="Max nodes per batch for Top-N evaluation loader (default: same as test budget)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate (default: 0.999; use 0.99 for quick smoke tests)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True, help="Enable DataLoader pin_memory")
    parser.add_argument("--no_dataloader_pin_memory", dest="dataloader_pin_memory", action="store_false", help="Disable DataLoader pin_memory")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=True, help="Enable DataLoader persistent_workers")
    parser.add_argument("--no_dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store_false", help="Disable DataLoader persistent_workers")
    parser.add_argument("--enable_oom_adaptive_batch", action="store_true", default=True, help="Auto-reduce max_nodes_per_batch when frequent CUDA OOM occurs")
    parser.add_argument("--disable_oom_adaptive_batch", dest="enable_oom_adaptive_batch", action="store_false", help="Disable adaptive OOM batch protection")
    parser.add_argument("--oom_reduce_threshold", type=int, default=3, help="Reduce batch node budget when OOM batches in an epoch reach this threshold")
    parser.add_argument("--oom_reduce_factor", type=float, default=0.85, help="Factor to shrink max_nodes_per_batch after OOM threshold (0-1)")
    parser.add_argument("--min_max_nodes_per_batch", type=int, default=12000, help="Lower bound for adaptive max_nodes_per_batch")
    parser.add_argument("--enable_val_oom_adaptive_batch", action="store_true", default=True, help="Auto-reduce validation node budget when validation OOM is frequent")
    parser.add_argument("--disable_val_oom_adaptive_batch", dest="enable_val_oom_adaptive_batch", action="store_false", help="Disable validation OOM adaptive protection")
    parser.add_argument("--val_oom_reduce_threshold", type=int, default=3, help="Reduce validation node budget when validation OOM batches reach this threshold")
    parser.add_argument("--val_oom_reduce_factor", type=float, default=0.85, help="Factor to shrink validation max_nodes_per_batch after OOM threshold (0-1)")
    parser.add_argument("--min_val_max_nodes_per_batch", type=int, default=None, help="Lower bound for adaptive validation max_nodes_per_batch (default: same as min_max_nodes_per_batch)")
    parser.add_argument("--oom_recover_epochs", type=int, default=3, help="Consecutive clean epochs before attempting batch budget recovery")
    parser.add_argument("--oom_recover_factor", type=float, default=1.1, help="Factor to grow max_nodes_per_batch during recovery (>1)")
    parser.add_argument("--split_train_frac", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--split_val_frac", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--split_test_frac", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for scaffold split")
    parser.add_argument("--split_cache_file", type=str, default=None, help="Path to persisted split JSON")
    parser.add_argument("--force_resplit", action="store_true", help="Force regenerate split JSON")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "inter_multiscale_off"],
        help="Ablation mode: none (full model), inter_multiscale_off (atom-atom only)",
    )
    parser.add_argument("--run_test_after_training", action="store_true", default=True, help="Run final test-set evaluation after training")
    parser.add_argument("--skip_test_after_training", dest="run_test_after_training", action="store_false", help="Skip final test-set evaluation")
    parser.add_argument("--test_topk", type=str, default="1,5,10", help="Comma-separated top-k values for final test evaluation")
    parser.add_argument("--test_pose_samples", type=int, default=10, help="Number of candidate poses per complex for Top-N evaluation")
    parser.add_argument("--center_proposal_weight", type=float, default=0.15, help="Loss weight for residue-level center proposal")
    parser.add_argument("--center_positive_radius", type=float, default=4.0, help="Positive radius in angstroms for residue center supervision")
    parser.add_argument("--center_proposal_topk", type=int, default=8, help="Number of diverse residue centers kept in stage-1 proposal")
    parser.add_argument("--center_refine_topk", type=int, default=3, help="Number of centers refined in stage-2 local docking")
    parser.add_argument("--center_nms_radius", type=float, default=6.0, help="Diversity radius in angstroms for center NMS")
    parser.add_argument("--stage1_pose_samples", type=int, default=2, help="Number of local docking samples per center in stage-1")
    parser.add_argument("--stage2_pose_samples", type=int, default=4, help="Number of local docking samples per center in stage-2")
    parser.add_argument("--crop_candidate_topk", type=int, default=8, help="Top-k proposal candidates used for weighted crop sampling in each curriculum bucket")
    parser.add_argument("--disable_jitter_crop", action="store_true", default=False, help="Disable jitter crop branch for ablation")
    parser.add_argument("--disable_hard_negative_crop", action="store_true", default=False, help="Disable hard-negative crop branch for ablation")
    parser.add_argument("--pose_ranking_pair_weight", type=float, default=0.2, help="Pairwise ranking loss weight for pose confidence")
    parser.add_argument("--pose_ranking_margin", type=float, default=0.5, help="Margin used by the pairwise ranking loss")
    parser.add_argument("--pose_bootstrap_weight", type=float, default=0.05, help="Bootstrap loss weight for model-generated poses")
    parser.add_argument("--pose_bootstrap_frequency", type=int, default=25, help="Run bootstrap scoring every N training batches (0 disables)")
    parser.add_argument("--pose_bootstrap_ode_steps", type=int, default=10, help="ODE steps used to generate bootstrap poses")
    parser.add_argument("--enable_fusion_calibration", action="store_true", default=True, help="Grid-search center/pose fusion weight on Val-Blind")
    parser.add_argument("--disable_fusion_calibration", dest="enable_fusion_calibration", action="store_false", help="Disable Val-Blind fusion calibration")
    parser.add_argument("--val_ode_steps", type=int, default=50, help="ODE integration steps for validation and test evaluation (default: 50)")
    parser.add_argument(
        "--checkpoint_selection_mode",
        type=str,
        default="composite",
        choices=[
            "composite",
            "reranked_top1_success_2a",
            "reranked_top5_success_2a",
            "reranked_top1_plus_oracle_top5",
        ],
        help="Primary blind metric used for best_selected_model checkpoint selection",
    )
    parser.add_argument("--fusion_search_center_weights", type=str, default="0,0.15,0.25,0.35,0.5,0.65", help="Comma-separated center weights for fusion ablation grid")
    parser.add_argument("--fusion_search_aff_weights", type=str, default="0", help="Comma-separated affinity weights for fusion ablation grid (default: '0' = disabled)")
    parser.add_argument("--fusion_search_clash_weights", type=str, default="0", help="Comma-separated clash weights for fusion ablation grid (default: '0' = disabled)")
    parser.add_argument("--blind_pool_refresh_every", type=int, default=5, help="Refresh blind candidate pool every N epochs")
    parser.add_argument("--blind_pool_start_epoch", type=int, default=10, help="Earliest epoch to start pool refresh")
    parser.add_argument("--blind_pool_max_complexes", type=int, default=500, help="Max complexes per pool refresh")
    parser.add_argument("--blind_pool_cache_bce_weight", type=float, default=0.5, help="Cache BCE loss weight")
    parser.add_argument("--blind_pool_cache_rank_weight", type=float, default=1.0, help="Cache pairwise ranking loss weight")
    parser.add_argument("--blind_pool_pairs_per_complex", type=int, default=4, help="Hard pairs sampled per complex from cache")

    parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if not (args.data_root and str(args.data_root).strip()):
        parser.error("必须指定 --data_root 或在 config 中设置 data_root，例如: data/processed/hiqbind")
    args.data_root = str(args.data_root).strip()
    args.index_file = os.path.join(args.data_root, "index.csv")

    try:
        parsed_topk = tuple(int(x.strip()) for x in args.test_topk.split(",") if x.strip())
        if not parsed_topk:
            raise ValueError("empty top-k list")
        args.test_topk_values = parsed_topk
    except Exception as e:
        raise ValueError(f"Invalid --test_topk='{args.test_topk}': {e}") from e

    try:
        args.parsed_fusion_center_weights = tuple(float(x.strip()) for x in args.fusion_search_center_weights.split(",") if x.strip())
        args.parsed_fusion_aff_weights = tuple(float(x.strip()) for x in args.fusion_search_aff_weights.split(",") if x.strip())
        args.parsed_fusion_clash_weights = tuple(float(x.strip()) for x in args.fusion_search_clash_weights.split(",") if x.strip())
    except Exception as e:
        raise ValueError(f"Invalid fusion search weights: {e}") from e
    
    # 动态计算 pro_res_cont_count: residue continuous schema + esm_dim
    args.pro_res_cont_count = len(PROTEIN_RESIDUE_CONT_SCHEMA) + args.esm_dim

    # 配置 logging
    # 将日志保存在 logs/train 目录下，并使用时间戳防止覆盖
    log_dir = os.path.join("logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{timestamp}"
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logging.info(f"Logging to {log_file}")

    # 为当前运行创建独立输出目录，避免覆盖历史 checkpoint/report
    base_save_dir = args.save_dir
    args.save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Run artifacts will be saved to {args.save_dir}")
    logger.info(f"Starting training with arguments: {args}")

    # 归一化统计在 trainer 中按 train split 计算并缓存。
    stats_file = os.path.join(args.data_root, "normalization_stats.pt")
    if os.path.exists(stats_file):
        logger.info(
            "Ignoring legacy global normalization stats at %s; trainer now computes train-split-only stats.",
            stats_file,
        )
    normalization_stats = None

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
            lig_atom_cont_count=args.lig_atom_cont_count,
            lig_mol_cont_count=args.lig_mol_cont_count,
            pro_atom_cont_count=args.pro_atom_cont_count,
            pro_res_cont_count=args.pro_res_cont_count,
            esm_dim=args.esm_dim,
            num_rbf=args.num_rbf,
            r_cutoff=args.r_cutoff,
            force_cutoff=args.force_cutoff,
            dynamic_inter_cutoff=args.dynamic_inter_cutoff,
            dynamic_inter_knn_k=args.dynamic_inter_knn_k,
            dynamic_residue_cutoff=args.dynamic_residue_cutoff,
            dynamic_residue_knn_k=args.dynamic_residue_knn_k,
            device=args.device,
            crop_radius=args.crop_radius,
            normalization_stats=normalization_stats,
            warmup_epochs=args.warmup_epochs,
            rmsd_check_ratio=args.rmsd_ratio,
            accumulation_steps=args.accumulation_steps,
            max_nodes_per_batch=args.max_nodes_per_batch,
            val_max_nodes_per_batch=args.val_max_nodes_per_batch,
            test_max_nodes_per_batch=args.test_max_nodes_per_batch,
            topn_max_nodes_per_batch=args.topn_max_nodes_per_batch,
            ema_decay=args.ema_decay,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            split_train_frac=args.split_train_frac,
            split_val_frac=args.split_val_frac,
            split_test_frac=args.split_test_frac,
            split_seed=args.split_seed,
            split_cache_file=args.split_cache_file,
            force_resplit=args.force_resplit,
            ablation_mode=args.ablation_mode,
            run_test_after_training=args.run_test_after_training,
            test_topk_values=args.test_topk_values,
            test_pose_samples=args.test_pose_samples,
            enable_oom_adaptive_batch=args.enable_oom_adaptive_batch,
            oom_reduce_threshold=args.oom_reduce_threshold,
            oom_reduce_factor=args.oom_reduce_factor,
            min_max_nodes_per_batch=args.min_max_nodes_per_batch,
            enable_val_oom_adaptive_batch=args.enable_val_oom_adaptive_batch,
            val_oom_reduce_threshold=args.val_oom_reduce_threshold,
            val_oom_reduce_factor=args.val_oom_reduce_factor,
            min_val_max_nodes_per_batch=args.min_val_max_nodes_per_batch,
            oom_recover_epochs=args.oom_recover_epochs,
            oom_recover_factor=args.oom_recover_factor,
            center_proposal_weight=args.center_proposal_weight,
            center_positive_radius=args.center_positive_radius,
            center_proposal_topk=args.center_proposal_topk,
            center_refine_topk=args.center_refine_topk,
            center_nms_radius=args.center_nms_radius,
            stage1_pose_samples=args.stage1_pose_samples,
            stage2_pose_samples=args.stage2_pose_samples,
            crop_candidate_topk=args.crop_candidate_topk,
            disable_jitter_crop=args.disable_jitter_crop,
            disable_hard_negative_crop=args.disable_hard_negative_crop,
            pose_ranking_pair_weight=args.pose_ranking_pair_weight,
            pose_ranking_margin=args.pose_ranking_margin,
            pose_bootstrap_weight=args.pose_bootstrap_weight,
            pose_bootstrap_frequency=args.pose_bootstrap_frequency,
            pose_bootstrap_ode_steps=args.pose_bootstrap_ode_steps,
            enable_fusion_calibration=args.enable_fusion_calibration,
            val_ode_steps=args.val_ode_steps,
            checkpoint_selection_mode=args.checkpoint_selection_mode,
            fusion_search_center_weights=args.parsed_fusion_center_weights,
            fusion_search_aff_weights=args.parsed_fusion_aff_weights,
            fusion_search_clash_weights=args.parsed_fusion_clash_weights,
            blind_pool_refresh_every=args.blind_pool_refresh_every,
            blind_pool_start_epoch=args.blind_pool_start_epoch,
            blind_pool_max_complexes=args.blind_pool_max_complexes,
            blind_pool_cache_bce_weight=args.blind_pool_cache_bce_weight,
            blind_pool_cache_rank_weight=args.blind_pool_cache_rank_weight,
            blind_pool_pairs_per_complex=args.blind_pool_pairs_per_complex,
            run_name=run_name,
            run_log_file=log_file,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
