"""
清理训练缓存脚本

清理 data/processed/pdbbind 下的缓存文件，包括：
1. cleaned 文件夹中每个样本的 .npz 文件（ESM embeddings 缓存）
2. processed 文件夹（整个文件夹）
"""

import shutil
import argparse
import logging

from pathlib import Path
from tqdm import tqdm


# 获取项目根目录 (假设脚本在 scripts/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def clean_esm_cache(data_root: str, dry_run: bool = False) -> tuple[int, float]:
    """
    Clean ESM cache files (.npz) in the 'cleaned' directory.
    
    Args:
        data_root: Path to data/processed/pdbbind
        dry_run: If True, only print what would be deleted without actually deleting
        
    Returns:
        (count, size_in_mb)
    """

    root = Path(data_root)
    esm_cache_dir = root / "cleaned"
    
    if not esm_cache_dir.exists():
        logger.warning(f"Directory not found: {esm_cache_dir}. Skipping ESM cache cleaning.")
        return 0, 0.0

    npz_files = list(esm_cache_dir.rglob("*_esm.npz"))
    count = len(npz_files)
    
    if count == 0:
        return 0, 0.0
        
    size_bytes = sum(f.stat().st_size for f in npz_files)
    size_mb = size_bytes / (1024 * 1024)
    
    logger.info(f"Found {count} .npz files ({size_mb:.2f} MB)")
    
    if dry_run:
        return count, size_mb
        
    logger.info("Deleting .npz files...")
    
    for f in tqdm(npz_files, desc="Cleaning ESM cache"):
        try:
            f.unlink()
        except OSError as e:
            logger.error(f"Failed to delete {f}: {e}")
            
    return count, size_mb


def clean_processed_folder(data_root: str, dry_run: bool = False) -> tuple[bool, float]:
    """
    Clean the entire 'processed' folder.
    
    Args:
        data_root: Path to data/processed/pdbbind
        dry_run: If True, only print what would be deleted without actually deleting
        
    Returns:
        (success, size_in_mb)
    """

    root = Path(data_root)
    processed_dir = root / "processed"
    
    if not processed_dir.exists():
        return True, 0.0
        
    # 计算大小
    files = list(processed_dir.rglob("*"))
    size_bytes = sum(f.stat().st_size for f in files if f.is_file())
    size_mb = size_bytes / (1024 * 1024)
    
    logger.info(f"Found processed folder: {len(files)} files ({size_mb:.2f} MB)")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete: {processed_dir}")
        return False, size_mb
        
    logger.info(f"Deleting processed folder: {processed_dir}")
    
    try:
        shutil.rmtree(processed_dir)
        logger.info("✓ Processed folder deleted successfully")
        return True, size_mb
    except Exception as e:
        logger.error(f"Failed to delete {processed_dir}: {e}")
        return False, 0.0


def clean_normalization_stats(data_root: str, dry_run: bool = False) -> tuple[bool, int]:
    """
    Clean normalization stats file.

    Args:
        data_root: Path to data/processed/pdbbind
        dry_run: If True, only print what would be deleted without actually deleting

    Returns:
        (success, file size in bytes)
    """

    stats_file = Path(data_root) / "normalization_stats.pt"
    
    if not stats_file.exists():
        logger.info(f"Normalization stats not found: {stats_file}")
        return False, 0
    
    size = stats_file.stat().st_size
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete: {stats_file}")
        return False, size
        
    try:
        stats_file.unlink()
        logger.info(f"Deleted normalization stats: {stats_file}")
        return True, size
        
    except Exception as e:
        logger.error(f"Failed to delete {stats_file}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description="Clean training cache files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 动态默认路径
    default_data_root = PROJECT_ROOT / "data/processed/pdbbind"
    
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(default_data_root),
        help="Path to processed data root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--clean-esm",
        action="store_true",
        default=False,
        help="Enable cleaning of ESM cache files (.npz) (Default: False)",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        default=False,
        help="Skip cleaning processed folder",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        default=False,
        help="Skip cleaning normalization stats",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        logger.error(f"Error: Data root not found: {data_root}")
        return

    logger.info("=" * 60)
    logger.info("EHFNet Cache Cleaner")
    logger.info("=" * 60)
    logger.info(f"Target: {data_root.absolute()}")
    logger.info(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'DELETION'}")
    logger.info("=" * 60)

    total_freed_bytes = 0

    # Clean ESM cache
    if args.clean_esm:
        npz_count, npz_size_mb = clean_esm_cache(str(data_root), args.dry_run)
        total_freed_bytes += npz_size_mb * 1024 * 1024
        if not args.dry_run and npz_count > 0:
            logger.info(f"✓ Deleted {npz_count} .npz files ({npz_size_mb:.2f} MB)")
    else:
        logger.info("Skipping ESM cache cleaning (use --clean-esm to enable).")

    # Clean processed folder
    if not args.skip_processed:
        success, proc_size_mb = clean_processed_folder(str(data_root), args.dry_run)
        total_freed_bytes += proc_size_mb * 1024 * 1024
    
    # Clean normalization stats
    if not args.skip_stats:
        success, stats_size = clean_normalization_stats(str(data_root), args.dry_run)
        total_freed_bytes += stats_size

    total_freed_mb = total_freed_bytes / (1024 * 1024)
    
    logger.info("\n" + "=" * 60)
    if args.dry_run:
        logger.info(f"[DRY RUN] Would free approximately {total_freed_mb:.2f} MB")
        logger.info("\nRe-run without --dry-run to actually delete files.")
    else:
        logger.info(f"Total space freed: {total_freed_mb:.2f} MB")
        logger.info("Cache cleaning completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
