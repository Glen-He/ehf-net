"""
清理训练缓存脚本

清理 data/processed/pdbbind 下的缓存文件，包括：
1. cleaned 文件夹中每个样本的 .npz 文件（ESM embeddings 缓存）
2. processed 文件夹（整个文件夹）
"""

import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def clean_esm_cache(data_root: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Clean ESM embedding cache files (.npz) in cleaned folder.

    Args:
        data_root: Path to data/processed/pdbbind
        dry_run: If True, only print what would be deleted without actually deleting

    Returns:
        (number of files deleted, total size in MB)
    """

    cleaned_dir = Path(data_root) / "cleaned"

    if not cleaned_dir.exists():
        print(f"Cleaned directory not found: {cleaned_dir}")
        return 0, 0

    npz_files = list(cleaned_dir.rglob("*.npz"))

    if not npz_files:
        print("No .npz files found in cleaned directory.")
        return 0, 0

    total_size = sum(f.stat().st_size for f in npz_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"\nFound {len(npz_files)} .npz files ({total_size_mb:.2f} MB)")

    if dry_run:
        print("[DRY RUN] Would delete:")
        for f in npz_files[:5]:  # Show first 5 as examples
            print(f"  - {f.relative_to(data_root)}")
        if len(npz_files) > 5:
            print(f"  ... and {len(npz_files) - 5} more files")
        return 0, 0

    print("Deleting .npz files...")
    for npz_file in tqdm(npz_files, desc="Cleaning ESM cache"):
        try:
            npz_file.unlink()
        except Exception as e:
            print(f"Failed to delete {npz_file}: {e}")

    return len(npz_files), int(total_size_mb)


def clean_processed_folder(data_root: str, dry_run: bool = False) -> tuple[bool, int]:
    """
    Clean the entire processed folder.

    Args:
        data_root: Path to data/processed/pdbbind
        dry_run: If True, only print what would be deleted without actually deleting

    Returns:
        (success, folder size in MB)
    """

    processed_dir = Path(data_root) / "processed"

    if not processed_dir.exists():
        print(f"\nProcessed directory not found: {processed_dir}")
        return False, 0

    # Calculate folder size
    total_size = sum(
        f.stat().st_size for f in processed_dir.rglob("*") if f.is_file()
    )
    total_size_mb = total_size / (1024 * 1024)
    num_files = sum(1 for _ in processed_dir.rglob("*") if _.is_file())

    print(f"\nFound processed folder: {num_files} files ({total_size_mb:.2f} MB)")

    if dry_run:
        print(f"[DRY RUN] Would delete entire folder: {processed_dir}")
        return False, int(total_size_mb)

    try:
        print(f"Deleting processed folder: {processed_dir}")
        shutil.rmtree(processed_dir)
        print("✓ Processed folder deleted successfully")
        return True, int(total_size_mb)

    except Exception as e:
        print(f"Failed to delete processed folder: {e}")
        return False, 0


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
        print(f"\nNormalization stats not found: {stats_file}")
        return False, 0
    
    size = stats_file.stat().st_size
    
    if dry_run:
        print(f"[DRY RUN] Would delete: {stats_file}")
        return False, size
        
    try:
        stats_file.unlink()
        print(f"\nDeleted normalization stats: {stats_file}")
        return True, size
    except Exception as e:
        print(f"Failed to delete {stats_file}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description="Clean training cache files in data/processed/pdbbind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed/pdbbind",
        help="Path to processed data root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--skip-esm",
        action="store_true",
        default=False,
        help="Skip cleaning ESM cache files (.npz)",
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
        print(f"Error: Data root not found: {data_root}")
        return

    print("=" * 60)
    print("EHFNet Cache Cleaner")
    print("=" * 60)
    print(f"Target: {data_root.absolute()}")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'DELETION'}")
    print("=" * 60)

    total_freed_bytes = 0

    # Clean ESM cache
    if not args.skip_esm:
        npz_count, npz_size_mb = clean_esm_cache(str(data_root), args.dry_run)
        total_freed_bytes += npz_size_mb * 1024 * 1024
        if not args.dry_run and npz_count > 0:
            print(f"✓ Deleted {npz_count} .npz files ({npz_size_mb} MB)")

    # Clean processed folder
    if not args.skip_processed:
        success, proc_size_mb = clean_processed_folder(str(data_root), args.dry_run)
        total_freed_bytes += proc_size_mb * 1024 * 1024
    
    # Clean normalization stats
    if not args.skip_stats:
        success, stats_size = clean_normalization_stats(str(data_root), args.dry_run)
        total_freed_bytes += stats_size

    total_freed_mb = total_freed_bytes / (1024 * 1024)
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY RUN] Would free approximately {total_freed_mb:.2f} MB")
        print("\nRe-run without --dry-run to actually delete files.")
    else:
        print(f"Total space freed: {total_freed_mb:.2f} MB")
        print("Cache cleaning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
