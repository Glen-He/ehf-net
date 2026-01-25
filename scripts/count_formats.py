"""
统计文件格式数量脚本

该脚本用于递归统计指定文件夹下 .sdf 和 .pdb 文件的数量。
"""

import argparse
from pathlib import Path


def count_files(folder_path):
    """
    统计指定文件夹内（包含子文件夹）的 .sdf 和 .pdb 文件数量。
    
    Args:
        folder_path (str): 目标文件夹路径
    """
    path_obj = Path(folder_path)
    
    if not path_obj.exists():
        print(f"Error: Path '{folder_path}' does not exist.")
        return
    
    if not path_obj.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return

    # 使用 rglob 进行递归搜索 (Case insensitive 通常较好，但 glob 在 Linux 区分大小写)
    # 这里假设后缀是准确的小写，或者我们可以做更复杂的检查
    # 为简单起见，这里统计 .sdf 和 .pdb (均视为小写)
    # 如果需要忽略大小写，可以手动 iterdir 判断 .suffix.lower()
    
    print(f"Scanning directory: {path_obj.resolve()} ...")
    
    sdf_count = 0
    pdb_count = 0
    
    # 遍历所有文件
    for file_path in path_obj.rglob("*"):

        if file_path.is_file():
            suffix = file_path.suffix.lower()

            if suffix == ".sdf":
                sdf_count += 1

            elif suffix == ".pdb":
                pdb_count += 1
    
    print("-" * 30)
    print(f"SDF files: {sdf_count}")
    print(f"PDB files: {pdb_count}")
    print("-" * 30)
    print(f"Total found: {sdf_count + pdb_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count .sdf and .pdb files recursively in a folder.")
    parser.add_argument("folders", type=str, nargs='*', help="Path(s) to the target folder(s)")
    
    args = parser.parse_args()
    
    target_folders = args.folders
    
    # 如果没有传入参数，则使用默认的路径
    if not target_folders:
        # 使用 absolute path
        target_folders = [
            "/pavo/glen/Code/EHFNet/data/raw/pdbbind/ligand",
            "/pavo/glen/Code/EHFNet/data/raw/pdbbind/protein"
        ]
        print("No folders provided. Using default paths:")

        for p in target_folders:
            print(f" - {p}")

        print("=" * 30)
    
    for folder in target_folders:
        count_files(folder)

