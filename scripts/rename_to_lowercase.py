"""
文件名与内容标准化脚本

将目标文件夹中的文件名统一转换为小写，且针对 .sdf 文件，
将其内容第一行的 Concatenated ID 也转换为小写。
"""

import os
import argparse
from pathlib import Path
import logging
from datetime import datetime

def setup_logger(log_root):
    """配置日志系统，将日志保存在 logs/data_processing 下"""
    log_dir = log_root / "data_processing"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rename_lowercase_{timestamp}.log"
    
    logger = logging.getLogger("rename_lowercase")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def process_file(file_path: Path, logger):
    current_path = file_path
    
    # 1. 重命名文件名 (转小写)
    try:
        if file_path.name != file_path.name.lower():
            new_name = file_path.name.lower()
            new_path = file_path.with_name(new_name)
            
            if new_path.exists():
                logger.warning(f"[SKIP RENAME] Target file already exists: {new_path} (Source: {file_path.name})")
            else:
                file_path.rename(new_path)
                logger.info(f"[RENAME] {file_path.name} -> {new_name}")
                current_path = new_path
    except Exception as e:
        logger.error(f"[ERROR RENAME] Failed to rename {file_path}: {e}")
        # 如果重命名失败，就不继续处理文件内容了，防止混淆
        return

    # 2. 如果是 SDF 文件，修改第一行内容 (转小写)
    if current_path.suffix.lower() == '.sdf':
        try:
            lines = []
            # 读取文件
            with open(current_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            if lines:
                original_header = lines[0].rstrip('\n')
                new_header = original_header.lower()
                
                # 只有当确实需要修改时才写入
                if original_header != new_header:
                    # 保持原有的换行符格式，如果有的话
                    # 这里简单处理：替换原来的字符串并加回换行符（如果原来有）
                    
                    # 检查是否有换行符
                    original_endswith_newline = lines[0].endswith('\n')
                    
                    lines[0] = new_header + ('\n' if original_endswith_newline else '')
                    
                    with open(current_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    logger.info(f"[UPDATE SDF] Updated content header in {current_path.name}: '{original_header}' -> '{new_header}'")
        except Exception as e:
            logger.error(f"[ERROR SDF] Failed to read/write SDF content for {current_path}: {e}")

def main():
    # 获取脚本所在目录的上一级目录作为项目根目录 (假设脚本在 scripts/ 下)
    # 项目结构: root/scripts/rename_lowercase.py
    project_root = Path(__file__).resolve().parent.parent
    
    # 默认路径
    default_ligand_dir = project_root / "data/raw/pdbbind/ligand"
    default_protein_dir = project_root / "data/raw/pdbbind/protein"
    default_log_root = project_root / "logs"

    parser = argparse.ArgumentParser(description="Rename files to lowercase and lowercase SDF header.")
    parser.add_argument("folders", nargs='*', type=str, help="Folders to process")
    
    args = parser.parse_args()
    
    # 确定要处理的文件夹
    if args.folders:
        targets = [Path(p) for p in args.folders]
    else:
        targets = [default_ligand_dir, default_protein_dir]
        print("No folders provided. Using default target folders:")
        for t in targets:
            print(f" - {t}")
        print("-" * 30)

    # 初始化日志
    logger, log_path = setup_logger(default_log_root)
    logger.info(f"Starting processing. Log saved to: {log_path}")

    total_files = 0
    
    for folder_path in targets:
        if not folder_path.exists():
            logger.error(f"[ERROR DIR] Directory not found: {folder_path}")
            continue
            
        logger.info(f"Processing directory: {folder_path}")
        
        # 遍历目录
        files = list(folder_path.iterdir())
        for file_path in files:
            if file_path.is_file():
                process_file(file_path, logger)
                total_files += 1

    logger.info("-" * 30)
    logger.info(f"Processing complete. Scanned {total_files} files.")

if __name__ == "__main__":
    main()
