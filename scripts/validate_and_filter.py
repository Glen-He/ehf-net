"""
数据集验证与清洗脚本

检查 ligand 和 protein 文件夹的一致性（配对存在），根据文件存在情况过滤 CSV 索引文件，
并清理多余的无配对文件。
"""

import argparse
import sys
import csv
import logging
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def check_folder_consistency(ligand_dir, protein_dir):
    """
    检查 ligand 和 protein 文件夹的一致性。
    规则: 每一个 xxx_ligand.sdf 必须对应一个 xxx_protein.pdb
    """
    logger.info(f"Checking consistency between:\n  Ligand: {ligand_dir}\n  Protein: {protein_dir}")
    
    if not ligand_dir.exists() or not protein_dir.exists():
        logger.error("Error: One or both directories do not exist.")
        return False

    #以此为基准：获取所有 ligand 前缀
    # 显式排序保证确定性
    ligand_files = sorted(list(ligand_dir.glob("*_ligand.sdf")))
    protein_files = sorted(list(protein_dir.glob("*_protein.pdb")))
    
    # 提取前缀集合
    # 假设文件名格式严格为 prefix_ligand.sdf 和 prefix_protein.pdb
    # 转换为小写以前缀匹配，因为之前脚本可能已经做了小写转换
    ligand_prefixes = {f.name.lower().replace('_ligand.sdf', '') for f in ligand_files}
    protein_prefixes = {f.name.lower().replace('_protein.pdb', '') for f in protein_files}
    
    # 检查 Ligand 有而 Protein 没有的 (Orphans)
    missing_proteins = ligand_prefixes - protein_prefixes
    
    if missing_proteins:
        logger.warning(f"\n[WARN] Found {len(missing_proteins)} ligands without corresponding protein files.")
        # 这里仅警告，因为稍后 cleanup 会统一处理
        return True 
    else:
        logger.info("\n[PASS] All ligand files have corresponding protein files.")
        return True

def filter_csv(input_csv, output_csv, ligand_dir, protein_dir):
    """
    检查 CSV 中的 Concatenated ID 是否有对应的文件存在。
    存在的写入新 CSV，不存在的跳过。
    返回一个包含所有有效 ID 的集合。
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    valid_ids = set()
    
    if not input_path.exists():
        logger.error(f"Error: CSV file {input_path} not found.")
        return set()

    logger.info(f"\nFiltering CSV based on file existence...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    valid_count = 0
    skipped_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8-sig', newline='') as infile, \
             open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                logger.error("Error: CSV file is empty.")
                return set()
                
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                concat_id = row.get('Concatenated ID', '').strip()
                if not concat_id:
                    continue
                
                # 构造预期的文件名
                # 假设之前已经做过转小写处理，这里确保 ID 也是小写
                # 文件名格式: [id]_ligand.sdf 和 [id]_protein.pdb
                
                # 注意：这里需要处理文件路径检查
                # 我们假设文件名是全小写的（因为 extract_affinity 已经转了 ID 为小写，rename_to_lowercase 也转了文件名）
                lower_id = concat_id.lower()
                
                ligand_name = f"{lower_id}_ligand.sdf"
                protein_name = f"{lower_id}_protein.pdb"
                
                ligand_file = ligand_dir / ligand_name
                protein_file = protein_dir / protein_name
                
                if ligand_file.exists() and protein_file.exists():
                    writer.writerow(row)
                    valid_ids.add(lower_id)
                    valid_count += 1
                else:
                    skipped_count += 1
                    # 可选：打印一些丢失的示例
                    if skipped_count <= 5:
                        logger.debug(f"Skipping {concat_id}: One or both files missing.")
        
        logger.info("-" * 30)
        logger.info(f"Filter Complete.")
        logger.info(f"Valid pairs retained: {valid_count}")
        logger.info(f"Skipped rows: {skipped_count}")
        logger.info(f"New CSV saved to: {output_path}")
        return valid_ids

    except Exception as e:
        logger.error(f"An error occurred during CSV filtering: {e}")
        return set()

def cleanup_extra_files(valid_ids, ligand_dir, protein_dir, dry_run=False):
    """
    删除不在 valid_ids 列表中的所有 sdf 和 pdb 文件。
    确保文件夹内容与过滤后的 CSV 严格一致。
    """
    logger.info(f"\nCleaning up extra files (Dry Run: {dry_run})...")
    
    deleted_count = 0
    
    # 扫描配体文件夹 (显式排序)
    for file_path in sorted(list(ligand_dir.glob("*_ligand.sdf"))):
        prefix = file_path.name.lower().replace('_ligand.sdf', '')
        if prefix not in valid_ids:
            if not dry_run:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
            else:
                 logger.info(f"  [Would Delete] {file_path.name}")
                 deleted_count += 1
    
    # 扫描蛋白文件夹 (显式排序)
    for file_path in sorted(list(protein_dir.glob("*_protein.pdb"))):
        prefix = file_path.name.lower().replace('_protein.pdb', '')
        if prefix not in valid_ids:
            if not dry_run:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
            else:
                 logger.info(f"  [Would Delete] {file_path.name}")
                 deleted_count += 1
                 
    action = "Deleted" if not dry_run else "Found"
    logger.info(f"Cleanup Complete. {action} {deleted_count} extra files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset file pairs, filter CSV, and cleanup extra files.")
    
    # 动态默认路径
    default_base = PROJECT_ROOT / "data/raw/pdbbind"
    default_ligand = default_base / "ligand"
    default_protein = default_base / "protein"
    default_input = default_base / "hiqbind_labels.csv"
    default_output = default_base / "hiqbind_filtered.csv"
    
    parser.add_argument("--ligand_dir", type=str, default=str(default_ligand), help="Path to ligand directory")
    parser.add_argument("--protein_dir", type=str, default=str(default_protein), help="Path to protein directory")
    parser.add_argument("--input_csv", type=str, default=str(default_input), help="Input CSV file")
    parser.add_argument("--output_csv", type=str, default=str(default_output), help="Output filtered CSV file")
    
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be deleted without actually deleting")

    args = parser.parse_args()
    
    ligand_path = Path(args.ligand_dir)
    protein_path = Path(args.protein_dir)
    
    # 1. 简单的文件夹存在性检查
    check_folder_consistency(ligand_path, protein_path)
    
    # 2. 过滤 CSV 并获取“白名单”
    valid_ids = filter_csv(args.input_csv, args.output_csv, ligand_path, protein_path)
    
    # 3. 如果成功获取了白名单，清理多余文件
    if valid_ids:
        cleanup_extra_files(valid_ids, ligand_path, protein_path, dry_run=args.dry_run)
    else:
        logger.warning("\n[SKIP] No valid IDs found or CSV processing failed. Skipping cleanup.")
