"""
亲和力提取脚本

读取包含蛋白-配体信息的 CSV 文件，提取 Concatenated ID 和 Log Binding Affinity 两列，
并进行格式标准化（ID 转小写，数值保留指定小数位）。
"""

import csv
import argparse
import logging

from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def extract_affinity(input_file, output_file, precision=3):
    """
    提取 CSV 文件中的 Concatenated ID 和 Log Binding Affinity 列。
    将 Concatenated ID 转换为小写，结合能保留指定小数位，并保存到新文件。
    """
    
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        logger.error(f"Error: Input file '{input_file}' not found.")
        return

    logger.info(f"Processing {input_path} ...")

    try:
        with open(input_path, mode='r', encoding='utf-8-sig', newline='') as infile, \
             open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
            
            # 使用 DictReader 读取
            reader = csv.DictReader(infile)
            
            # 自动去除可能的BOM并normalize列名（可选优化，这里假设列名准确但可能有BOM，已用utf-8-sig解决）
            if not reader.fieldnames:
                 logger.error("Error: Empty file or no header found.")
                 return
            
            fieldnames = ['Concatenated ID', 'Log Binding Affinity']
            
            # 检查列是否存在
            if not set(fieldnames).issubset(set(reader.fieldnames)):
                 missing = set(fieldnames) - set(reader.fieldnames)
                 logger.error(f"Error: Missing columns in input file: {missing}")
                 # 打印实际列名以供调试
                 logger.error(f"Found columns: {reader.fieldnames}")
                 return

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            count = 0
            for row in reader:
                # 提取并处理数据
                raw_id = row.get('Concatenated ID', '')
                if not raw_id:
                    continue
                    
                concatenated_id = raw_id.lower().strip()
                affinity_str = row.get('Log Binding Affinity', '').strip()
                
                # 尝试格式化 affinity
                try:
                    val = float(affinity_str)
                    # 使用 f-string 格式化，保留指定位数
                    affinity_processed = f"{val:.{precision}f}"
                except ValueError:
                    # 如果转换失败（例如空值或非数字），保留原始值
                    affinity_processed = affinity_str

                writer.writerow({
                    'Concatenated ID': concatenated_id,
                    'Log Binding Affinity': affinity_processed
                })
                count += 1
                
        logger.info(f"Successfully processed {count} records.")
        logger.info(f"Saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ID and Affinity from CSV.")
    
    # 动态默认路径
    default_base = PROJECT_ROOT / "data/raw/pdbbind"
    default_input = default_base / "hiqbind_info.csv"
    default_output = default_base / "hiqbind_labels.csv"
    
    parser.add_argument("--input", "-i", type=str, default=str(default_input), help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default=str(default_output), help="Output CSV file path")
    parser.add_argument("--precision", "-p", type=int, default=6, help="Decimal precision for affinity (default: 6)")
    
    args = parser.parse_args()
    
    input_full_path = Path(args.input)
    output_full_path = Path(args.output)

    logger.info(f"Input: {input_full_path}")
    logger.info(f"Output: {output_full_path}")
    logger.info(f"Precision: {args.precision} decimal places")
    logger.info("-" * 30)

    extract_affinity(input_full_path, output_full_path, precision=args.precision)
