"""
亲和力提取脚本

读取包含蛋白-配体信息的 CSV 文件，提取 Concatenated ID 和 Log Binding Affinity 两列，
并进行格式标准化（ID 转小写，数值保留指定小数位）。
"""

import csv
import os
import argparse
from pathlib import Path

def extract_affinity(input_file, output_file, precision=3):
    """
    提取 CSV 文件中的 Concatenated ID 和 Log Binding Affinity 列。
    将 Concatenated ID 转换为小写，结合能保留指定小数位，并保存到新文件。
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Processing {input_path} ...")

    try:
        with open(input_path, mode='r', encoding='utf-8', newline='') as infile, \
             open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.DictReader(infile)
            fieldnames = ['Concatenated ID', 'Log Binding Affinity']
            
            # check if columns exist
            # reader.fieldnames 可能是 None，需要先做判断
            if not reader.fieldnames:
                 print("Error: Empty file or no header found.")
                 return

            if not set(fieldnames).issubset(set(reader.fieldnames)):
                 missing = set(fieldnames) - set(reader.fieldnames)
                 print(f"Error: Missing columns in input file: {missing}")
                 return

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            count = 0
            for row in reader:
                # 提取并处理数据
                concatenated_id = row['Concatenated ID'].lower()
                affinity_str = row['Log Binding Affinity']
                
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
                
        print(f"Successfully processed {count} records.")
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ID and Affinity from CSV.")
    parser.add_argument("--input", "-i", type=str, help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file path")
    # 添加精度参数，默认为 6
    parser.add_argument("--precision", "-p", type=int, default=6, help="Decimal precision for affinity (default: 6)")
    
    args = parser.parse_args()
    
    # 默认路径使用绝对路径，确保在任何目录下运行都能找到文件
    default_base = Path("/pavo/glen/Code/EHFNet/data/raw/pdbbind")
    
    input_full_path = Path(args.input) if args.input else default_base / "hiqbind_info.csv"
    output_full_path = Path(args.output) if args.output else default_base / "hiqbind_labels.csv"

    print(f"Input: {input_full_path}")
    print(f"Output: {output_full_path}")
    print(f"Precision: {args.precision} decimal places")
    print("-" * 30)

    extract_affinity(input_full_path, output_full_path, precision=args.precision)
