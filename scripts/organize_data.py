"""
数据重组脚本

将分散在 ligand/ 和 protein/ 文件夹中的原始数据，按照 PDB ID 重新组织成
标准的子文件夹结构，并生成对应的索引文件。
"""

import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def organize_data(raw_root, target_root, index_file):
    """
    Reorganize flat raw data into standard PDBBind structure.
    
    Source structure:
        raw_root/
            ligand/
                {pdb_id}_ligand.sdf
            protein/
                {pdb_id}_protein.pdb
                {pdb_id}_esm.npz (optional)
                
    Target structure:
        target_root/
            index.csv
            cleaned/    <-- Changed from 'raw'
                {pdb_id}/
                    {pdb_id}_ligand.sdf
                    {pdb_id}_protein.pdb
                    {pdb_id}_esm.npz (optional)
    """
    
    # 1. Setup target directories
    # 更改子文件夹名称：从 'raw' 改为 'cleaned'
    target_subdir = os.path.join(target_root, "cleaned")
    os.makedirs(target_subdir, exist_ok=True)
    
    print(f"Source: {raw_root}")
    print(f"Target: {target_root}")
    
    # 2. Process Index CSV
    print("Processing index file...")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
        
    df = pd.read_csv(index_file)
    
    # Map custom columns to standard names
    rename_map = {
        "Concatenated ID": "pdb_id",
        "Log Binding Affinity": "affinity"
    }
    
    # Check if mapping is needed
    if "Concatenated ID" in df.columns:
        df.rename(columns=rename_map, inplace=True)
        print("  Renamed columns to standard format.")
    
    # Ensure required columns exist
    if "pdb_id" not in df.columns or "affinity" not in df.columns:
        raise ValueError(f"CSV missing required columns (pdb_id, affinity). Found: {df.columns.tolist()}")
        
    # Save standard index.csv
    target_index_path = os.path.join(target_root, "index.csv")
    df[["pdb_id", "affinity"]].to_csv(target_index_path, index=False)
    print(f"  Saved standard index to: {target_index_path}")
    
    # 3. Reorganize Files
    print("Reorganizing files...")
    ligand_dir = os.path.join(raw_root, "ligand")
    protein_dir = os.path.join(raw_root, "protein")
    
    success_count = 0
    missing_count = 0
    
    for pdb_id in tqdm(df["pdb_id"]):
        pdb_id = str(pdb_id).lower()
        
        # Define source paths
        # Note: Handling potential variations in naming if necessary. 
        # Assuming strict naming: {pdb_id}_ligand.sdf / {pdb_id}_protein.pdb
        src_ligand = os.path.join(ligand_dir, f"{pdb_id}_ligand.sdf")
        # Try .mol2 if .sdf is missing
        if not os.path.exists(src_ligand):
             src_ligand_mol2 = os.path.join(ligand_dir, f"{pdb_id}_ligand.mol2")
             if os.path.exists(src_ligand_mol2):
                 src_ligand = src_ligand_mol2
        
        src_protein = os.path.join(protein_dir, f"{pdb_id}_protein.pdb")
        src_esm = os.path.join(protein_dir, f"{pdb_id}_esm.npz")
        
        # Check existence
        if not os.path.exists(src_ligand) or not os.path.exists(src_protein):
            # print(f"Missing files for {pdb_id}")
            missing_count += 1
            continue
            
        # Create target folder
        # target_subdir 是我们在开头定义的新变量名 (指向 cleaned 文件夹)
        target_pdb_dir = os.path.join(target_subdir, pdb_id)
        os.makedirs(target_pdb_dir, exist_ok=True)
        
        # Copy files
        try:
            shutil.copy2(src_ligand, target_pdb_dir)
            shutil.copy2(src_protein, target_pdb_dir)
            
            if os.path.exists(src_esm):
                shutil.copy2(src_esm, target_pdb_dir)
                
            success_count += 1
        except Exception as e:
            print(f"Error copying {pdb_id}: {e}")
            
    print(f"\nDone!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Missing/Skipped: {missing_count}")
    print(f"  Output directory: {target_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize raw data into PDBBind structure")
    parser.add_argument("--raw_root", type=str, default="/pavo/glen/Code/EHFNet/data/raw/pdbbind", help="Path to source raw directory (containing ligand/protein folders)")
    # 更改默认输出目录名为 pdbbind_processed
    parser.add_argument("--target_root", type=str, default="/pavo/glen/Code/EHFNet/data/processed/pdbbind", help="Path to target directory")
    parser.add_argument("--index_file", type=str, default="/pavo/glen/Code/EHFNet/data/raw/pdbbind/hiqbind_filtered.csv", help="Path to source CSV index file")
    
    args = parser.parse_args()
    
    # 打印一下当前的配置，方便用户确认
    print("-" * 30)
    print(f"Index File  : {args.index_file}")
    print(f"Raw Root    : {args.raw_root}")
    print(f"Target Root : {args.target_root}")
    print("-" * 30)

    organize_data(args.raw_root, args.target_root, args.index_file)
