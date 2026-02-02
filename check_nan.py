import torch
import os

from tqdm import tqdm

root = "/pavo/glen/Code/EHFNet/data/processed/pdbbind/processed"
files = [f for f in os.listdir(root) if f.endswith('.pt')]
nan_files = []

print(f"Checking {len(files)} files for NaN...")

for f in tqdm(files):

    try:
        path = os.path.join(root, f)
        data = torch.load(path, weights_only=False)
        
        # 检查坐标
        if torch.isnan(data['ligand_atom'].pos).any():
            nan_files.append((f, "ligand_pos"))
            continue

        if torch.isnan(data['protein_atom'].pos).any():
            nan_files.append((f, "protein_pos"))
            continue
            
        # 检查 ESM
        if hasattr(data['protein_atom'], 'x') and data['protein_atom'].x is not None:
            
            if torch.isnan(data['protein_atom'].x).any():
                nan_files.append((f, "protein_esm"))
                continue
                
    except Exception as e:
        nan_files.append((f, f"error: {e}"))

print("\nSummary:")
print(f"Total files checked: {len(files)}")
print(f"Total NaN/Error files: {len(nan_files)}")

if nan_files:
    print("First 10 problematic files:")

    for f, reason in nan_files[:10]:
        print(f"  {f}: {reason}")
