import torch
import os
from tqdm import tqdm
from glob import glob

def compute_stats(data_dir, output_file):
    print(f"Scanning files in {data_dir}...")
    files = glob(os.path.join(data_dir, "*.pt"))
    
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    # Accumulators
    stats = {
        "ligand_atom": {"sum": None, "sum_sq": None, "count": 0},
        "protein_atom": {"sum": None, "sum_sq": None, "count": 0},
        "protein_residue": {"sum": None, "sum_sq": None, "count": 0},
        "ligand_molecule": {"sum": None, "sum_sq": None, "count": 0},
        "affinity": {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    }

    print(f"Processing {len(files)} files...")
    
    for f in tqdm(files):
        try:
            data = torch.load(f, weights_only=False)
            
            # Ligand Atom
            x = data['ligand_atom'].x_cont
            if stats["ligand_atom"]["sum"] is None:
                stats["ligand_atom"]["sum"] = torch.zeros(x.shape[1])
                stats["ligand_atom"]["sum_sq"] = torch.zeros(x.shape[1])
            stats["ligand_atom"]["sum"] += x.sum(dim=0)
            stats["ligand_atom"]["sum_sq"] += (x ** 2).sum(dim=0)
            stats["ligand_atom"]["count"] += x.shape[0]

            # Protein Atom
            x = data['protein_atom'].x_cont
            if stats["protein_atom"]["sum"] is None:
                stats["protein_atom"]["sum"] = torch.zeros(x.shape[1])
                stats["protein_atom"]["sum_sq"] = torch.zeros(x.shape[1])
            stats["protein_atom"]["sum"] += x.sum(dim=0)
            stats["protein_atom"]["sum_sq"] += (x ** 2).sum(dim=0)
            stats["protein_atom"]["count"] += x.shape[0]

            # Protein Residue
            x = data['protein_residue'].x_cont
            if stats["protein_residue"]["sum"] is None:
                stats["protein_residue"]["sum"] = torch.zeros(x.shape[1])
                stats["protein_residue"]["sum_sq"] = torch.zeros(x.shape[1])
            stats["protein_residue"]["sum"] += x.sum(dim=0)
            stats["protein_residue"]["sum_sq"] += (x ** 2).sum(dim=0)
            stats["protein_residue"]["count"] += x.shape[0]
            
            # Ligand Molecule
            x = data['ligand_molecule'].x_cont
            if stats["ligand_molecule"]["sum"] is None:
                stats["ligand_molecule"]["sum"] = torch.zeros(x.shape[1])
                stats["ligand_molecule"]["sum_sq"] = torch.zeros(x.shape[1])
            stats["ligand_molecule"]["sum"] += x.sum(dim=0)
            stats["ligand_molecule"]["sum_sq"] += (x ** 2).sum(dim=0)
            stats["ligand_molecule"]["count"] += x.shape[0]

            # Affinity
            if hasattr(data, "y_energy") and data.y_energy is not None:
                y = data.y_energy
                stats["affinity"]["sum"] += y.sum().item()
                stats["affinity"]["sum_sq"] += (y ** 2).sum().item()
                stats["affinity"]["count"] += y.numel()

        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    # Compute Mean and Std
    final_stats = {}
    for key, val in stats.items():
        if val["count"] > 0:
            mean = val["sum"] / val["count"]
            # Var = E[X^2] - (E[X])^2
            # Std = sqrt(Var)
            mean_sq = val["sum_sq"] / val["count"]
            if isinstance(mean, torch.Tensor):
                var = mean_sq - mean ** 2
                std = torch.sqrt(torch.clamp(var, min=1e-6))
            else:
                var = mean_sq - mean ** 2
                std = (var if var > 0 else 0.0) ** 0.5
                # Convert scalars to tensor
                mean = torch.tensor(mean)
                std = torch.tensor(std)
                
            final_stats[key] = {"mean": mean, "std": std}
            print(f"{key}: mean={mean}, std={std}")
        else:
            print(f"Warning: No data found for {key}")

    torch.save(final_stats, output_file)
    print(f"Stats saved to {output_file}")

if __name__ == "__main__":
    compute_stats(
        data_dir="data/processed/pdbbind/processed",
        output_file="data/processed/pdbbind/normalization_stats.pt"
    )
