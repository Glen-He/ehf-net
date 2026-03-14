import os
import sys
import time
import argparse
import torch
import gc
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ehfnet.training.trainer import GraphCollator
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.models.ehfnet import EHFNet
from ehfnet.encoders.feature_specs import (
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.training.flow_matcher import ConditionalFlowMatcher
from ehfnet.training.losses import FlowMatchingLoss

def profile():
    data_root = "data/processed/pdbbind"
    index_file = "data/processed/pdbbind/index.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing Dataset...")
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])
    dataset = PDBBindDataset(
        root=data_root, index_file=index_file, esm_root=None,
        esm="auto",  pocket_radius=20.0,
    )
    
    train_loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collator.collate, persistent_workers=True
    )
    
    print("Initializing Model...")
    model = EHFNet(
        hidden_dim=128,
        time_dim=64,
        num_gnn_blocks=6,
        lig_atom_cont_count=len(LIGAND_ATOM_CONT_SCHEMA),
        lig_mol_cont_count=len(LIGAND_MOLECULE_CONT_SCHEMA),
        pro_atom_cont_count=len(PROTEIN_ATOM_CONT_SCHEMA),
        pro_res_cont_count=len(PROTEIN_RESIDUE_CONT_SCHEMA) + 960,
    ).to(device)
    
    matcher = ConditionalFlowMatcher()
    criterion = FlowMatchingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    criterion.train()

    times = {"data_wait": 0., "data_to_gpu": 0., "matcher": 0., "forward": 0., "loss": 0., "backward": 0., "optim": 0.}
    
    print("Starting Profiling for 10 batches...")
    
    # Run warmup
    print("Warmup...")
    batch = next(iter(train_loader))
    batch = batch.to(device)
    t, x_t, targets = matcher.sample_location_and_target(x_1=batch["ligand_atom"].pos, data=batch, current_epoch=0, total_epochs=200)
    batch["ligand_atom"].pos = x_t
    batch.t = t
    loss_dict = criterion(model(batch, t), targets, batch)
    loss_dict["total"].backward()
    optimizer.step()
    print("Warmup done.")

    data_iterator = iter(train_loader)
    
    for batch_idx in range(10):
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t0 = time.time()
        batch = next(data_iterator)
        torch.cuda.synchronize()
        times["data_wait"] += time.time() - t0
        
        t0 = time.time()
        batch = batch.to(device)
        num_protein_atoms = batch["protein_atom"].pos.shape[0]
        
        if num_protein_atoms > 10000:
            continue

        torch.cuda.synchronize()
        times["data_to_gpu"] += time.time() - t0
        
        t0 = time.time()
        with torch.no_grad():
            x_1 = batch["ligand_atom"].pos
            t, x_t, targets = matcher.sample_location_and_target(
                x_1=x_1, data=batch, current_epoch=0, total_epochs=200
            )
        batch["ligand_atom"].pos = x_t
        batch.t = t
        torch.cuda.synchronize()
        times["matcher"] += time.time() - t0
        
        t0 = time.time()
        predictions = model(batch, t)
        torch.cuda.synchronize()
        times["forward"] += time.time() - t0
        
        t0 = time.time()
        targets["binding_affinity_target"] = batch.get("y_energy", None)
        loss_dict = criterion(predictions, targets, batch)
        loss = loss_dict["total"]
        torch.cuda.synchronize()
        times["loss"] += time.time() - t0
        
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        times["backward"] += time.time() - t0
        
        t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        times["optim"] += time.time() - t0
        
        print(f"Processed batch {batch_idx+1}/10 (Atoms: {num_protein_atoms})")
        
    print("\n--- Profiling Results (Total Time for Valid Batches) ---")
    total_time = sum(times.values())
    for k, v in times.items():
        print(f"{k.capitalize():<12}: {v:.3f}s ({(v/total_time)*100:.1f}%)")
    print(f"Total time for 10 batches: {total_time:.3f}s")

if __name__ == "__main__":
    profile()
