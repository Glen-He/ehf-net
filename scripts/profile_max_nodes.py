"""
Profile Max Nodes Script

逐步增大测试图的节点数，以测算当前硬件与模型配置能支撑的 `max_nodes_per_batch` 上限。
"""

import sys
import gc
import torch
import logging

from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from torch_geometric.data import HeteroData
from ehfnet.models import EHFNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_dummy_batch(num_nodes: int, device: torch.device) -> HeteroData:
    """创建占位数据的异构图"""
    
    # 模拟配体和蛋白的大致比例 (例如蛋白很大，配体很小)
    n_prot = int(num_nodes * 0.9)
    n_lig = num_nodes - n_prot
    
    data = HeteroData()
    
    # 填入 EHFNet 所需的节点特征形状
    # LIGAND
    data["ligand_atom"].x_cat = torch.zeros((n_lig, 18), dtype=torch.long, device=device)
    data["ligand_atom"].x_cont = torch.randn((n_lig, 9), dtype=torch.float32, device=device)
    data["ligand_atom"].pos = torch.randn((n_lig, 3), dtype=torch.float32, device=device)
    data["ligand_atom"].batch = torch.zeros(n_lig, dtype=torch.long, device=device)
    
    data["ligand_molecule"].x_cont = torch.randn((1, 9), dtype=torch.float32, device=device)
    data["ligand_molecule"].batch = torch.zeros(1, dtype=torch.long, device=device)
    
    # PROTEIN
    data["protein_atom"].x_cat = torch.zeros((n_prot, 1), dtype=torch.long, device=device)
    data["protein_atom"].x_cont = torch.randn((n_prot, 5), dtype=torch.float32, device=device)
    data["protein_atom"].pos = torch.randn((n_prot, 3), dtype=torch.float32, device=device)
    data["protein_atom"].batch = torch.zeros(n_prot, dtype=torch.long, device=device)
    data["protein_atom"].residue_idx = torch.arange(n_prot, dtype=torch.long, device=device)
    
    # 假设每个 atom 对应一个 residue (最坏情况测试)
    data["protein_residue"].x_cat = torch.zeros((n_prot, 2), dtype=torch.long, device=device)
    data["protein_residue"].x_cont = torch.randn((n_prot, 14 + 960), dtype=torch.float32, device=device)
    data["protein_residue"].pos = torch.randn((n_prot, 3), dtype=torch.float32, device=device)
    data["protein_residue"].batch = torch.zeros(n_prot, dtype=torch.long, device=device)
    data["protein_residue"].esm_missing_mask = torch.zeros(n_prot, dtype=torch.bool, device=device)
    
    data["protein_pocket"].x_cont = torch.randn((1, 14+960), dtype=torch.float32, device=device)
    data["protein_pocket"].batch = torch.zeros(1, dtype=torch.long, device=device)
    data["protein_pocket"].num_nodes = 1
    
    # 构建密集的 KNN 边 (模拟 builder.py 的 K=128)
    k_intra = min(128, n_prot - 1)
    if k_intra > 0:
        # P-P
        src = torch.arange(n_prot, device=device).repeat_interleave(k_intra)
        dst = torch.randperm(n_prot, device=device)[:k_intra].repeat(n_prot)
        data["protein_residue", "interacts_with", "protein_residue"].edge_index = torch.stack([src, dst])
        data["protein_atom", "bonded_to", "protein_atom"].edge_index = torch.stack([src, dst])
        
        # INTER-EDGES (L-P)
        k_inter = min(32, n_lig - 1)
        if k_inter > 0:
            src_lp = torch.arange(n_lig, device=device).repeat_interleave(k_inter)
            dst_lp = torch.randperm(n_prot, device=device)[:k_inter].repeat(n_lig)
            data["ligand_atom", "inter_proximity", "protein_residue"].edge_index = torch.stack([src_lp, dst_lp])
            data["protein_residue", "inter_proximity", "ligand_atom"].edge_index = torch.stack([dst_lp, src_lp])
            
            data["ligand_atom", "inter_proximity", "protein_atom"].edge_index = torch.stack([src_lp, dst_lp])
            data["protein_atom", "inter_proximity", "ligand_atom"].edge_index = torch.stack([dst_lp, src_lp])
    
    # 其它约束
    data.torsion_indices = torch.zeros((0, 4), dtype=torch.long, device=device)
    data.torsion_moving_mask = torch.zeros((0, n_lig), dtype=torch.bool, device=device)
    data.t = torch.rand(1, device=device)
    
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("No CUDA device found. Profiling on CPU is meaningless for OOM.")
        return
        
    logger.info(f"Initializing EHFNet on {device}...")
    
    model = EHFNet(
        hidden_dim=128,
        time_dim=128,
        num_gnn_blocks=4,
        lig_atom_cont_count=9,
        lig_mol_cont_count=9,
        pro_atom_cont_count=5,
        pro_res_cont_count=14+960,
    ).to(device)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    start_nodes = 2000
    step_nodes = 2000
    max_test_nodes = 40000
    
    last_successful_nodes = 0
    
    logger.info("Starting memory profiler. This will intentionally cause an OOM error.")
    logger.info("-" * 50)
    
    for current_nodes in range(start_nodes, max_test_nodes + step_nodes, step_nodes):
        try:
            logger.info(f"Testing {current_nodes} nodes...")
            
            # 清理显存
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            
            # 前向+反向惩罚
            batch = create_dummy_batch(current_nodes, device)
            out = model(batch, batch.t)
            
            loss = out["v_trans"].sum() + out["v_rot"].sum()
            loss.backward()
            optimizer.step()
            
            last_successful_nodes = current_nodes
            logger.info(f"  -> SUCCESS! Max VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"  -> OOM at {current_nodes} nodes!")
                break
            else:
                logger.error(f"  -> Unexpected Error: {e}")
                break
                
    logger.info("-" * 50)
    logger.info(f"Profiling complete.")
    logger.info(f"Last successful node count: {last_successful_nodes}")
    if last_successful_nodes > 0:
        recommended = int(last_successful_nodes * 0.8)
        logger.info(f"RECOMMENDED --max_nodes_per_batch: {recommended}")
        logger.info("You can use this value safely in training to maximize GPU usage.")

if __name__ == "__main__":
    main()
