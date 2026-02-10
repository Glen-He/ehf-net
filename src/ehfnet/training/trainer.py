"""
训练循环

提供 EHFNet 模型的训练和验证功能。
"""

import os
import math
import torch
import logging
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from torch_scatter import scatter_mean

from ehfnet.models import EHFNet
from ehfnet.graph import GraphCollator
from ehfnet.datasets.pdbbind import PDBBindDataset
from ehfnet.datasets.splitter import ScaffoldSplitter
from ehfnet.training.losses import FlowMatchingLoss
from ehfnet.training.flow_matcher import ConditionalFlowMatcher


logger = logging.getLogger(__name__)


def train(
    *,
    data_root: str,
    index_file: str,
    save_dir: str = "./checkpoints",
    esm_path: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    clip_grad: float = 10.0,
    hidden_dim: int = 128,
    num_gnn_blocks: int = 6,
    lig_atom_cont_count: int = 9,
    lig_mol_cont_count: int = 9,
    pro_atom_cont_count: int = 5,
    pro_res_cont_count: int = 974,     # 14 (torsion) + 960 (ESM)
    esm_dim: int = 960,
    device: str | torch.device = "auto",
    pocket_radius: float | None = 20.0,
    normalization_stats: dict | None = None,
    warmup_epochs: int = 20,
    rmsd_check_ratio: float = 0.1,
):
    """
    训练 EHFNet 模型

    Args:
        data_root: PDBBind 数据根目录
        index_file: 索引 CSV 文件路径
        save_dir: 模型保存目录
        esm_path: 预计算的 ESM 嵌入路径
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减
        clip_grad: 梯度裁剪阈值
        hidden_dim: 隐藏层维度
        num_gnn_blocks: GNN 块数量
        lig_atom_cont_count: 配体原子连续特征数量
        lig_mol_cont_count: 配体分子连续特征数量
        pro_atom_cont_count: 蛋白原子连续特征数量
        pro_res_cont_count: 蛋白残基连续特征数量
        esm_dim: ESM embedding 维度
        device: 训练设备 ("cpu", "cuda", "cuda:0", "cuda:1" 等)，默认为 "auto" (自动检测)
        pocket_radius: 口袋提取半径 (Å)
        normalization_stats: 归一化统计数据
        warmup_epochs: 空间课程学习预热轮数
        rmsd_check_ratio: 验证集中计算 RMSD 的样本比例 (0.0 ~ 1.0)
                          例如 0.1 表示随机抽取 10% 的 batch 进行耗时的 RMSD 推演
    """

    # 1. 准备环境
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device(device)
        
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Using device: {device}")

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)

    except Exception:
        pass

    # 2. 准备数据
    logger.info("Initializing Dataset...")
    collator = GraphCollator(follow_batch=["ligand_atom", "protein_atom"])

    dataset = PDBBindDataset(
        root=data_root,
        index_file=index_file,
        esm_root=esm_path,
        esm="auto",
        esm_dim=esm_dim,
        pocket_radius=pocket_radius,
    )

    # [修改] 使用工程级 Scaffold Splitter
    logger.info("Splitting dataset by Scaffold...")
    
    # 实例化 Splitter
    splitter = ScaffoldSplitter(include_chirality=False, seed=42)
    
    # 执行划分 (Train 90%, Val 10%, Test 0%)
    train_set, val_set, _ = splitter.split(
        dataset,
        frac_train=0.9,
        frac_val=0.1,
        frac_test=0.0
    )

    logger.info(f"Final Dataset Sizes: Train={len(train_set)}, Val={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collator.collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collator.collate,
        pin_memory=True,
    )

    # [新增逻辑] 计算需要检查的 Batch 数量
    total_val_batches = len(val_loader)
    rmsd_check_batches = int(total_val_batches * rmsd_check_ratio)
    # 确保至少检查 1 个 batch (如果 ratio > 0)
    if rmsd_check_ratio > 0 and rmsd_check_batches < 1:
        rmsd_check_batches = 1
    
    logger.info(f"Validation Sampling: Check RMSD for {rmsd_check_batches}/{total_val_batches} batches ({rmsd_check_ratio*100:.1f}%)")

    # 3. 准备模型组件
    logger.info("Initializing Model & Flow Components...")

    model = EHFNet(
        hidden_dim=hidden_dim,
        time_dim=hidden_dim,
        num_gnn_blocks=num_gnn_blocks,
        lig_atom_cont_count=lig_atom_cont_count,
        lig_mol_cont_count=lig_mol_cont_count,
        pro_atom_cont_count=pro_atom_cont_count,
        pro_res_cont_count=pro_res_cont_count,
        normalization_stats=normalization_stats,
    ).to(device)

    matcher = ConditionalFlowMatcher(
        sigma_min=1e-4,
        warmup_epochs=warmup_epochs,
    )
    criterion = FlowMatchingLoss().to(device)

    # 4. 优化器
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=5
    )

    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # 5. 训练循环
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        criterion.train()

        train_loss_meter = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 流匹配训练步骤
            # 生成训练目标不需要梯度
            with torch.no_grad():
                x_1 = batch["ligand_atom"].pos
                t, x_t, v_target = matcher.sample_location_and_target(
                    x_1=x_1, 
                    data=batch, 
                    x_0=None,
                    current_epoch=epoch,
                    total_epochs=epochs,
                )

            batch["ligand_atom"].pos = x_t

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                predictions = model(batch, t)

                targets = {
                    "v_atomic_target": v_target,
                    "binding_affinity_target": batch.get("y_energy", None),
                }

                loss_dict = criterion(predictions, targets, batch)
                loss = loss_dict["total"]

                if torch.isnan(loss) or loss > 1e6:
                    logger.error(f"{'NaN' if torch.isnan(loss) else 'Huge'} Loss detected!")
                    for k, v in loss_dict.items():
                        logger.error(f"  {k}: {v}")
                    
                    # 检查目标值
                    gt_trans, gt_rot, gt_torsion = criterion.decomposer.decompose(
                        pos=batch["ligand_atom"].pos,
                        vel=targets["v_atomic_target"],
                        masses=batch["ligand_atom"].masses,
                        batch=batch["ligand_atom"].batch,
                        torsion_indices=getattr(batch, "torsion_indices", None),
                        torsion_moving_mask=getattr(batch, "torsion_moving_mask", None)
                    )
                    logger.error(f"  GT Trans norm: {gt_trans.norm().item()}")
                    logger.error(f"  GT Rot norm: {gt_rot.norm().item()}")
                    if gt_torsion.numel() > 0:
                        logger.error(f"  GT Torsion norm: {gt_torsion.norm().item()}")
                    
                    raise RuntimeError("Stopping due to invalid Loss")

            # 反向传播
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # 记录日志
            train_loss_meter += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "w_tr": f"{loss_dict.get('weight_trans', torch.tensor(0)).item():.2f}",
                    "w_rot": f"{loss_dict.get('weight_rot', torch.tensor(0)).item():.2f}",
                    "w_tor": f"{loss_dict.get('weight_torsion', torch.tensor(0)).item():.2f}",
                    "w_ene": f"{loss_dict.get('weight_energy', torch.tensor(0)).item():.2f}",
                }
            )

        avg_train_loss = train_loss_meter / len(train_loader)

        # 验证
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_val_loss = compute_validation_loss(
            model=model,
            matcher=matcher,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_rmsd_batches=rmsd_check_batches,
        )
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # 准备保存数据
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        # 1. 始终保存最新模型（作为保底）
        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pt"))

        # 2. 只有当 Val Loss 有效且创下新低时保存最佳模型
        if not math.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")

        # 3. 每 10 轮保存一个永久备份
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))


@torch.no_grad()
def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int = 1,
    max_rmsd_batches: int = 10,
) -> float:
    """
    验证函数：计算 Loss 并统计全量 RMSD 指标
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    # RMSD 统计容器
    all_rmsd_init = []
    all_rmsd_final = []
    
    # 固定随机种子 (保持验证集生成的一致性)
    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

    # 使用 tqdm 显示验证进度，因为现在推演会稍微花点时间
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for i, batch in enumerate(pbar):
        try:
            batch = batch.to(device)
            x_1 = batch["ligand_atom"].pos

            # 1. 计算 Loss (用于早停和模型选择)
            t, x_t, v_target = matcher.sample_location_and_target(
                x_1=x_1, 
                data=batch, 
                current_epoch=epoch if epoch is not None else 0,
                total_epochs=total_epochs
            )

            batch["ligand_atom"].pos = x_t
            predictions = model(batch, t)

            targets = {
                "v_atomic_target": v_target,
                "binding_affinity_target": batch.get("y_energy", None),
            }

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"]
            
            # 过滤爆炸 Loss
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                total_loss += loss.item()
                valid_batches += 1

            # 2. 全量 RMSD 推演
            # -----------------------------------------------------------
            if i < max_rmsd_batches:
                try:
                    # 克隆数据用于推演
                    infer_batch = batch.clone()
                    infer_batch["ligand_atom"].pos = x_1 
                    
                    # 获取当前空间尺度
                    current_scale = matcher.get_spatial_scale(epoch if epoch is not None else 0)

                    # 生成随机初始位姿
                    x_0_infer = matcher._generate_random_pose(
                        x_ref=x_1,
                        batch=infer_batch["ligand_atom"].batch,
                        B=int(infer_batch["ligand_atom"].batch.max().item()) + 1,
                        masses=infer_batch["ligand_atom"].masses,
                        torsion_indices=getattr(infer_batch, "torsion_indices", None),
                        torsion_moving_mask=getattr(infer_batch, "torsion_moving_mask", None),
                        translation_scale=current_scale
                    )
                    
                    # 记录初始 RMSD
                    sq_diff_init = ((x_0_infer - x_1) ** 2).sum(dim=-1)
                    msd_init = scatter_mean(sq_diff_init, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_init = torch.sqrt(msd_init)
                    all_rmsd_init.append(rmsd_init)

                    # 执行推演
                    infer_batch["ligand_atom"].pos = x_0_infer
                    final_pos, _ = matcher.ode_solve(
                        model=model,
                        data=infer_batch,
                        steps=20,       # 保持 20 步以兼顾速度和精度
                        method="euler"
                    )
                    
                    # 记录最终 RMSD
                    sq_diff_final = ((final_pos - x_1) ** 2).sum(dim=-1)
                    msd_final = scatter_mean(sq_diff_final, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_final = torch.sqrt(msd_final)
                    all_rmsd_final.append(rmsd_final)
                    
                except Exception as e:
                    logger.warning(f"RMSD inference failed for batch {i}: {e}")
            # -----------------------------------------------------------

        except Exception as e:
            logger.warning(f"Validation batch failed: {e}")
            continue

    # 计算并打印统计结果
    if len(all_rmsd_final) > 0:
        # 拼接所有 batch 的 RMSD
        cat_rmsd_init = torch.cat(all_rmsd_init)
        cat_rmsd_final = torch.cat(all_rmsd_final)
        
        mean_init = cat_rmsd_init.mean().item()
        mean_final = cat_rmsd_final.mean().item()
        
        # 计算成功率 (<2A 和 <5A)
        success_2a = (cat_rmsd_final < 2.0).float().mean().item() * 100
        success_5a = (cat_rmsd_final < 5.0).float().mean().item() * 100
        
        logger.info("-" * 60)
        logger.info(f"[Validation Full Stats] Epoch {epoch+1}")
        logger.info(f"  Mean RMSD: {mean_init:.2f} -> {mean_final:.2f} Å")
        logger.info(f"  Success Rate (<2Å): {success_2a:.2f}%")
        logger.info(f"  Success Rate (<5Å): {success_5a:.2f}%")
        logger.info("-" * 60)

    if valid_batches == 0:
        return float("nan")
        
    return total_loss / valid_batches
