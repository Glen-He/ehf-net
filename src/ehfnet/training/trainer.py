"""
训练循环

提供 EHFNet 模型的训练和验证功能。
"""

import os
import math
import torch
import logging
import gc
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# autocast / GradScaler 已移除：三维坐标平方距离在 FP16 下易溢出（‖r‖²>65504）

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
    rmsd_check_ratio: float = 1.0,
    accumulation_steps: int = 1,
    max_nodes_per_batch: int = 10000,
    ema_decay: float = 0.999,
):
    """
    训练 EHFNet 模型

    Args:
        data_root: PDBBind 数据根目录
        index_file: 索引 CSV 文件路径
        save_dir: 模型保存目录
        esm_path: 预计算的 ESM 嵌入路径
        epochs: 训练轮数

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
                          例如 0.1 表示随机抽取 10% 的 batch 进行耗时的 RMSD 推演        accumulation_steps: 梯度累积步数。当显存较小时，可设为 2/4 模拟更大 batch_size
        ema_decay: EMA 衰减率，默认 0.999；小规模试跑可设为 0.99 加快吸收    """

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

    # 统一亲和力统计来源：以当前 Dataset 统计为准，避免外部 stats 与训练集不一致
    if normalization_stats is None:
        normalization_stats = {}

    normalization_stats["affinity"] = {
        "mean": torch.tensor(dataset.affinity_stats["mean"], dtype=torch.float32),
        "std": torch.tensor(dataset.affinity_stats["std"], dtype=torch.float32),
    }

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

    from torch_geometric.loader import DynamicBatchSampler
    
    logger.info(f"Using DynamicBatchSampler with max_num={max_nodes_per_batch} nodes per batch.")
    # 注意: node mode 时, max_num 控制的是包含所有图累计的最大节点数
    train_sampler = DynamicBatchSampler(
        train_set,
        max_num=max_nodes_per_batch,
        mode="node",
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        collate_fn=collator.collate,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        batch_sampler=train_sampler,
    )
    
    val_sampler = DynamicBatchSampler(
        val_set,
        max_num=max_nodes_per_batch,
        mode="node",
        shuffle=False,
    )
    val_loader = DataLoader(
        val_set,
        collate_fn=collator.collate,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        batch_sampler=val_sampler,
    )

    # [新增逻辑] 动态计算 Batch 数量 (DynamicBatchSampler 没有固定的 len)
    try:
        total_val_batches = len(val_loader)

    except ValueError:
        total_val_batches = max(1, len(val_set) // 4)
        
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
        sigma_min=1e-3,
        warmup_epochs=warmup_epochs,
    )
    criterion = FlowMatchingLoss().to(device)
    # 速度分解由 matcher 内部完成，trainer 不持有分解器

    # 4. 优化器
    # [修改] criterion 无可学习参数，仅优化 model
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Warmup + 余弦退火（Step 级），防止初期梯度冲击 + 中后期平滑收敛
    # [修复] 以 optimizer.step() 次数（而非 batch 数）定义里程碑，
    try:
        total_train_batches = len(train_loader)
    except ValueError:
        total_train_batches = max(1, len(train_set) // 4)
    updates_per_epoch = math.ceil(total_train_batches / accumulation_steps)
    total_steps = epochs * updates_per_epoch
    warmup_steps = max(1, warmup_epochs) * updates_per_epoch  # LR warmup 与 Curriculum 同步
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps],
    )

    # EMA 模型：惰性初始化——EHFNet 含 LazyModule，参数在首次前向后才完成初始化；
    # 若在此处立即构建 AveragedModel 会对未初始化参数调用 .detach() 而崩溃。
    # 解决方案：首次 optimizer.step() 后再构建，此时 lazy 参数已确定。
    # ema_decay 由外部传入，小规模试跑可设 0.99，正式训练用 0.999。
    ema_model: AveragedModel | None = None

    # 5. 训练循环
    best_val_loss = float("inf")
    best_rmsd = float("inf")

    for epoch in range(epochs):
        # [新增] Epoch 开始前的深度清理
        # 这有助于整理上一轮遗留的碎片
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        criterion.train()

        train_loss_meter = 0.0
        pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="graphs")
        
        actual_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
            pbar.update(num_graphs)
            actual_batches += 1
            
            batch = batch.to(device)

            # [移除] 防爆显存保护：既然用了 DynamicBatchSampler，它会自动截断图大小。不再直接丢弃 >10000 原子的批次，因为现在单个批次可能就只有1个刚好万原子左右的图。

            # 动态梯度累积：由于使用了 DynamicBatchSampler，每次循环的真实图数量不再是固定的 batch_size。
            # 为了公平，我们可以退化成最简单的 batch 步数累积。
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()

            # 流匹配训练步骤
            # 生成训练目标不需要梯度
            with torch.no_grad():
                x_1 = batch["ligand_atom"].pos
                # matcher 直接返回 SE(3) x T^m 切向量目标字典
                t, x_t, targets = matcher.sample_location_and_target(
                    x_1=x_1,
                    data=batch,
                    current_epoch=epoch,
                    total_epochs=epochs,
                )

            batch["ligand_atom"].pos = x_t
            batch.t = t  # 注入时间步，供 Loss 时间掩码使用

            # FP32 前向传播（不使用 autocast，避免 FP16 距离平方溢出）
            predictions = model(batch, t)

            # 补充结合能 target
            targets["binding_affinity_target"] = batch.get("y_energy", None)

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"]

            # 防御性检查：若 loss 无梯度，跳过该 batch
            if loss.grad_fn is None:
                logger.warning(f"Batch {batch_idx}: loss has no grad_fn, skipping.")
                optimizer.zero_grad()
                continue

            if torch.isnan(loss) or loss > 200:
                logger.warning(f"{'NaN' if torch.isnan(loss) else 'Huge'} Loss on batch {batch_idx}, skipping.")
                for k, v in loss_dict.items():
                    logger.warning(f"  {k}: {v}")
                logger.warning(f"  GT Trans norm: {targets['v_trans_target'].norm().item()}")
                logger.warning(f"  GT Rot norm: {targets['v_rot_target'].norm().item()}")
                gt_tor = targets['v_torsion_target']
                if gt_tor.numel() > 0:
                    logger.warning(f"  GT Torsion norm: {gt_tor.norm().item()}")
                optimizer.zero_grad()
                continue

            # 反向传播（纯 FP32）
            # 梯度累积：将损失除以累积步数，确保梯度幅度与实际 batch_size 无关
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            # 仅在完成一个完整累积周期后才更新参数
            is_last_in_cycle = (batch_idx + 1) % accumulation_steps == 0
            if is_last_in_cycle:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                # GradScaler 移除后需手动补回：Inf/NaN 梯度直接跳过，防止权重被污染
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Batch {batch_idx}: grad_norm={grad_norm:.4g}, skipping optimizer step.")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    # 惰性构建 EMA（首次 step 后 lazy 参数已全部初始化）
                    if ema_model is None:
                        ema_model = AveragedModel(
                            model,
                            avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                        )
                    ema_model.update_parameters(model)
                    scheduler.step()

            # 记录日志
            train_loss_meter += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L_tr": f"{loss_dict.get('loss_trans', torch.tensor(0)).item():.3f}",
                    "L_rot": f"{loss_dict.get('loss_rot', torch.tensor(0)).item():.3f}",
                    "L_tor": f"{loss_dict.get('loss_torsion', torch.tensor(0)).item():.3f}",
                    "L_ene": f"{loss_dict.get('loss_energy', torch.tensor(0)).item():.3f}",
                    "L_cls": f"{loss_dict.get('loss_clash', torch.tensor(0)).item():.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # 循环结束，如果有剩余积累的梯度，进行最后一次 step
        if actual_batches > 0 and actual_batches % accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()
                if ema_model is None:
                    ema_model = AveragedModel(
                        model,
                        avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1.0 - ema_decay) * p,
                    )
                ema_model.update_parameters(model)
                scheduler.step()

        pbar.close()
        avg_train_loss = train_loss_meter / max(1, actual_batches)

        # 验证
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [新增] 训练结束，验证开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_val_loss = compute_validation_loss(
            model=ema_model if ema_model is not None else model,
            matcher=matcher,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_rmsd_batches=rmsd_check_batches,
            dataset=dataset,
            warmup_epochs=warmup_epochs,
        )
        
        # [新增] 验证结束，下一轮开始前的清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 提取指标
        if isinstance(avg_val_loss, dict):
            val_metrics = avg_val_loss
            avg_val_loss_scalar = val_metrics["val_loss"]
            mean_rmsd = val_metrics["mean_rmsd_final"]
        else:
            avg_val_loss_scalar = avg_val_loss
            mean_rmsd = float("inf")

        # ReduceLROnPlateau 已移除，scheduler 已在 Step 级自动推进

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss_scalar:.4f} | Val RMSD: {mean_rmsd:.4f}"
        )

        # 准备保存数据
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.module.state_dict() if ema_model is not None else model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_rmsd": best_rmsd,
        }

        # 1. 始终保存最新模型（作为保底）
        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pt"))

        # 2. Best model：Curriculum 结束后 + RMSD 创新低时保存
        is_warmup = epoch < warmup_epochs
        if not is_warmup and mean_rmsd < best_rmsd:
            best_rmsd = mean_rmsd
            checkpoint["best_rmsd"] = best_rmsd
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            logger.info(f"Saved best model with Mean Final RMSD: {best_rmsd:.4f}")

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
    dataset: PDBBindDataset | None = None,
    warmup_epochs: int = 20,
) -> dict | float:
    """
    验证函数：计算 Loss 并统计全量 RMSD 指标
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    # 亲和力统计容器
    affinity_preds = []
    affinity_targets = []
    
    # RMSD 统计容器
    all_rmsd_init = []
    all_rmsd_final = []
    
    # 固定随机种子 (保持验证集生成的一致性)
    if epoch is not None:
        torch.manual_seed(42 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

    # 使用 tqdm 显示验证进度，按精确样本数统计
    pbar = tqdm(total=len(loader.dataset), desc=f"Epoch {(epoch or 0) + 1} [Val]", leave=False, unit="graphs")

    for i, batch in enumerate(loader):
        num_graphs = int(batch["ligand_atom"].batch.max().item()) + 1
        pbar.update(num_graphs)
        try:
            batch = batch.to(device)
            x_1 = batch["ligand_atom"].pos

            # 1. 计算 Loss (用于早停和模型选择)
            t, x_t, targets = matcher.sample_location_and_target(
                x_1=x_1,
                data=batch,
                current_epoch=epoch if epoch is not None else 0,
                total_epochs=total_epochs
            )

            batch["ligand_atom"].pos = x_t
            batch.t = t  # 注入时间步，供 Loss 时间掩码使用

            predictions = model(batch, t)

            # matcher 已返回分解好的 SE(3) 目标，直接补全结合能
            targets["binding_affinity_target"] = batch.get("y_energy", None)

            loss_dict = criterion(predictions, targets, batch)
            loss = loss_dict["total"]
            
            # 过滤爆炸 Loss
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                total_loss += loss.item()
                valid_batches += 1
                
            # [新增] 收集亲和力预测 (用于计算 RMSE)
            # 只有当 Loss 有效时，且 t > 0.5 时，才收集预测值，避免无监督的噪声污染统计数据
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1e6:
                # 遵循 losses.py 中的物理约束，仅在 t > 0.5 时监督能量
                if t is not None:
                    valid_mask = t > 0.5
                else:
                    valid_mask = torch.ones_like(batch.get("y_energy", torch.zeros(1)), dtype=torch.bool)
                
                if valid_mask.any():
                    pred_aff = predictions.get("binding_affinity", None)
                    if pred_aff is not None:
                        # 仅选取 t > 0.5 的预测值
                        pred_aff_valid = pred_aff[valid_mask]
                        # 双重检查：预测值本身也不能含 NaN
                        if not torch.isnan(pred_aff_valid).any():
                            affinity_preds.append(pred_aff_valid.cpu())
                            # target 统一为 raw（若已提供 raw 则直接用，否则做一次反归一化）
                            if hasattr(batch, "y_energy_raw"):
                                target_raw_valid = batch.y_energy_raw[valid_mask]
                                affinity_targets.append(target_raw_valid.cpu())
                            else:
                                y_norm = batch.get("y_energy", None)
                                if y_norm is not None and dataset is not None:
                                    target_raw_valid = dataset.denormalize_affinity(y_norm[valid_mask].cpu())
                                    affinity_targets.append(target_raw_valid)
            
            # 2. 全量 RMSD 推演
            # -----------------------------------------------------------
            if i < max_rmsd_batches:
                try:
                    # 克隆数据用于推演
                    infer_batch = batch.clone()
                    infer_batch["ligand_atom"].pos = x_1 
                    
                    # 验证始终使用全难度扰动（与最终推理条件一致）
                    x_0_infer = matcher._generate_random_pose(
                        x_ref=x_1,
                        batch=infer_batch["ligand_atom"].batch,
                        B=int(infer_batch["ligand_atom"].batch.max().item()) + 1,
                        masses=infer_batch["ligand_atom"].masses,
                        torsion_indices=getattr(infer_batch, "torsion_indices", None),
                        torsion_moving_mask=getattr(infer_batch, "torsion_moving_mask", None),
                        epoch=warmup_epochs,
                    )
                    
                    # 记录初始 RMSD
                    sq_diff_init = ((x_0_infer - x_1) ** 2).sum(dim=-1)
                    msd_init = scatter_mean(sq_diff_init, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_init = torch.sqrt(msd_init)
                    # [修改] 强制转 CPU，切断 GPU 显存占用
                    all_rmsd_init.append(rmsd_init.detach().cpu())

                    # 执行推演（100 步 RK4，与推理对齐）
                    infer_batch["ligand_atom"].pos = x_0_infer
                    final_pos, _ = matcher.ode_solve(
                        model=model,
                        data=infer_batch,
                        steps=100,
                        method="rk4"
                    )
                    
                    # 记录最终 RMSD
                    sq_diff_final = ((final_pos - x_1) ** 2).sum(dim=-1)
                    msd_final = scatter_mean(sq_diff_final, infer_batch["ligand_atom"].batch, dim=0)
                    rmsd_final = torch.sqrt(msd_final)
                    # [修改] 强制转 CPU，切断 GPU 显存占用
                    all_rmsd_final.append(rmsd_final.detach().cpu())
                    
                except Exception as e:
                    logger.warning(f"RMSD inference failed for batch {i}: {e}")
            # -----------------------------------------------------------

        except Exception as e:
            logger.warning(f"Validation batch failed: {e}")
            continue

    # 计算并打印统计结果
    if len(affinity_preds) > 0 and dataset is not None:
        cat_preds = torch.cat(affinity_preds).view(-1)
        cat_targets = torch.cat(affinity_targets).view(-1)
        
        # 模型输出是 norm，验证时仅在这里做一次反归一化
        raw_preds: torch.Tensor = dataset.denormalize_affinity(cat_preds)
        
        mse_val = F.mse_loss(raw_preds, cat_targets)
        rmse_val = torch.sqrt(mse_val).item()
        mae_val = F.l1_loss(raw_preds, cat_targets).item()
        
        logger.info(f"[Validation Affinity] RMSE: {rmse_val:.4f} pKd | MAE: {mae_val:.4f} pKd")

    mean_final = float("inf")
    success_5a = 0.0

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
        logger.info(f"[Validation Full Stats] Epoch {(epoch or 0) + 1}")
        logger.info(f"  Mean RMSD: {mean_init:.2f} -> {mean_final:.2f} Å")
        logger.info(f"  Success Rate (<2Å): {success_2a:.2f}%")
        logger.info(f"  Success Rate (<5Å): {success_5a:.2f}%")
        logger.info("-" * 60)

    if valid_batches == 0:
        return {"val_loss": float("nan"), "mean_rmsd_final": float("inf"), "success_5a": 0.0}

    # [新增] 显式清理现场
    del all_rmsd_init
    del all_rmsd_final
    del affinity_preds
    del affinity_targets
    
    # 强制通知 CUDA 释放缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "val_loss": total_loss / valid_batches,
        "mean_rmsd_final": mean_final,
        "success_5a": success_5a,
    }
