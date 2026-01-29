"""
训练循环

提供 EHFNet 模型的训练和验证功能。
"""

import os
import torch
import logging
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from ehfnet.models import EHFNet
from ehfnet.graph import GraphCollator
from ehfnet.datasets.pdbbind import PDBBindDataset
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

    # 简单的划分（实际项目中建议按 scaffold 划分）
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator.collate,
        pin_memory=True,
    )

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
    ).to(device)

    matcher = ConditionalFlowMatcher(sigma_min=1e-4)
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
                    x_1=x_1, data=batch, x_0=None
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
        )
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": criterion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")


@torch.no_grad()
def compute_validation_loss(
    *,
    model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    criterion: FlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
) -> float:
    """
    验证函数

    Args:
        model: 模型
        matcher: ConditionalFlowMatcher
        criterion: 损失函数
        loader: 数据加载器
        device: 设备
        epoch: 当前轮数（用于固定随机种子）

    Returns:
        平均验证损失
    """

    model.eval()
    total_loss = 0.0

    # 固定随机种子，使验证集损失更稳定
    if epoch is not None:
        torch.manual_seed(42 + epoch)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + epoch)

    for batch in loader:
        batch = batch.to(device)
        x_1 = batch["ligand_atom"].pos

        t, x_t, v_target = matcher.sample_location_and_target(x_1=x_1, data=batch)

        batch["ligand_atom"].pos = x_t
        predictions = model(batch, t)

        targets = {
            "v_atomic_target": v_target,
            "binding_affinity_target": batch.get("y_energy", None),
        }

        loss_dict = criterion(predictions, targets, batch)
        total_loss += loss_dict["total"].item()

    return total_loss / len(loader)
