# EHFNet: Equivariant Hierarchical Flow Matching for Molecular Docking

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependency Manager: uv](https://img.shields.io/badge/dependency-uv-purple)](https://github.com/astral-sh/uv)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

**EHFNet (Equivariant Hierarchical Flow Network)** 是一个基于**条件流匹配 (Conditional Flow Matching)** 的蛋白质-配体分子对接模型。模型在 $\mathrm{SE}(3) \times \mathbb{T}^m$ 流形上学习从随机初始构象到真实结合态的连续速度场，通过刚体运动（平移 + 旋转）与柔性扭转角的联合优化生成对接构象。

> **Note:** This project is under active development. APIs and model architectures are subject to change.

## Highlights

- **SE(3) × T^m 流形建模**：统一学习刚体平移、旋转与扭转速度场。
- **层次化异构图编码**：原子与残基双尺度交互，兼顾几何细节与全局语义。
- **动态节点预算批采样**：基于 `max_nodes_per_batch` 控制显存上限，训练更稳定。
- **工程化训练策略**：Spatial Curriculum、EMA、梯度累积与裁剪协同提升收敛质量。

## Method

### Flow Matching on SE(3) × T^m

给定配体真实结合态坐标 $x_1$ 与随机初始坐标 $x_0$，定义线性插值路径：

$$x_t = (1 - t)\,x_0 + t\,x_1, \quad t \in [0, 1]$$

模型学习条件速度场 $v_\theta(x_t, t)$，使 ODE 积分 $\dot{x} = v_\theta(x, t)$ 从 $x_0$ 恢复 $x_1$。训练目标由 Kabsch 对齐 + 有限差分从插值路径分解得到，分别对应：

- **平移速度** $v_{\mathrm{trans}} \in \mathbb{R}^3$（质心刚体平移）
- **旋转角速度** $\omega \in \mathbb{R}^3$（Rodrigues 参数化，轴角表示）
- **扭转角速度** $\dot{\tau} \in \mathbb{R}^T$（每个可旋转键一个标量）

推理时支持 Euler 和 RK4 两种 ODE 积分器，默认 100 步。

### Architecture

**分层异构图编码器（4 阶段，每 GNN block 重复）：**

```
ligand atoms  ──[EGNN_Sparse]──→  ligand atoms      (Intra, 等变坐标更新)
ligand atoms  ──[FrameConv]───→  protein residues   (Aggregate, SE(3)-不变)
prot residues ──[FrameConv]───→  protein residues   (Inter-Residue)
prot residues ──[FrameConv]───→  ligand atoms       (Broadcast, SE(3)-不变)
```

**SE(3)-不变消息传递（FrameAwareConv）：**

非 EGNN 阶段的消息公式（严格旋转不变）：

$$m_{i \to j} = \varphi_{\mathrm{msg}}\!\left(h_i,\; h_j,\; \mathrm{RBF}(d_{ij}),\; \hat{R}_j^\top \hat{r}_{ij}\right) \cdot \sigma\!\left(\varphi_{\mathrm{gate}}(\mathrm{RBF}(d_{ij}))\right)$$

其中 $\hat{R}_j \in \mathrm{SO}(3)$ 是以 $j$ 节点邻域均值方向 Gram-Schmidt 正交化构造的局部帧。通过向该局部帧投影，相对位置特征在任意全局旋转 $Q$ 下保持严格不变：

$$(Q\hat{R}_j)^\top (Q\hat{r}_{ij}) = \hat{R}_j^\top Q^\top Q\hat{r}_{ij} = \hat{R}_j^\top \hat{r}_{ij} \quad \checkmark$$

**等变运动读出：**

*平移*：先聚合 EGNN 配体原子速度的质心均值提取方向，再由分子级 MLP 预测幅度：

$$v_{\mathrm{trans}} = \frac{\bar{v}_{\mathrm{CoM}}}{\|\bar{v}_{\mathrm{CoM}}\|} \cdot \mathrm{softplus}(f_s(h_{\mathrm{mol}}))$$

先聚合再归一化（而非逐原子归一化后聚合），保留了原子速度场的力学权重。

*旋转*：体帧角速度 MLP + 主惯量帧投影，规避角动量 $L$ 与角速度 $\omega = I^{-1}L$ 的量纲不匹配：

$$v_{\mathrm{rot}} = R_{\mathrm{frame}} \cdot \frac{\omega_{\mathrm{body}}}{\|\omega_{\mathrm{body}}\|} \cdot \mathrm{softplus}(g_s(h_{\mathrm{mol}}))$$

$R_{\mathrm{frame}} \in \mathrm{SO}(3)$ 由当前坐标 SVD 主轴确定（detach）， $h_{\mathrm{mol}}$ 为 SE(3)-不变量，保证 $v_{\mathrm{rot}}$ 等变。

*扭转*：以旋转键两端原子特征拼接输入，MLP 直接输出标量（旋转不变量），无需坐标投影。

**辅助任务：** 亲和力预测头（ $t > 0.5$ 时激活）+ 位阻惩罚（ $t > 0.8$ 时激活）。

**训练策略：**
- 空间课程学习（Spatial Curriculum）：前 `warmup_epochs` 从局部扰动逐步扩展到全局搜索
- 梯度累积 + 全局梯度裁剪
- EMA 权重平滑（用于验证和推理）

## Installation

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖。

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/Glen-He/EHFNet.git
cd EHFNet

# 安装所有依赖
uv sync
```

## Data Preparation

数据目录结构：

```text
data_root/
├── cleaned/
│   ├── 1a2b/
│   │   ├── 1a2b_ligand.sdf       # 配体（支持 .sdf / .mol2）
│   │   ├── 1a2b_protein.pdb      # 蛋白（建议去水去离子）
│   │   └── 1a2b_esm.npz          # （可选）预计算 ESM Embeddings
│   └── ...
├── index.csv                     # 训练索引
└── processed/                    # 自动生成的预处理缓存
```

`index.csv` 格式：

```csv
pdb_id,affinity
1a2b,6.5
3c4d,7.2
```

### 数据处理流程

```bash
# 1. 从原始 PDBBind 索引提取亲和力标签
uv run python scripts/extract_affinity.py \
    --input  data/raw/pdbbind/hiqbind_info.csv \
    --output data/raw/pdbbind/hiqbind_labels.csv

# 2. 文件完整性校验与过滤
uv run python scripts/validate_and_filter.py \
    --ligand_dir  data/raw/pdbbind/ligand \
    --protein_dir data/raw/pdbbind/protein \
    --input_csv   data/raw/pdbbind/hiqbind_labels.csv \
    --output_csv  data/raw/pdbbind/hiqbind_filtered.csv

# 3. 重组为嵌套目录结构
uv run python scripts/organize_data.py \
    --raw_root    data/raw/pdbbind \
    --target_root data/processed/pdbbind \
    --index_file  data/raw/pdbbind/hiqbind_filtered.csv
```

## Training

### 推荐配置（24GB 单卡）

说明：当前版本采用 **节点预算模式**。`--max_nodes_per_batch` 是显存占用的主控参数，
`--accumulation_steps` 用于在不增加峰值显存的情况下提升等效 batch。

```bash
uv run python train.py \
    --data_root  ./data/processed/pdbbind \
    --index_file ./data/processed/pdbbind/index.csv \
    --epochs 100 \
    --max_nodes_per_batch 20000 \
    --accumulation_steps 8 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --num_gnn_blocks 4 \
    --warmup_epochs 20 \
    --ema_decay 0.999 \
    --rmsd_ratio 0.1 \
    --device cuda:0
```

### 后台运行

```bash
nohup uv run python train.py \
    --data_root  ./data/processed/pdbbind \
    --index_file ./data/processed/pdbbind/index.csv \
    --epochs 100 \
    --max_nodes_per_batch 20000 \
    --accumulation_steps 8 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --num_gnn_blocks 4 \
    --warmup_epochs 20 \
    --ema_decay 0.999 \
    > logs/nohup.log 2>&1 &

tail -f logs/nohup.log
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | — | 数据根目录路径 |
| `--index_file` | str | — | 索引 CSV 文件路径 |
| `--save_dir` | str | `./checkpoints` | 检查点保存目录 |
| `--device` | str | `cuda:0` | 训练设备（`cuda:0`、`cuda:1`、`cpu` 等） |
| `--epochs` | int | 100 | 训练轮数 |
| `--max_nodes_per_batch` | int | 20000 | 动态批采样的单批最大节点数。决定了显存占用的上限。 |
| `--accumulation_steps` | int | 8 | 梯度累积步数，等效扩大 batch size |
| `--lr` | float | 1e-4 | 学习率 |
| `--weight_decay` | float | 1e-6 | 权重衰减 |
| `--clip_grad` | float | 1.0 | 梯度裁剪阈值 |
| `--warmup_epochs` | int | 20 | 空间课程学习预热轮数 |
| `--ema_decay` | float | 0.999 | EMA 衰减系数 |
| `--rmsd_ratio` | float | 0.1 | 验证集中执行 RMSD 推演的样本比例 |
| `--hidden_dim` | int | 128 | 隐藏层维度 |
| `--num_gnn_blocks` | int | 4 | GNN Block 数量（显存不足时可降至 3） |
| `--pocket_radius` | float | 12.0 | 口袋提取截断半径（Å） |
| `--esm_dim` | int | 960 | ESM Embedding 维度（ESMC-300M 为 960） |

## Monitoring

每轮训练结束后自动输出：

- **分项损失**：`loss_trans` / `loss_rot` / `loss_torsion` / `loss_energy` / `loss_clash`
- **验证 RMSD**：Init RMSD → Final RMSD，以及 <2Å 和 <5Å 成功率
- **日志文件**：`logs/train/train_{timestamp}.log`

## Utilities

```bash
# 清理所有预处理缓存（修改数据处理逻辑后使用）
uv run python scripts/clean_cache.py

# 仅清理 ESM 缓存
uv run python scripts/clean_cache.py --skip-processed
```

## License

This project is licensed under the [MIT License](LICENSE).
