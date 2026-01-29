# EHFNet: Equivariant Hierarchical Flow Matching for Molecular Docking

EHFNet 是一个基于**流匹配 (Flow Matching)** 和**等变图神经网络 (EGNN)** 的深度学习模型，旨在解决分子对接中的构象生成与结合能预测问题。

该项目利用几何深度学习技术，在保证旋转平移等变性的前提下，通过学习从噪声分布到真实结合构象的速度场，实现高效、精准的分子对接预测。

## 🌟 核心特性

- **流匹配生成 (Flow Matching)**: 替代传统的扩散模型，通过学习速度场直接生成分子构象轨迹。
- **物理感知 (Physics-Aware)**: 显式建模平移、旋转和扭转 (Torsion) 动力学，并结合物理能量项进行预测。
- **分层编码**: 结合原子级和分子级特征，支持 ESM 蛋白质语言模型嵌入。

## 🛠️ 安装

本项目**仅支持**使用 [uv](https://github.com/astral-sh/uv) 进行高效的依赖管理。

### 前置要求
- Python >= 3.12
- CUDA (推荐用于加速训练)
- `uv` 包管理器

### 安装步骤

1. **安装 uv** (如果尚未安装):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **克隆仓库**:
   ```bash
   git clone https://github.com/your-username/ehfnet.git
   cd ehfnet
   ```

3. **同步环境与依赖**:
   ```bash
   uv sync
   ```

## 📂 数据准备

训练数据需存放在 `data_root` 下的 `cleaned` 文件夹中。

**目录结构要求**：
```text
data_root/
├── cleaned/                      # 必须命名为 cleaned
│   ├── 1a2b/                     # 每个 PDB ID 一个文件夹
│   │   ├── 1a2b_ligand.sdf       # 配体文件 (支持 .sdf 或 .mol2)
│   │   ├── 1a2b_protein.pdb      # 蛋白文件
│   │   └── 1a2b_esm.npz          # (可选) 预计算的 ESM embedding
│   ├── 3c4d/
│   │   ├── ...
│   └── ...
├── index.csv                     # 索引文件 (建议放在 data_root 下)
└── processed/                    # 程序会自动生成此目录用于缓存
```

### 索引文件 (Index File)
您需要提供一个索引文件来指定训练样本。仅支持 **CSV 格式**：
必须包含 `pdb_id` 和 `affinity` 两列。

```csv
pdb_id,affinity
1a2b,6.5
3c4d,7.2
```

## 🚀 训练

使用 `train.py` 脚本启动训练。

### 基本用法

```bash
uv run python train.py \
    --data_root /path/to/data \
    --index_file /path/to/data/index.csv \
    --save_dir ./checkpoints
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | 必填 | 数据集根目录 (需包含 `cleaned` 子文件夹) |
| `--index_file` | 必填 | 索引文件路径 (CSV格式) |
| `--save_dir` | `./checkpoints` | 模型检查点和日志保存目录 |
| `--batch_size` | 8 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--esm_path` | None | (可选) 全局 ESM embedding 存储路径 |
| `--esm_dim` | 960 | ESM embedding 维度 (默认为 960) |
| `--num_gnn_blocks` | 6 | GNN 层数 |
| `--device` | `auto` | 指定训练设备 (如 `cuda:0`, `cuda:1`, `cpu`) |

### 示例：使用 ESM 特征训练

```bash
uv run python train.py \
    --data_root ./data \
    --index_file ./data/index.csv \
    --esm_dim 960 \
    --batch_size 16 \
    --device cuda:0
```

## 📊 输出与日志

- **日志**: 训练日志将同时输出到终端和 `logs/train/train_{timestamp}.log`。
- **模型**: 验证集损失最低的模型将保存为 `save_dir/best_model.pt`。
- **监控**: 进度条会实时显示 Loss 及其分量 (平移、旋转、扭转、能量)。

## 📝 许可证

[MIT License](LICENSE)
