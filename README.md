# EHFNet: Equivariant Hierarchical Flow Matching for Molecular Docking

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependency Manager: uv](https://img.shields.io/badge/dependency-uv-purple)](https://github.com/astral-sh/uv)

**EHFNet (Equivariant Hierarchical Flow Network)** 是一个基于**流匹配 (Flow Matching)** 和**等变图神经网络 (EGNN)** 的下一代分子对接模型。

不同于传统的扩散模型 (Diffusion Models)，EHFNet 通过学习从噪声分布到真实结合构象的**最优传输路径 (Optimal Transport Path)**，实现了**确定性、快速且物理一致**的构象生成。

## 🌟 核心特性 (Key Features)

*   **⚡ 流匹配生成 (Flow Matching)**: 直接建模概率流的速度场 (Velocity Field)，生成轨迹更平滑，推理速度比传统扩散模型快 10-50 倍。
*   **⚛️ 物理感知 (Physics-Aware)**:
    *   显式分解为 **平移 (Translation)**、**旋转 (Rotation)** 和 **扭转 (Torsion)** 动力学。
    *   内置物理能量预测头，确保生成的构象符合生物物理约束。
*   **🎓 空间课程学习 (Spatial Curriculum)**: 采用动态难度调度策略，从局部微调逐步过渡到全局搜索，显著加速模型收敛。
*   **🧬 层次化编码**: 融合原子级几何特征与 ESM-2 蛋白质语言模型特征，捕获深层生物学语义。

## 🛠️ 安装 (Installation)

本项目采用现代化的 Python 工具链，**仅支持**使用 [uv](https://github.com/astral-sh/uv) 进行环境管理。

### 前置要求
*   Python >= 3.12
*   CUDA Toolkit 11.8+ (推荐)
*   `uv` 包管理器

### 快速开始

1.  **安装 uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **克隆仓库**:
    ```bash
    git clone https://github.com/your-lab/ehfnet.git
    cd ehfnet
    ```

3.  **一键初始化环境**:
    ```bash
    uv sync
    ```

## 📂 数据准备 (Data Preparation)

请遵循以下目录结构组织您的 PDBBind 数据集：

```text
data_root/                        # [必须] 数据根目录
├── cleaned/                      # [必须] 存放原始结构文件
│   ├── 1a2b/                     # 每个 PDB ID 一个子文件夹
│   │   ├── 1a2b_ligand.sdf       # 配体 (支持 .sdf/.mol2)
│   │   ├── 1a2b_protein.pdb      # 蛋白 (推荐去水去离子)
│   │   └── 1a2b_esm.npz          # (可选) 预计算的 ESM Embeddings
│   └── ...
├── index.csv                     # [必须] 训练索引文件
└── processed/                    # [自动生成] 预处理后的 Tensor 缓存
```

**index.csv 格式示例**:
```csv
pdb_id,affinity
1a2b,6.5
3c4d,7.2
```

## 🛠️ 数据处理流程

本项目提供了完整的工程化数据处理脚本，帮助您从原始数据生成标准数据集。

### 1. 提取亲和力标签
从原始 PDBBind 索引文件中提取 `pdb_id` 和 `affinity`：
```bash
uv run python scripts/extract_affinity.py \
    --input data/raw/pdbbind/hiqbind_info.csv \
    --output data/raw/pdbbind/hiqbind_labels.csv
```

### 2. 验证与清洗
检查文件完整性，过滤缺失文件：
```bash
uv run python scripts/validate_and_filter.py \
    --ligand_dir data/raw/pdbbind/ligand \
    --protein_dir data/raw/pdbbind/protein \
    --input_csv data/raw/pdbbind/hiqbind_labels.csv \
    --output_csv data/raw/pdbbind/hiqbind_filtered.csv
```

### 3. 重组数据结构
将扁平的文件夹结构重组为标准的嵌套结构：
```bash
uv run python scripts/organize_data.py \
    --raw_root data/raw/pdbbind \
    --target_root data/processed/pdbbind \
    --index_file data/raw/pdbbind/hiqbind_filtered.csv
```

## 🚀 训练 (Training)

使用 `train.py` 启动训练。我们提供了丰富的命令行参数以支持灵活配置。

### 推荐命令

```bash
# 单卡训练 (启用课程学习)
uv run python train.py \
    --data_root ./data/processed/pdbbind \
    --index_file ./data/processed/pdbbind/index.csv \
    --batch_size 16 \
    --warmup_epochs 20 \
    --device cuda:0
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **基础配置** | | | |
| `--data_root` | str | 必填 | 数据根目录 |
| `--index_file` | str | 必填 | 索引 CSV 文件路径 |
| `--save_dir` | str | `./checkpoints` | 模型与日志保存路径 |
| `--device` | str | `auto` | 训练设备 (`cuda:0`, `cpu`) |
| **训练超参** | | | |
| `--epochs` | int | 100 | 总训练轮数 |
| `--batch_size` | int | 8 | 批次大小 |
| `--lr` | float | 1e-4 | 学习率 (配合 ReduceLROnPlateau) |
| `--warmup_epochs` | int | 20 | **[重要]** 空间课程学习的预热轮数 |
| **模型结构** | | | |
| `--hidden_dim` | int | 128 | 隐藏层维度 |
| `--num_gnn_blocks` | int | 6 | EGNN 层数 |
| `--esm_dim` | int | 960 | ESM Embedding 维度 |

## 📊 监控与评估 (Monitoring)

训练过程中，控制台和日志文件将实时输出详细指标：

1.  **Loss Components**: 分解为平移、旋转、扭转和能量损失，便于诊断。
2.  **Validation RMSD**: 每个 Epoch 结束时，会自动在验证集上执行全量推演，报告：
    *   **Init RMSD -> Final RMSD**: 评估模型对构象的优化能力。
    *   **Success Rate (<2Å)**: 高精度对接成功率。
    *   **Success Rate (<5Å)**: 中等精度对接成功率。

日志文件默认保存在 `logs/train/train_{timestamp}.log`。

## 🧹 数据清理

如果修改了数据处理逻辑或遇到缓存问题，请使用内置脚本一键清理：

```bash
# 清理所有缓存数据 (processed + ESM cache)
uv run python scripts/clean_cache.py

# 仅清理 ESM 缓存
uv run python scripts/clean_cache.py --skip-processed
```

## 📝 许可证

本项目基于 [MIT License](LICENSE) 开源。使用时请保留版权声明。
