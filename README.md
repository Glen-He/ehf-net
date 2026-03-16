# EHFNet: Equivariant Hierarchical Flow Matching for Molecular Docking

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependency Manager: uv](https://img.shields.io/badge/dependency-uv-purple)](https://github.com/astral-sh/uv)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

**EHFNet (Equivariant Hierarchical Flow Network)** 是一个基于**条件流匹配 (Conditional Flow Matching)** 的蛋白质-配体分子对接模型。模型在 $\mathrm{SE}(3) \times \mathbb{T}^m$ 流形上学习从随机初始构象到真实结合态的连续速度场，通过刚体运动（平移 + 旋转）与柔性扭转角的联合优化生成对接构象。


## Highlights

- **SE(3) × T^m 流形建模**：统一学习刚体平移、旋转与扭转速度场。
- **层次化异构图编码**：原子与残基双尺度交互，兼顾几何细节与全局语义。
- **成本感知批采样**：基于样本级节点/边/扭转画像打包 batch，并使用静态成本预算驱动吞吐。
- **工程化训练策略**：Spatial Curriculum、EMA、梯度累积与裁剪协同提升收敛质量。
- **稳健显存控制**：统一采用高静态预算、真实 OOM 后收缩与 batch 二分重试，降低长尾样本带来的 OOM 波动。

## Method

### Flow Matching on SE(3) × T^m

给定配体真实结合态坐标 $x_1$ 与随机初始坐标 $x_0$，定义线性插值路径：

$$x_t = (1 - t)\,x_0 + t\,x_1, \quad t \in [0, 1]$$

模型学习条件速度场 $v_\theta(x_t, t)$，使 ODE 积分 $\dot{x} = v_\theta(x, t)$ 从 $x_0$ 恢复 $x_1$。训练目标由 Kabsch 对齐 + 有限差分从插值路径分解得到，分别对应：

- **平移速度** $v_{\mathrm{trans}} \in \mathbb{R}^3$（质心刚体平移）
- **旋转角速度** $\omega \in \mathbb{R}^3$（Rodrigues 参数化，轴角表示）
- **扭转角速度** $\dot{\tau} \in \mathbb{R}^T$（每个可旋转键一个标量）

推理时支持 Euler 和 RK4 两种 ODE 积分器，默认 50 步。

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

*平移*：Hybrid Fusion。门控网络自适应融合两路信号：
1. **物理先验**：EGNN 配体原子速度的质心均值。
2. **体帧预测**：MLP 在主惯量帧内预测平移方向，并经 $R_{\mathrm{frame}}$ 投影至世界帧保障等变性。

*旋转*：体帧角速度 MLP + 主惯量帧投影，规避角动量 $L$ 与角速度 $\omega = I^{-1}L$ 的量纲不匹配：

$$v_{\mathrm{rot}} = R_{\mathrm{frame}} \cdot \frac{\omega_{\mathrm{body}}}{\|\omega_{\mathrm{body}}\|} \cdot \mathrm{softplus}(g_s(h_{\mathrm{mol}}))$$

$R_{\mathrm{frame}} \in \mathrm{SO}(3)$ 由当前坐标 SVD 主轴确定（detach）， $h_{\mathrm{mol}}$ 为 SE(3)-不变量，保证 $v_{\mathrm{rot}}$ 等变。

*扭转*：以旋转键两端原子特征拼接输入，MLP 直接输出标量（旋转不变量），无需坐标投影。

**辅助任务：**
- 亲和力预测头与位阻惩罚共享同一套时间门控，由训练进度与当前时间步 `t` 共同调节。
- 训练时，亲和力损失、位阻损失和构象质量损失都通过平滑时间门控逐步打开，而不是使用固定的硬阈值。
- 验证时，亲和力误差统计仅在 `t > 0.8` 的样本上汇总，以聚焦接近终态区域的预测质量。

**训练策略：**
- 空间课程学习（Spatial Curriculum）：前 `warmup_epochs` 从局部扰动逐步扩展到全局搜索
- 严格样本级梯度累积（Sample-level Accumulation）
- 成本感知批采样：按样本级成本画像构建 batch
- 静态预算主导：默认尽量吃满预算，只在真实 OOM 后收缩
- OOM 二分重试：成本超限或触发 OOM 时优先拆分 batch，而不是直接整批跳过
- 自适应成本预算：OOM 频发时自动降低阶段预算，连续稳定后可逐步回升
- CUDA 显存碎片率控制（内置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）
- EMA 权重平滑（用于验证和推理）
- 综合指标体系：支持输出 Median RMSD、Centroid Distance、Pearson/Spearman 相关性指标

### Ranking Strategy

- **默认最终重排**：最终 pose 排序默认采用单一 `pose_rank_score` 与 `center` 置信分支的线性融合；`affinity` 与 `clash` 作为弱辅助信号默认参与重排，权重分别为 `0.08` 与 `0.12`。
- **两阶段选择**：stage-1 先按每个中心下的最佳 pose 分数筛选 refinement centers；stage-2 再对所有候选 pose 执行统一重排并统计 Top-N 指标。
- **默认主排序头**：`pose_rank_score` 是唯一的 pose 质量/排序输出，同时承担质量校准与最终排序职责。
- **设计动机**：`pose` 分支负责同一复合物内部的几何排序，`center` 分支提供 proposal 先验；二者分工明确，避免将复合物级亲和力标签直接当作 pose 级排序标签。

### Auxiliary Supervision

- **RMSD-first 主链路**：默认先强化 `translation / rotation / torsion` 几何主损失，再在中后期逐步引入 ranking、bootstrap 和 blind-pool replay；训练期排序软标签与验证指标统一基于 symmetry-aware RMSD。
- **单一 `pose_rank_score` 头**：仍然统一承担构象质量估计与最终排序，但默认只在几何主链基本收敛后再逐步放大监督强度。
- **亲和力与位阻分支**：二者共享同一套平滑时间门控，仅在训练后期和较大 `t` 区域逐步增强权重；验证时亲和力误差只在 `t > 0.8` 的样本上统计。
- **blind-pool replay**：定位为后置 reranker 增强阶段；除 pose 排序损失外，还包含 center-value supervision，用于把 proposal 质量与后续 pose 成功率重新耦合。

### Validation Protocol

- **日常监控**：默认对验证集做 `30%` partial lightweight validation；验证、bootstrap 与 blind 推理统一使用 `ode_method`，默认值为 `euler`。
- **周期全量验证**：每隔 `val_full_every` 个 epoch 执行一次 `100%` full lightweight validation。
- **尾段高覆盖验证**：最后 `val_full_last_epochs` 个 epoch 始终执行 `100%` full lightweight validation，但积分方法仍遵循同一个 `ode_method` 配置。
- **checkpoint 选择**：训练中所有 checkpoint 选择均基于 lightweight validation，默认使用 `rmsd_priority` 规则，优先考虑 `Success@2A / Success@5A / Mean RMSD`；完整 blind Top-N 仅在训练结束后运行。
- **子集抽样方式**：partial validation 每轮按固定随机种子重采样确定性子集，用于兼顾覆盖率与反馈速度。

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
│   │   └── 1a2b_esm_chainseg_esmc_300m.npz # （可选）预计算 ESM Embeddings
│   └── ...
├── index.csv                     # 训练索引
└── processed/                    # 自动生成的预处理缓存
```

原始目录中的 **`index.csv`** 表头必须为：

```csv
Concatenated ID,Log Binding Affinity
1a2b,6.5
3c4d,7.2
```

processed 下会生成供训练使用的 `index.csv`。

### 数据处理流程

**约定：** 把原始数据放在 `data/raw/` 下，按**数据集名称**建一个文件夹（如 `hiqbind`、`pdbbind`）。该文件夹内**必须**包含：

- 目录 `ligand/`（配体文件，如 `{id}_ligand.sdf` 或 `.mol2`）
- 目录 `protein/`（蛋白文件，如 `{id}_protein.pdb`）
- 文件 **`index.csv`**（固定文件名；表头必须为 **`Concatenated ID`** 与 **`Log Binding Affinity`**）

运行脚本时传入数据集名称（与上述文件夹名一致）。脚本**不删除、不修改 raw 下任何内容**，仅在 `data/processed/<数据集名>/` 下生成 `cleaned/<id>/` 与 `index.csv`。后续预处理与训练均使用 **processed** 目录。

**示例目录：**

```text
data/raw/hiqbind/
├── ligand/
│   ├── 1a2b_ligand.sdf
│   └── ...
├── protein/
│   ├── 1a2b_protein.pdb
│   └── ...
└── index.csv
```

**命令示例：**

```bash
uv run python scripts/prepare_data.py hiqbind

# 使用 pdbbind
uv run python scripts/prepare_data.py pdbbind

# 仅执行指定步骤（当前支持 organize）
uv run python scripts/prepare_data.py hiqbind --steps organize
```

步骤含义：脚本**不删除、不修改 raw 下任何内容**，仅读取并生成新文件到 processed。`organize` 从 raw 读 `index.csv`，将「配体/蛋白文件都存在且亲和力有效」的条目复制到 `data/processed/<dataset>/cleaned/<id>/`（目标文件名与 SDF 第一行为小写），并写出仅含这些条目的 `index.csv`；缺文件或无效亲和力的行跳过并打日志。后续预处理与训练均使用 **processed** 目录。

### 预处理（图缓存 + ESM）

所有预处理命令都**必须**指定 `--data-root`（数据根目录，与 prepare_data 输出一致，目录内须含 `index.csv`），例如：`data/processed/hiqbind`。

预处理会缓存完整蛋白图（包含 ESM embedding）；局部上下文的定位与裁剪在训练/推理阶段通过 runtime crop 完成。`build` 默认读取 `configs/train.toml` 里的 `device`，也可用 `--device` 临时覆盖成其他 GPU 或 `cpu`。`esm`、`esm_model_name` 与 `esm_dim` 也由配置显式提供，其中 `esm` 支持 `auto`、`file`、`off` 三种模式；ESM 缓存文件名会包含模型名，例如 `1a2b_esm_chainseg_esmc_300m.npz`，避免更换模型后误复用旧缓存。默认情况下，三个子命令都会将文本日志写到 `logs/preprocess/`，文件名形如 `preprocess_build_2026-03-15_22-30-45.log`；若传入 `--smoke`，则改为写到 `logs/smoke/preprocess/`。几何筛选阈值以及统计推荐 cutoff 的分位数、缩放系数和上下界由 `configs/train.toml` 中的 `[preprocess]` 段统一配置。

子命令：

- **`build`**：构建图缓存（HeteroData，含 ESM embedding）并做几何检查，写入 `<data-root>/cache/`。
- **`stats`**：基于图缓存计算特征统计和距离分布，输出 `dataset_profile.json`。
- **`clean`**：按 `--target` 清理缓存（`graph` / `esm` / `all`），支持 `--dry-run` 预览。

`build` 会优先复用已有的 HuggingFace 模型缓存；若未显式设置 `HF_HOME` 或 `HF_HUB_CACHE`，则自动回退到项目根目录下的 `.hf-cache/`，避免因系统缓存目录不可写而重复下载模型。

**首次运行：**

```bash
uv run python scripts/preprocess.py build --data-root data/processed/hiqbind
uv run python scripts/preprocess.py stats --data-root data/processed/hiqbind

# 显式指定 ESM 缓存目录并强制重建
uv run python scripts/preprocess.py build \
    --data-root data/processed/hiqbind \
    --esm-root data/esm-cache \
    --force-rebuild

# 指定统计输出文件并限制样本数
uv run python scripts/preprocess.py stats \
    --data-root data/processed/hiqbind \
    --output data/processed/hiqbind/dataset_profile.json \
    --max-samples 200
```

**smoke 运行：**

```bash
uv run python scripts/preprocess.py --smoke build --data-root data/processed/hiqbind_smoke200
uv run python scripts/preprocess.py --smoke stats --data-root data/processed/hiqbind_smoke200

# 关闭 smoke 分组并写回默认日志目录
uv run python scripts/preprocess.py --no-smoke stats --data-root data/processed/hiqbind_smoke200
```

**非首次运行**需根据修改的参数判断要清理什么缓存：

| 修改了什么 | 需要清理 | 清理命令 |
|-----------|---------|---------|
| 蛋白/配体文件或 `index.csv` | 图缓存 + ESM 缓存 | `--target all` |
| ESM 相关参数（`esm_dim`、ESM 模型等） | 图缓存 + ESM 缓存（ESM embedding 嵌入在图缓存中） | `--target all` |
| 图拓扑参数（`r_cutoff_intra`、`max_neighbors_intra` 等） | 图缓存 | `--target graph` |
| 特征编码参数（atom/residue feature schema） | 图缓存 | `--target graph` |

```bash
# 清理后重建
uv run python scripts/preprocess.py clean --data-root data/processed/hiqbind --target all
uv run python scripts/preprocess.py build --data-root data/processed/hiqbind
uv run python scripts/preprocess.py stats --data-root data/processed/hiqbind

# 预览将删除的文件（不实际删除）
uv run python scripts/preprocess.py clean --data-root data/processed/hiqbind --target all --dry-run
```

## Training

### 推荐配置（24GB 单卡）

采用 **成本感知批处理模式**：训练启动时会为每个样本缓存节点/边/扭转画像，动态估计跨图边成本，并据此打包 batch。主链路使用高静态预算；只有在真实 OOM 或不可分超预算样本出现时，才会触发 batch 二分重试与阶段预算收缩。
`--accumulation_steps` 用于在不增加峰值显存的情况下提升等效 batch。
训练入口会在导入 `torch` 前读取 `configs/train.toml` 的 `[runtime].torch_cuda_alloc_conf`，并在未显式设置环境变量时注入 `PYTORCH_CUDA_ALLOC_CONF`。
若传入 `--smoke`（或在 `configs/train.toml` 的 `[logging]` 中设置 `smoke = true`），训练正式日志会写到 `logs/smoke/train/`；默认仍写到 `logs/train/`。

训练/推理主流程为 proposal-guided local docking：
1. residue-level center proposal
2. 10Å local crop
3. local flow docking
4. pose confidence reranking

仅支持 **rigid-protein**：蛋白坐标在训练与推理期间保持静态。图中的局部摘要节点命名为 `protein_context`，缓存使用语义化标识 `graph_cache_context_rigid` / `ehfnet_feature_signature_context_rigid` / `blind_pool_context_rigid`。

**请务必指定数据目录**：训练不会使用任何默认数据路径。方式二选一即可：

- 在 **`configs/train.toml`** 的 `[data]` 里设置 `data_root = "data/processed/你的数据集名"`（该文件为本地自定义配置，可按自己当前文件夹填写）；
- 或命令行传入：`--data_root data/processed/hiqbind`。

**规范**：只接受**一个文件夹**（预处理输出目录，内含 `cleaned/` 与 **`index.csv`**）。不允许自定义 index 路径，程序固定使用该目录下的 `index.csv`。

```bash
uv run python train.py --config configs/train.toml
# 或直接指定数据目录
uv run python train.py --data_root data/processed/hiqbind
```

```bash
# smoke 训练：将文本日志集中写到 logs/smoke/
uv run python train.py \
    --data_root data/processed/hiqbind_smoke200 \
    --smoke
```

常改训练参数已整理到 `configs/train.toml`，模型/图拓扑/flow-matching/loss 相关参数已整理到 `configs/model.toml`；这些模型内部参数需要在配置中显式给出，不再依赖代码层隐式兜底。命令行参数会覆盖配置文件。

训练阶段默认采用“分层轻量验证 + 最终完整测试”拆分：

- 默认每个 epoch 只在验证集上抽取 `30%` 子集做 partial 轻量验证；
- 每隔 `10` 个 epoch 会对验证集执行一次 `100%` full 轻量验证；
- 最后 `10` 个 epoch 会始终执行 `100%` full 轻量验证；
- 验证、bootstrap 与 blind 推理统一使用单一 `ode_method` 配置，默认是 `euler`；
- 轻量验证默认汇总的是 symmetry-aware single-shot RMSD、single-shot success 和少量亲和力指标；
- blind pool 刷新与最终 blind Top-N 测试使用独立成本预算，日志也会分别报告；
- 完整 blind Top-N 评测只在训练结束后的最终测试阶段执行一次。

```bash
uv run python train.py \
    --data_root data/processed/hiqbind \
    --epochs 100 \
    --train_cost_budget 1300000 \
    --val_cost_budget 1300000 \
    --blind_pool_cost_budget 1600000 \
    --final_topn_cost_budget 1600000 \
    --accumulation_steps 8 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --num_gnn_blocks 4 \
    --crop_radius 10 \
    --warmup_epochs 20 \
    --ema_decay 0.999 \
    --val_subset_ratio 0.3 \
    --val_full_every 10 \
    --val_full_last_epochs 10 \
    --ode_method euler \
    --split_train_frac 0.7 \
    --split_val_frac 0.1 \
    --split_test_frac 0.2 \
    --split_seed 42 \
    --ablation_mode none \
    --run_test_after_training \
    --test_topk 1,5,10 \
    --center_proposal_topk 8 \
    --center_refine_topk 3 \
    --stage1_pose_samples 2 \
    --stage2_pose_samples 4 \
    --enable_train_budget_callback \
    --oom_reduce_threshold 3 \
    --oom_reduce_factor 0.85 \
    --min_train_cost_budget 480000 \
    --enable_val_budget_callback \
    --val_oom_reduce_threshold 3 \
    --val_oom_reduce_factor 0.85 \
    --min_val_cost_budget 120000 \
    --train_budget_window_size 8 \
    --train_budget_recover_window_count 2 \
    --train_budget_recover_step 80000 \
    --train_offender_cooldown 6 \
    --val_budget_window_size 4 \
    --val_budget_recover_window_count 2 \
    --val_budget_recover_step 80000 \
    --val_offender_cooldown 4 \
    --device cuda:0
```

### 后台运行

```bash
run_suffix="$(date '+%Y-%m-%d_%H-%M-%S')"
mkdir -p logs/smoke/nohup
nohup uv run python train.py \
    --data_root data/processed/hiqbind \
    --smoke \
    --epochs 100 \
    --train_cost_budget 1300000 \
    --val_cost_budget 1300000 \
    --blind_pool_cost_budget 1600000 \
    --final_topn_cost_budget 1600000 \
    --accumulation_steps 8 \
    --lr 1e-4 \
    --hidden_dim 128 \
    --num_gnn_blocks 4 \
    --crop_radius 10 \
    --warmup_epochs 20 \
    --ema_decay 0.999 \
    --val_subset_ratio 0.3 \
    --val_full_every 10 \
    --val_full_last_epochs 10 \
    --ode_method euler \
    --enable_train_budget_callback \
    --oom_reduce_threshold 3 \
    --oom_reduce_factor 0.85 \
    --min_train_cost_budget 480000 \
    --enable_val_budget_callback \
    --val_oom_reduce_threshold 3 \
    --val_oom_reduce_factor 0.85 \
    --min_val_cost_budget 120000 \
    --train_budget_window_size 8 \
    --train_budget_recover_window_count 2 \
    --train_budget_recover_step 80000 \
    --train_offender_cooldown 6 \
    --val_budget_window_size 4 \
    --val_budget_recover_window_count 2 \
    --val_budget_recover_step 80000 \
    --val_offender_cooldown 4 \
    --run_suffix "${run_suffix}" \
    > "logs/smoke/nohup/nohup_train_${run_suffix}.log" 2>&1 &

tail -f "logs/smoke/nohup/nohup_train_${run_suffix}.log"
```

同一次训练会生成两份文本日志。默认目录如下：

- `logs/train/train_<run_suffix>.log`：Python 侧正式训练日志。
- `logs/nohup/nohup_train_<run_suffix>.log`：shell 级后台输出，通常更完整。

若使用 `--smoke`，则对应改为：

- `logs/smoke/train/train_<run_suffix>.log`
- `logs/smoke/nohup/nohup_train_<run_suffix>.log`

训练产物仍保存在 `checkpoints/train_<run_suffix>/`。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | 见下 | **必填**（config 或 CLI）。数据根目录（一个文件夹），**必须**内含 `index.csv`，如 `data/processed/hiqbind`。不允许自定义 index 路径。 |
| `--save_dir` | str | `./checkpoints` | 运行产物根目录。每次训练会自动创建与日志同名的子目录，如 `checkpoints/train_2026-03-07_15-30-00/`。 |
| `--device` | str | `configs/train.toml` 中的 `device` | 训练设备（`cuda:0`、`cuda:1`、`cpu` 等） |
| `--run_suffix` | str | 自动生成 | 可选运行后缀；用于让训练正式日志、`nohup` 日志和产物目录共享同一次运行标识。 |
| `--smoke` | flag | `false` | 将文本日志重定向到 `logs/smoke/...`，便于集中清理 smoke 运行日志；不影响 checkpoint 目录。 |
| `--epochs` | int | 100 | 训练轮数 |
| `--train_cost_budget` | int | 1300000 | 训练阶段的静态成本预算；主链路默认尽量吃满预算，只在真实 OOM 后收缩。 |
| `--val_cost_budget` | int | `1300000` | 训练期轻量验证使用的静态成本预算。 |
| `--blind_pool_cost_budget` | int | `1600000` | blind pool 刷新阶段的静态成本预算。 |
| `--final_topn_cost_budget` | int | `1600000` | 最终 blind Top-N 评估阶段的静态成本预算。 |
| `--accumulation_steps` | int | 8 | 梯度累积步数，等效扩大 batch size |
| `--lr` | float | 1e-4 | 学习率 |
| `--weight_decay` | float | 1e-6 | 权重衰减 |
| `--clip_grad` | float | 1.0 | 梯度裁剪阈值 |
| `--warmup_epochs` | int | 20 | 空间课程学习预热轮数 |
| `--ema_decay` | float | 0.999 | EMA 衰减系数 |
| `--val_subset_ratio` | float | 0.3 | 默认 partial 轻量验证覆盖的验证集比例 |
| `--val_full_every` | int | 10 | 每隔多少个 epoch 执行一次 `100%` full 轻量验证，`0` 表示关闭周期全量验证 |
| `--val_full_last_epochs` | int | 10 | 训练末尾连续执行 `100%` full 轻量验证的 epoch 数 |
| `--ode_method` | str | `euler` | 验证、bootstrap 与 blind 推理统一使用的 ODE 求解器，可选 `euler` / `rk4` |

### Compute Budget Semantics

- `cost_budget` 不是样本数或图数，而是由节点数、静态/动态图边数与扭转项共同估计得到的计算成本单位。
- `train_cost_budget` 对应训练主循环，服务于前向、反向与梯度累积的吞吐需求。
- `val_cost_budget` 对应训练期验证；`partial` 与 `full` 共用同一套静态预算基线，但会保留各自的 OOM 收缩状态。
- `blind_pool_cost_budget` 只对应 blind pool 刷新；`final_topn_cost_budget` 只对应最终 blind Top-N 推理，两者互不拖累。

### RMSD-First Curriculum

- 配体初始位姿固定采用 `rdkit_decoupled`：从 ligand 文件读取分子拓扑后，由 RDKit 重新嵌入三维构象，再进入后续的平移、旋转与扭转随机化流程；不再保留任何真值构象直通或 `reference_pose` 对照分支。
- 默认课程会先用 `gt / jitter / proposal_pos` 建立几何主链，再逐步引入 `near_miss` 与 `hard_neg`。
- `same-center` ranking、`wrong-center` ranking、bootstrap 与 blind-pool replay 都通过显式进度阈值后置开启，不再依赖 “blind pool 一出现就整套切换” 的隐式开关。
- 训练期 pose soft target、lightweight validation 与最终评估统一使用 symmetry-aware RMSD 作为 RMSD 口径，不再保留 direct RMSD 分支。
- 在线排序也拆成独立控制：`same-center` 由微批大小控制显存峰值，`wrong-center` 继续用独立预算回调控制额外排序分支。
- 不同阶段还会叠加各自的 `phase_multiplier` 与 batch 二分重试，因此相同数值预算在不同阶段并不代表相同的实际 batch 规模。
| `--max_oom_retry_splits` | int | 3 | batch 触发 OOM 或预算超限后允许递归二分重试的最大深度 |
| `--hidden_dim` | int | 128 | 隐藏层维度 |
| `--num_gnn_blocks` | int | 4 | GNN Block 数量（显存不足时可降至 3） |
| `--crop_radius` | float | 10.0 | 运行时 local crop 半径（Å） |
| `--esm` | str | `auto` | ESM 处理模式：`auto`=缺缓存时自动计算，`file`=仅读取缓存，`off`=完全关闭 ESM |
| `--esm_model_name` | str | `esmc_300m` | ESM 主干模型名；修改后需重建图缓存与 ESM 缓存 |
| `--esm_dim` | int | 960 | ESM Embedding 维度（ESMC-300M 为 960） |
| `--dynamic_inter_max_neighbors` | int | 48 | 动态 `ligand_atom ↔ protein_atom` 边的单源邻居上限 |
| `--dynamic_residue_max_neighbors` | int | 32 | 动态 `ligand_atom ↔ protein_residue` 边的单源邻居上限 |
| `--dynamic_residue_candidate_topk` | int | 96 | 每个复合物在构建动态配体-残基边前保留的候选残基上限 |
| `--split_train_frac` | float | 0.7 | 训练集比例（Scaffold Split） |
| `--split_val_frac` | float | 0.1 | 验证集比例（Scaffold Split） |
| `--split_test_frac` | float | 0.2 | 独立测试集比例（Scaffold Split） |
| `--split_seed` | int | 42 | 划分随机种子 |
| `--split_cache_file` | str | `cache/splits/hiqbind_scaffold_split.json` | 划分索引 JSON 路径；用于严格复现 |
| `--force_resplit` | flag | 关闭 | 强制重建划分索引 |
| `--ablation_mode` | str | `none` | 消融模式：`none` / `inter_multiscale_off` |
| `--run_test_after_training` | flag | 启用 | 训练后自动运行独立 test 评估 |
| `--test_topk` | str | `1,5,10` | Top-N 成功率统计阈值列表 |
| `--center_proposal_weight` | float | 0.15 | blind-pool replay 中 center value supervision 的损失权重 |
| `--center_positive_radius` | float | 4.0 | crop curriculum 与 blind center 指标使用的命中半径（Å） |
| `--center_guidance_learned_start` | float | 0.35 | 中心打分从几何先验切换到学习分数的训练进度阈值 |
| `--center_proposal_topk` | int | 8 | stage-1 保留的候选中心数量 |
| `--center_refine_topk` | int | 3 | stage-2 深化对接的中心数量 |
| `--center_nms_radius` | float | 6.0 | 候选中心去冗余半径（Å） |
| `--stage1_pose_samples` | int | 2 | 每个中心在 stage-1 的局部采样数 |
| `--stage2_pose_samples` | int | 4 | 每个中心在 stage-2 的局部采样数 |
| `--crop_proposal_start` | float | 0.10 | `proposal_pos` 进入中心课程的训练进度阈值 |
| `--crop_near_miss_start` | float | 0.35 | `near_miss` 进入中心课程的训练进度阈值 |
| `--crop_hard_negative_start` | float | 0.65 | `hard_neg` 进入中心课程的训练进度阈值 |
| `--crop_min_residues` | int | 12 | 运行时 local crop 至少保留的蛋白残基数 |
| `--crop_atom_margin` | float | 2.0 | 运行时 local crop 中原子距离裁剪的额外缓冲半径（Å） |
| `--pose_ranking_pair_weight` | float | 0.1 | 单一 `pose_rank_score` 头的 pairwise ranking 损失权重 |
| `--pose_ranking_margin` | float | 0.5 | pairwise ranking margin |
| `--ranking_same_center_start` | float | 0.55 | `same-center` pairwise ranking 启用的训练进度阈值 |
| `--ranking_wrong_center_start` | float | 0.75 | `wrong-center` pairwise ranking 启用的训练进度阈值 |
| `--pose_bootstrap_weight` | float | 0.05 | model-generated pose bootstrap 损失权重 |
| `--pose_bootstrap_start` | float | 0.80 | bootstrap ranking 监督启用的训练进度阈值 |
| `--pose_bootstrap_frequency` | int | 25 | 每 N 个训练 batch 执行一次 bootstrap pose 打分（0 表示关闭） |
| `--pose_bootstrap_ode_steps` | int | 10 | bootstrap pose 生成使用的 ODE 步数 |
| `--blind_pool_refresh_on_best_update` | flag | 关闭 | 新最佳 checkpoint 出现时是否立刻额外刷新一次 blind pool；默认关闭，避免 pool 刷新反过来主导训练节奏 |
| `--blind_pool_pairs_per_complex` | int | 4 | 每个复合物从 blind pool 回放时采样的困难配对基数 |
| `--replay_start_ratio` | float | 0.85 | blind-pool replay 作为后置增强阶段启用的训练进度阈值 |
| `--val_ode_steps` | int | 50 | 训练期轻量验证与最终 blind 测试共用的 ODE 积分步数 |
| `--crop_candidate_topk` | int | 8 | crop curriculum 在各 proposal bucket 内做 weighted sampling 时使用的 top-k 池大小 |
| `--disable_jitter_crop` | flag | 关闭 | 关闭 jitter crop，用于 ablation |
| `--disable_hard_negative_crop` | flag | 关闭 | 关闭 hard-negative crop，用于 ablation |
| `--checkpoint_selection_mode` | str | `rmsd_priority` | 训练期轻量验证的 checkpoint 选择主指标：`rmsd_priority` / `composite` / `mean_rmsd` / `val_loss` / `single_shot_success_2a` / `single_shot_success_5a` |
| `--dataloader_num_workers` | int | 4 | DataLoader worker 数 |
| `--dataloader_pin_memory / --no_dataloader_pin_memory` | flag | 启用 | 是否启用 pinned memory |
| `--dataloader_persistent_workers / --no_dataloader_persistent_workers` | flag | 启用 | 是否启用 persistent workers |
| `--enable_train_budget_callback / --disable_train_budget_callback` | flag | 启用 | 是否启用训练阶段的窗口式预算回调 |
| `--oom_reduce_threshold` | int | 3 | 单个训练窗口触发多少次根 OOM 事件后降低训练成本预算 |
| `--oom_reduce_factor` | float | 0.85 | OOM 后的训练预算缩放比例（0~1） |
| `--min_train_cost_budget` | int | 480000 | 训练阶段预算回调允许降到的最小预算 |
| `--enable_val_budget_callback / --disable_val_budget_callback` | flag | 启用 | 是否启用验证阶段的窗口式预算回调 |
| `--val_oom_reduce_threshold` | int | 3 | 单个验证窗口 OOM 达阈值后降低验证成本预算 |
| `--val_oom_reduce_factor` | float | 0.85 | 验证降批比例（0~1） |
| `--min_val_cost_budget` | int | `120000` | 验证阶段预算回调允许降到的最小预算 |
| `--train_budget_window_size` | int | 8 | 训练预算回调使用的根 batch 窗口大小 |
| `--train_budget_recover_window_count` | int | 2 | 连续多少个干净训练窗口后加性回升预算 |
| `--train_budget_recover_step` | int | 80000 | 每次训练预算回升的加性步长 |
| `--train_offender_cooldown` | int | 6 | 训练坏样本的冷却时长，以根 batch 事件计 |
| `--val_budget_window_size` | int | 4 | 验证预算回调使用的窗口大小 |
| `--val_budget_recover_window_count` | int | 2 | 连续多少个干净验证窗口后加性回升预算 |
| `--val_budget_recover_step` | int | 80000 | 每次验证预算回升的加性步长 |
| `--val_offender_cooldown` | int | 4 | 验证坏样本的冷却时长 |
| `--same_center_micro_batch_size` | int | 2 | online same-center ranking 的初始微批大小；该分支不再整批二次前向 |
| `--same_center_budget_window_size` | int | 8 | same-center ranking 微批恢复窗口大小 |
| `--same_center_budget_recover_window_count` | int | 2 | same-center ranking 回升所需的连续干净窗口数 |
| `--same_center_budget_recover_step` | int | 1 | same-center ranking 每次回升的微批步长 |
| `--same_center_offender_cooldown` | int | 6 | same-center ranking 坏样本冷却时长 |
| `--ranking_budget_window_size` | int | 8 | wrong-center ranking 分支的恢复窗口大小 |
| `--ranking_budget_recover_window_count` | int | 2 | wrong-center ranking 回升所需的连续干净窗口数 |
| `--ranking_offender_cooldown` | int | 6 | wrong-center ranking 坏样本冷却时长 |
| `--ranking_wrong_center_cap` | int | 1 | wrong-center ranking 最大启用级别，`0` 表示退化为 same-center-only |
| `--replay_micro_batch_size` | int | 4 | replay 候选打分的初始微批大小 |
| `--replay_budget_window_size` | int | 8 | replay 微批恢复窗口大小 |
| `--replay_budget_recover_window_count` | int | 2 | replay 微批回升所需的连续干净窗口数 |
| `--replay_candidate_cooldown` | int | 6 | replay 复杂样本冷却时长 |
| `--replay_max_candidates_per_complex` | int | 8 | replay 每个复合物保留的最大候选数 |

### OOM 说明（实践建议）

- 训练、验证和 Top-N 都改为 **成本感知批处理**，不再只按静态节点数估计显存。
- 编码器会对动态 `ligand_atom ↔ protein_residue` 边先做候选残基预筛，再应用半径/kNN 稀疏连接，显著收缩长尾样本成本。
- 训练、验证、blind pool 刷新和最终 Top-N 都支持 batch 二分重试；若成本超限或触发 OOM，会优先拆成更小子批次而不是整批放弃。
- **验证预算回调**：`partial` 与 `full` validation 各自维护独立的预算恢复状态，不影响训练预算，也不会互相拖累。
- **same-center ranking 微批**：online same-center 排序不再复制整 batch 做第二次前向，而是按图切成独立微批；若微批 OOM，会递归二分直到单图，再由窗口式回调降低后续微批上限。
- 若显存紧张，优先降低 `--train_cost_budget`，其次降低 `--num_gnn_blocks`。
- 在共享 GPU 场景，其他进程会挤占显存，建议训练前确认空闲显存。

## Monitoring

每轮训练结束后自动输出训练期轻量验证结果：

- **分项损失**：`loss_trans` / `loss_rot` / `loss_torsion` / `loss_energy` / `loss_clash`
- **轻量验证指标**：`val_loss`、`mean_rmsd_final`、`single_shot_success_2a/5a`、`cost_guard_skips`
- **日志文件**：默认为 `logs/train/train_{timestamp}.log`；若启用 `--smoke`，则为 `logs/smoke/train/train_{timestamp}.log`
- **运行目录**：`checkpoints/train_{timestamp}/`
- **最新模型**：`checkpoints/train_{timestamp}/latest_model.pt`
- **组合最优模型**：`checkpoints/train_{timestamp}/best_composite_model.pt`
- **Single-shot 2Å 成功率最优模型**：`checkpoints/train_{timestamp}/best_single_shot_success2a_model.pt`
- **Mean RMSD 最优模型**：`checkpoints/train_{timestamp}/best_rmsd_model.pt`
- **默认别名**：`checkpoints/train_{timestamp}/best_model.pt`（与 best_selected_model.pt 相同）

训练结束后（若启用 `--run_test_after_training`）额外输出：

- **独立测试集报告**：`checkpoints/train_{timestamp}/reports/test_metrics.json`
- **三段式 blind 指标**：proposal recall、best-of-k upper bound、最终排序实际效果，以及 proposal/local/ranking failure decomposition
- **Top-N 指标**：`top1/top5/top10` 在 `<2Å` 和 `<5Å` 下的成功率，以及 Top-N 最优 RMSD 的均值
- **默认重排逻辑**：最终 pose 排序默认采用 `pose_rank_score` 与 `center` 分支的线性融合；亲和力与位阻分支以弱辅助信号形式参与，默认权重分别为 `0.08` 与 `0.12`

## Ablation Protocol

推荐可用于论文/专利交底的实验矩阵：

### 核心 ablation（3 组）

1. **Proposal-aware crop 组件对比**
    - `--ablation_mode none`（主方案：proposal-aware crop）
    - 可进一步叠加 `--disable_jitter_crop` / `--disable_hard_negative_crop` 做细粒度 crop ablation
2. **BCE only vs BCE + Pairwise ranking**
    - `--pose_ranking_pair_weight 0`（baseline）
    - `--pose_ranking_pair_weight 0.2`（主方案）
3. **No bootstrap vs Bootstrap**
    - `--pose_bootstrap_frequency 0`（baseline）
    - `--pose_bootstrap_frequency 25`（主方案）
### 多尺度交互 ablation

- `--ablation_mode inter_multiscale_off`
- 仅保留 `ligand_atom <-> protein_atom` 跨图边，关闭 atom-residue / atom-context / molecule-context 多尺度交互

最佳实践：所有实验保持同一 `split_cache_file`、同一 seed、同一训练超参，只改变目标 ablation 开关。

## Project Structure

```text
src/ehfnet/
├── contracts/               # schema 签名与缓存版本契约
│   ├── cache.py             # 图缓存 / ESM 缓存版本标识
│   ├── checkpoint.py        # checkpoint 特征签名与一致性校验
│   └── blind_pool.py        # blind pool 一致性签名
├── data/
│   ├── datasets/            # 数据集核心
│   │   ├── protein_ligand.py    # ProteinLigandDataset
│   │   ├── layout.py            # 数据目录与路径约定
│   │   ├── ligand_sanitize.py   # 配体 RDKit 清洗
│   │   ├── pose_initialization.py # 随机初始构象生成
│   │   └── splitter.py          # Scaffold Split
│   ├── featurizers/         # 输入特征编码
│   │   ├── chemistry.py         # 元素 / 残基枚举
│   │   ├── feature_specs.py     # 特征 schema 定义
│   │   ├── ligand_encoder.py    # 配体原子特征编码
│   │   ├── protein_encoder.py   # 蛋白原子 / 残基特征编码
│   │   ├── protein_segments.py  # 蛋白链段连续性分析
│   │   └── esm_embedding.py     # ESM 语言模型嵌入
│   └── preprocess/          # 预处理编排
│       ├── build_graph_sample.py # 单样本图构建
│       ├── context_repair.py     # 缓存图 context 节点修复
│       └── metadata.py           # 清洗元数据提取
├── geometry/                # SE(3) 几何计算
│   ├── dynamics.py          # 流形插值 / ODE 积分 / Kabsch
│   └── static.py            # 二面角 / 扭转可动原子
├── graph/                   # 异构图构建
│   ├── schema.py            # 节点 / 边类型契约
│   ├── collate.py           # batch 拼接
│   ├── inter_edges.py       # 动态跨图边
│   ├── builders/            # 图构建器
│   ├── crop/                # 运行时局部裁剪
│   ├── features/            # context 节点特征
│   └── topology/            # 层内 / 层间 / 聚合边
├── models/                  # 模型定义
│   ├── ehfnet.py            # EHFNet 主模型
│   ├── layers/              # 编码器 / 嵌入 / FrameConv / RBF
│   └── heads/               # 预测头
├── runtime/                 # 运行期工厂
│   └── factories.py         # build_model / build_dataset
└── training/                # 训练系统
    ├── trainer.py           # 主训练循环
    ├── batch_helpers.py     # batch 工具（裁剪 / 质量目标 / loss 上下文）
    ├── normalization.py     # 训练集归一化统计
    ├── center_sampling.py   # 中心采样策略与 bootstrap
    ├── checkpoint_io.py     # checkpoint 选择与序列化
    ├── validation.py        # 验证 loss 与 Top-N 评估
    ├── flow_matcher.py      # 条件流匹配控制器
    ├── losses.py            # 流匹配多任务损失
    ├── blind_pool.py        # blind candidate pool 管理
    ├── candidate_generation.py  # 统一候选生成引擎
    ├── rerank_losses.py     # reranker 训练损失
    └── inference/           # blind pipeline 推理工具
        ├── center_utils.py  # 中心提议与融合
        └── metrics.py       # 候选记录汇总
```

## License

This project is licensed under the [MIT License](LICENSE).
