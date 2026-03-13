# Bottom-Up Refactor Notes

这份说明书记录了 2026-03-13 这轮底层重构的核心改动、设计原因和后续验证重点。

## 1. 本轮修改的目标

本轮不是围绕训练主循环做局部修补，而是从 preprocessing / encoding / geometry 三条底层路径修正会直接污染模型输入语义的结构问题：

1. ESM 序列构造不再把整条蛋白按 segment 粗暴拼接
2. backbone torsion 邻接不再依赖 `resid ± 1`
3. 配体 RDKit sanitization 不再在失败后静默继续
4. `Element.UNK` 的连续特征不再使用错误量纲
5. 主惯量帧对线性/平面分子的修复不再被错误缩进绕过

## 2. 主要改动

### 2.1 统一蛋白连续链段定义

新增文件：

- `src/ehfnet/encoders/protein_segments.py`

新增能力：

- `segment_residues_by_continuity(...)`
- `is_peptide_continuous(...)`

判据：

1. 若链标签存在，要求链标签一致
2. `prev.C` 和 `next.N` 都存在
3. `||C_prev - N_next|| <= 2.2A`

这样做的目的：

- 避免多链被错误拼成一条序列
- 避免缺失残基、链断裂、编号跳跃时仍被错误视为连续
- 让 ESM 分段和 torsion 邻接使用同一条真实蛋白拓扑定义

### 2.2 ESM embedding 分段策略升级

修改文件：

- `src/ehfnet/encoders/esm_embedding.py`

行为变化：

- 旧逻辑：按 `segment` 提取整段序列
- 新逻辑：按 `segment_residues_by_continuity(...)` 的输出切成真实连续肽链段

保留的约定：

- residue 到 embedding 的映射仍然使用 `residue.ix`
- 也就是说，索引对齐仍然稳定；只修正了 sequence context

### 2.3 backbone torsion 邻接升级

修改文件：

- `src/ehfnet/encoders/protein_encoder.py`

行为变化：

- 旧逻辑：`prev_res = resid - 1`, `next_res = resid + 1`
- 新逻辑：先按真实连续肽链段切分，再在段内按位置取前后邻居

这样可以正确处理：

- PDB 编号跳跃
- insertion / missing residues
- 多链混合

### 2.4 配体 sanitization 升级

修改文件：

- `src/ehfnet/datasets/prepare.py`

新增策略：

1. 先尝试 full sanitize
2. full sanitize 失败时，显式记录 warning
3. 再尝试保留环/芳香性/杂化等核心化学语义的 partial sanitize
4. partial 仍失败则直接拒绝该 ligand

核心原则：

- 不再允许“半失败的化学语义”静默流入 descriptor 提取

### 2.5 `Element.UNK` 连续特征量纲修正

修改文件：

- `src/ehfnet/encoders/chemistry.py`

旧问题：

- `UNK.vdw_radius_mm3 = 1.70`
- 其他元素同字段量级在 `140~236`

新策略：

- `UNK` 连续特征改为已知元素的中位量级

目的：

- 避免未知元素节点在连续特征空间出现极端量纲跳变
- 保留 `atomic_idx = UNK` 这个显式分类信号

### 2.6 主惯量帧退化修复

修改文件：

- `src/ehfnet/geometry/dynamics.py`

修复内容：

- 线性/平面分子的 `small_sv_mask` 修复分支从错误的条件缩进中解开
- 现在即使 batch 内没有 `<3` 原子分子，线性/平面分子也会走修复逻辑

## 3. ESM 缓存兼容性变更

修改文件：

- `src/ehfnet/datasets/pdbbind.py`

本轮引入了新的缓存版本标签：

- `chainseg_v2`

新缓存文件名：

- 本地：`{pdb_id}_esm_chainseg_v2.npz`
- 全局：`{pdb_id}_chainseg_v2.npz`

原因：

- 老缓存是基于旧的粗粒度 segment 拼接策略生成的
- 若继续复用旧缓存，新的连续链段修复不会真正生效

## 4. 为什么这样改是更优方案

### 4.1 不是只修索引，而是修 sequence semantics

旧 ESM 方案最大的问题不在 `ix` 映射，而在 sequence context 错误。

本轮保留：

- `ix -> embedding` 的稳定映射

本轮修正：

- 多链/断链的伪上下文

### 4.2 不是只修 torsion 数值，而是统一 protein topology source of truth

现在 ESM 分段和 torsion 邻接都来自同一个 helper：

- `segment_residues_by_continuity(...)`

这样可以避免两条预处理路径各自定义“邻接”的语义，后面再漂。

### 4.3 不是宽容脏 ligand，而是拒绝伪化学语义

这轮对 ligand 的策略是：

- 能构建可信的化学语义就保留
- 不能就丢弃

这比继续在失败状态下计算 QED / scaffold / aromaticity descriptor 更稳。

## 5. 受影响文件

- `src/ehfnet/encoders/protein_segments.py` (new)
- `src/ehfnet/encoders/esm_embedding.py`
- `src/ehfnet/encoders/protein_encoder.py`
- `src/ehfnet/encoders/chemistry.py`
- `src/ehfnet/datasets/prepare.py`
- `src/ehfnet/datasets/pdbbind.py`
- `src/ehfnet/geometry/dynamics.py`

## 6. 已做验证

已完成：

- 相关文件 `py_compile` 静态编译通过

未完成：

- 端到端 runtime smoke
- 新 ESM cache 的实际重建
- 多链 / 断链样本的 spot check

## 7. 建议的后续验证

建议至少做下面 4 组检查：

### 7.1 ESM 分段 spot check

挑若干典型样本：

- 单链连续
- 多链同 segment
- 缺失残基
- 编号跳跃

打印：

- segment key
- segment 长度
- residue ix 范围

确认不会再把多链硬拼。

项目里已经提供了一个可执行检查脚本：

- `scripts/check_esm_alignment.py`

示例：

```bash
python scripts/check_esm_alignment.py \
  --protein /path/to/xxxx_protein.pdb \
  --cache /path/to/xxxx_esm_chainseg_v2.npz \
  --mode auto \
  --dump-csv /tmp/xxxx_esm_alignment.csv
```

它会检查：

- 连续链段切分是否符合预期
- residue.ix 是否与 ESM embedding 一一对应
- 是否存在 missing / extra embedding 索引
- 并可导出逐残基对齐表

### 7.2 torsion 邻接 spot check

随机抽样若干残基，检查：

- `prev_res_by_ix`
- `next_res_by_ix`

确认不是简单 `resid ± 1`。

### 7.3 ligand sanitization 统计

记录：

- full sanitize 成功数量
- partial sanitize 成功数量
- 最终丢弃数量

如果 partial / discard 比例异常高，需要回头看数据质量。

### 7.4 几何帧稳定性

对线性分子、近平面分子做 smoke：

- `compute_principal_frame(...)` 是否输出有限值
- 不同 batch 组合下 `small_sv_mask` 是否都能触发

## 8. 当前剩余风险

本轮已经修掉了最明显的 preprocessing / geometry correctness 问题，但还存在这些待验证点：

1. 新的 ESM 分段会强制生成新缓存，第一次处理成本会升高
2. ligand partial sanitize 的样本占比需要实际统计
3. `TangentTargetProjector.decompose(...)` 的 dense CPU solve 仍然偏重，属于后续性能优化项，不是这轮 correctness 修复重点
