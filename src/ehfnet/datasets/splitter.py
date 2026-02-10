"""
数据集划分模块 (Splitting Strategy)

提供基于分子骨架 (Bemis-Murcko Scaffold) 的数据集划分策略。
相比随机划分，骨架划分能更真实地评估模型对未知化学空间的泛化能力。

Features:
- 确定性排序：保证可复现性
- 鲁棒的 RDKit 处理：自动处理解析失败的分子
- 贪心平衡算法：确保划分比例尽可能精准
"""

import os
import random
import logging
from collections import defaultdict
from typing import Protocol, cast

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class SizedDataset(Protocol):
    """
    定义支持 len() 的 Dataset 协议
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> object: ...


class PDBBindLikeDataset(SizedDataset, Protocol):
    """
    定义带 index_df 和 raw_dir 属性的 Dataset 协议
    """

    index_df: pd.DataFrame
    raw_dir: str


def generate_scaffold(mol: Chem.Mol, include_chirality: bool = False) -> str:
    """
    计算分子的 Bemis-Murcko 骨架 SMILES。
    
    Args:
        mol: RDKit 分子对象
        include_chirality: 是否在骨架中保留手性信息

    Returns:
        骨架的 SMILES 字符串。如果计算失败，返回空字符串。
    """

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    
    except Exception:
        return ""


def _read_ligand_safe(path: str) -> Chem.Mol | None:
    """
    安全读取配体文件 (SDF/MOL2)，带自动清洗和错误处理。
    """

    if not os.path.exists(path):
        return None

    mol = None

    try:

        # 优先尝试 SDF (PDBBind 标准格式)
        if path.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path, sanitize=False)
            mol = suppl[0] if len(suppl) > 0 else None

        # 备选 MOL2
        elif path.endswith(".mol2"):
            mol = Chem.MolFromMol2File(path, sanitize=False)
        
        # 简单的清洗，防止坏分子导致骨架提取崩溃
        if mol is not None:

            try:
                Chem.SanitizeMol(mol)

            except Exception:
                mol.UpdatePropertyCache(strict=False)

    except Exception:
        return None

    return mol


class ScaffoldSplitter:
    """
    骨架划分器
    """

    def __init__(
        self,
        include_chirality: bool = False,
        seed: int = 42
    ):
        """
        Args:
            include_chirality: 是否区分手性骨架
            seed: 随机种子 (用于处理同骨架内的排序)
        """

        self.include_chirality = include_chirality
        self.seed = seed

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.9,
        frac_val: float = 0.1,
        frac_test: float = 0.0,
        index_df: pd.DataFrame | None = None
    ) -> tuple[Subset, Subset, Subset]:
        """
        执行划分。

        Args:
            dataset: PDBBindDataset 实例
            frac_train: 训练集比例
            frac_val: 验证集比例
            frac_test: 测试集比例
            index_df: 数据集的索引 DataFrame (用于加速路径查找)

        Returns:
            (train_subset, val_subset, test_subset)
        """

        # 参数校验
        if not np.isclose(frac_train + frac_val + frac_test, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {frac_train + frac_val + frac_test}")

        if index_df is None:
            if hasattr(dataset, "index_df") and hasattr(dataset, "raw_dir"):
                index_df = getattr(dataset, "index_df")
            else:
                raise ValueError("Dataset must have 'index_df' attribute or it must be provided.")
        
        # 此时 index_df 保证不为 None
        assert index_df is not None

        logger.info(f"Start Scaffold Split (seed={self.seed})...")
        
        # 1. 生成所有样本的骨架
        # scaffold_map: dict[scaffold_smiles, list[dataset_index]]
        scaffold_map = defaultdict(list)
        invalid_count = 0
        
        # 获取 raw_dir (假设 dataset 结构)
        raw_dir = getattr(dataset, "raw_dir", ".")

        # 使用 tqdm 显示进度，因为 I/O 较慢
        pbar = tqdm(index_df.iterrows(), total=len(index_df), desc="Analyzing Scaffolds")
        
        for idx, row in pbar:
            pdb_id = row["pdb_id"]
            
            # 构建文件路径逻辑 (复用 pdbbind.py 的逻辑)
            # 这里为了解耦，重新构建路径检查，也可以让 dataset 提供 helper
            pdb_dir = os.path.join(raw_dir, pdb_id)
            lig_sdf = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
            lig_mol2 = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")
            
            mol_path = lig_sdf if os.path.exists(lig_sdf) else lig_mol2
            
            mol = _read_ligand_safe(mol_path)
            
            if mol is None:
                # 读取失败的分子归入 "null_scaffold"，防止丢数据
                scaffold = "null_scaffold_error"
                invalid_count += 1

            else:
                scaffold = generate_scaffold(mol, self.include_chirality)

            scaffold_map[scaffold].append(idx)

        if invalid_count > 0:
            logger.warning(f"Failed to process {invalid_count} ligands. Assigned to 'null' group.")

        # 2. 排序骨架组
        # 关键步骤：按每个骨架包含的分子数量降序排列
        # 这样处理是为了先分配“大块头”，防止最后剩下的大簇导致比例严重失衡
        scaffold_sets = list(scaffold_map.values())
        # 排序键：(分子数量, 第一个分子的索引) -> 保证确定性
        scaffold_sets.sort(key=lambda x: (len(x), x[0]), reverse=True)

        # 3. 贪心分配 (Greedy Allocation)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        # 确保 dataset 支持 len() 操作
        if not hasattr(dataset, '__len__'):
            raise TypeError("Dataset must have __len__ method")
        
        sized_dataset = cast(SizedDataset, dataset)
        dataset_size = len(sized_dataset)
        train_cutoff = int(dataset_size * frac_train)
        val_cutoff = int(dataset_size * (frac_train + frac_val))

        for group in scaffold_sets:
            # 当前 group 的长度
            l = len(group)
            
            # 逻辑：只要训练集没满，就往训练集塞
            # 这种策略保证了训练集的骨架多样性最大化
            if len(train_indices) + l <= train_cutoff:
                train_indices.extend(group)

            elif len(train_indices) + len(val_indices) + l <= val_cutoff:
                val_indices.extend(group)
                
            else:
                test_indices.extend(group)

        # 4. 组内 Shuffle
        # 骨架分组完成后，我们在各个集内部打乱顺序，方便 DataLoader 取样
        rng = random.Random(self.seed)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        logger.info(
            f"Split Result - Train: {len(train_indices)} ({len(train_indices)/dataset_size:.1%}), "
            f"Val: {len(val_indices)} ({len(val_indices)/dataset_size:.1%}), "
            f"Test: {len(test_indices)} ({len(test_indices)/dataset_size:.1%})"
        )
        
        # 检查是否因为大骨架导致验证集过小
        if len(val_indices) == 0 and frac_val > 0:
            logger.warning("Validation set is empty! This happens when one huge scaffold dominates the dataset.")

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices)
        )
