"""
PDBBind 数据集

提供基于 PDBBind 目录结构的 Dataset 实现，并将原始复合物处理为 HeteroData 缓存。
"""

import os
import os.path as osp
import logging
import torch
import pandas as pd

from torch import Tensor
from tqdm import tqdm
from collections.abc import Callable
from typing import Any, cast, overload

from rdkit.Chem import ChemicalFeatures, RDConfig

from torch_geometric.data import Dataset, HeteroData

from ehfnet.graph import GraphBuilder, ESMEmbeddingFiller
from ehfnet.datasets.prepare import prepare_graph, get_esm_model


logger = logging.getLogger(__name__)


def load_index(index_file: str) -> pd.DataFrame:
    """
    加载数据索引文件。

    仅支持 CSV 格式：
    - 必须包含列 {"pdb_id", "affinity"}

    Args:
        index_file: 索引文件路径

    Returns:
        包含 pdb_id 与 affinity 的 DataFrame
    """

    if not osp.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    if not index_file.endswith(".csv"):
        raise ValueError("Index file must be a CSV file.")

    df = pd.read_csv(index_file)
    required_cols = {"pdb_id", "affinity"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # 过滤掉 affinity 为 NaN 的无效行
    initial_len = len(df)
    df = df.dropna(subset=["affinity"])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaN affinity from index.")

    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()

    return df


def ligand_path(pdb_id: str, pdb_dir: str) -> str | None:
    """
    获取配体文件路径（优先 SDF，其次 MOL2）。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录

    Returns:
        配体文件路径；若不存在返回 None
    """

    sdf = osp.join(pdb_dir, f"{pdb_id}_ligand.sdf")

    if osp.exists(sdf):
        return sdf

    mol2 = osp.join(pdb_dir, f"{pdb_id}_ligand.mol2")

    if osp.exists(mol2):
        return mol2

    return None


def protein_path(pdb_id: str, pdb_dir: str) -> str | None:
    """
    获取蛋白质 PDB 文件路径。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录

    Returns:
        蛋白质 PDB 路径；若不存在返回 None
    """

    p = osp.join(pdb_dir, f"{pdb_id}_protein.pdb")
    return p if osp.exists(p) else None


def esm_cache_paths(pdb_id: str, pdb_dir: str, esm_root: str | None) -> tuple[str | None, str | None]:
    """
    获取 ESM embedding 缓存的读路径与写路径。

    优先使用复合物目录下的本地缓存；若配置了 esm_root，则尝试在 esm_root 下读取/写入。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录
        esm_root: 全局 ESM 缓存目录（可选）

    Returns:
        (read_path, write_path)
    """


    local_path = osp.join(pdb_dir, f"{pdb_id}_esm.npz")

    if osp.exists(local_path):
        return local_path, local_path

    if esm_root and osp.isdir(esm_root):
        global_path = osp.join(esm_root, f"{pdb_id}.npz")

        if osp.exists(global_path):
            return global_path, global_path

        return None, global_path

    return None, local_path


class PDBBindDataset(Dataset):
    """
    PDBBind 数据集。

    读取 index_file 并在 processed_dir 下缓存每个复合物的图数据文件 data_{pdb_id}.pt。
    """


    def __init__(
        self,
        root: str,
        index_file: str,
        *,
        esm_root: str | None = None,
        esm: str = "auto",
        esm_model_name: str = "esmc_300m",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        r_cutoff_intra: float = 5.0,
        r_cutoff_inter: float = 6.0,
        max_neighbors_intra: int = 64,
        max_neighbors_inter: int = 32,
        interaction_profile: str = "full",
        force_reprocess: bool = False,
        esm_dim: int = 960,
        pocket_radius: float | None = 20.0,
    ) -> None:
        """
        Args:
            root: 数据集根目录（包含 raw/processed）
            index_file: 索引文件路径（CSV 或 PDBBind INDEX 格式）
            esm_root: 全局 ESM 缓存目录（可选）
            esm: ESM 处理模式（如 "auto"）
            esm_model_name: ESM 模型名称
            transform: PyG Dataset 的 transform（可选）
            pre_transform: PyG Dataset 的 pre_transform（可选）
            pre_filter: PyG Dataset 的 pre_filter（可选）
            r_cutoff_intra: 图内边半径阈值
            r_cutoff_inter: 跨图边半径阈值
            max_neighbors_intra: 图内最大邻居数
            max_neighbors_inter: 跨图最大邻居数
            interaction_profile: 跨图交互配置（"full" 或 "atom_only"）
            force_reprocess: 是否强制重建缓存
            esm_dim: ESM embedding 维度
            pocket_radius: 口袋提取半径 (Å)。设为 None 则不进行裁剪。
        """
        self.index_file = index_file
        self.esm_root = esm_root
        self.esm = esm
        self.esm_model_name = esm_model_name
        self.force_reprocess = force_reprocess
        self.esm_dim = esm_dim
        self.pocket_radius = pocket_radius

        self.index_df = load_index(index_file)

        # 强制指定 cleaned 目录
        self.cleaned_dir = osp.join(root, "cleaned")

        if not osp.exists(self.cleaned_dir):

            # 兼容性检查：如果 cleaned 不存在但 root 下直接有数据，发出警告
            if not self.index_df.empty:
                first_pdb = self.index_df.iloc[0]["pdb_id"]

                if osp.exists(osp.join(root, first_pdb)):
                    logger.warning(f"Data found in {root}, but expected in {self.cleaned_dir}. Please move data to 'cleaned' subdirectory.")
        
        fdef_path = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.feature_factory = cast(Any, ChemicalFeatures).BuildFeatureFactory(fdef_path)

        # 默认为 960 (ESMC 300M)，如果使用 ESM-3 请改为 1152
        esm_filler = ESMEmbeddingFiller(embed_dim=self.esm_dim)

        self.graph_builder = GraphBuilder(
            r_cutoff_intra=r_cutoff_intra,
            r_cutoff_inter=r_cutoff_inter,
            max_neighbors_intra=max_neighbors_intra,
            max_neighbors_inter=max_neighbors_inter,
            esm_filler=esm_filler,
            interaction_profile=interaction_profile,
        )

        self._esm_model = None
        
        # [新增] 计算亲和力统计数据用于归一化
        self.affinity_stats = self._compute_affinity_stats()

        super().__init__(root, transform, pre_transform, pre_filter)
        self._build_valid_index()


    def _compute_affinity_stats(self) -> dict[str, float]:
        """
        计算亲和力标签的均值和标准差。
        """
        if self.index_df.empty:
            return {"mean": 0.0, "std": 1.0}
            
        affinities = self.index_df["affinity"].to_numpy(dtype=float)
        mean = float(affinities.mean())
        std = float(affinities.std())
        std = max(std, 1e-3)
        
        logger.info(f"Affinity stats: mean={mean:.4f}, std={std:.4f}")
        return {"mean": mean, "std": std}

    @overload
    def denormalize_affinity(self, val: Tensor) -> Tensor: ...
    @overload
    def denormalize_affinity(self, val: float) -> float: ...
    def denormalize_affinity(self, val: float | Tensor) -> float | Tensor:
        """
        将归一化的亲和力值还原为 pKd。
        """
        
        return val * self.affinity_stats["std"] + self.affinity_stats["mean"]

    @property
    def raw_dir(self) -> str:
        return self.cleaned_dir

    @property
    def raw_file_names(self) -> list[str]:
        """
        Returns:
            raw 目录需要存在的文件名列表
        """

        return []

    @property
    def processed_file_names(self) -> list[str]:
        """
        Returns:
            processed 目录期望生成的文件名列表
        """

        return [f"data_{pdb_id}.pt" for pdb_id in self.index_df["pdb_id"]]

    def download(self) -> None:
        return None

    def process(self):
        """
        处理并缓存全部样本到 processed_dir。
        """

        if self.force_reprocess:
            logger.info("Force reprocess enabled - overwriting existing cache")

        total = len(self.index_df)
        logger.info(f"Processing {total} complexes...")

        success_count = 0
        skip_count = 0
        error_count = 0

        for _, row in tqdm(self.index_df.iterrows(), total=total, desc="Processing"):
            pdb_id = row["pdb_id"]
            affinity = float(row["affinity"])
            out_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

            if not self.force_reprocess and osp.exists(out_path):
                skip_count += 1
                continue

            try:
                data = self._process_one(pdb_id, affinity)

                if data is None:
                    error_count += 1
                    continue

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, out_path)
                success_count += 1

            except Exception as e:
                logger.warning(f"Error processing {pdb_id}: {e}")
                error_count += 1

        logger.info(
            f"Processing complete: {success_count} success, {skip_count} cached, {error_count} errors"
        )

        # 释放 ESM 模型显存，因为它在训练阶段不再需要
        if self._esm_model is not None:
            logger.info("Releasing ESM model from GPU memory...")
            del self._esm_model
            self._esm_model = None
            torch.cuda.empty_cache()


    def _process_one(self, pdb_id: str, affinity: float) -> HeteroData | None:
        """
        处理单个复合物并构建图数据。

        Args:
            pdb_id: 复合物 ID
            affinity: 亲和力标签

        Returns:
            生成的 HeteroData；若缺文件或处理失败返回 None
        """

        pdb_dir = osp.join(self.raw_dir, pdb_id)

        lig = ligand_path(pdb_id, pdb_dir)
        pro = protein_path(pdb_id, pdb_dir)

        if lig is None or pro is None:
            return None

        esm_cache_path, esm_cache_write_path = esm_cache_paths(pdb_id, pdb_dir, self.esm_root)

        if self.esm == "auto" and esm_cache_path is None and self._esm_model is None:
            self._esm_model = get_esm_model(model_name=self.esm_model_name)

        data = prepare_graph(
                    pdb_id=pdb_id,
                    ligand_path=lig,
                    protein_path=pro,
                    affinity=affinity,
                    feature_factory=self.feature_factory,
                    graph_builder=self.graph_builder,
                    esm_cache_path=esm_cache_path,
                    esm_cache_write_path=esm_cache_write_path,
                    esm=self.esm,
                    esm_model=self._esm_model,
                    esm_model_name=self.esm_model_name,
                    pocket_radius=self.pocket_radius,
                )

        return data

    def _build_valid_index(self):
        """
        扫描 processed_dir 并构建可用样本索引。
        """

        os.makedirs(self.processed_dir, exist_ok=True)
        processed_files = os.listdir(self.processed_dir)
        allowed = set(self.index_df["pdb_id"].tolist())
        valid_pdb_ids = sorted(
            [
                f.replace("data_", "").replace(".pt", "")
                for f in processed_files
                if f.startswith("data_") and f.endswith(".pt")
                and f.replace("data_", "").replace(".pt", "") in allowed
            ]
        )

        self._valid_pdb_ids = valid_pdb_ids
        self._pdb_to_idx = {pdb: i for i, pdb in enumerate(valid_pdb_ids)}
        logger.info(f"Dataset ready: {len(valid_pdb_ids)} valid samples")


    def len(self) -> int:
        """
        Returns:
            可用样本数量
        """

        return len(self._valid_pdb_ids)


    def get(self, idx: int) -> HeteroData:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            HeteroData 样本
        """

        if idx < 0 or idx >= len(self._valid_pdb_ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self._valid_pdb_ids)})")

        pdb_id = self._valid_pdb_ids[idx]
        file_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

        data = cast(HeteroData, torch.load(file_path, weights_only=False))
        
        # [新增] 几何合理性检查：检测原子间最小距离，防止奇异解
        if "ligand_atom" in data and hasattr(data["ligand_atom"], "pos"):
            lig_pos = data["ligand_atom"].pos
            if lig_pos.shape[0] > 1:
                # 计算配体原子间的最小距离
                dist_mat = torch.cdist(lig_pos, lig_pos, p=2)
                # 排除对角线（自身距离为0）
                dist_mat = dist_mat + torch.eye(dist_mat.shape[0], device=dist_mat.device) * 1000.0
                min_dist = dist_mat.min().item()
                
                # 原子间最小合理距离约 0.5 Å（共价键长度通常 > 1.0 Å）
                if min_dist < 0.5:
                    logger.warning(
                        f"Sample {pdb_id} has unreasonable geometry: min atom distance = {min_dist:.3f} Å. "
                        "This may cause numerical instability."
                    )
        
        # [新增] 实时归一化逻辑
        if hasattr(data, "y_energy"):
            raw_val = data.y_energy
            
            # [鲁棒性修改] 确保 raw_val 是 Tensor，防止意外的类型问题
            if not isinstance(raw_val, torch.Tensor):
                raw_val = torch.tensor(raw_val, dtype=torch.float)
                
            # 保存原始值用于评估
            data.y_energy_raw = raw_val
            
            # 归一化: (x - mean) / std
            # 显式转换为 float tensor 进行计算，确保结果仍为 Tensor
            mean = self.affinity_stats["mean"]
            std = self.affinity_stats["std"]
            data.y_energy = (raw_val - mean) / std
            
        return data
