"""
蛋白配体数据集。

负责索引加载、样本缓存访问、预处理调度和图样本读取，
是训练与评估阶段访问数据的主入口。
"""


import json
import logging
import os
import os.path as osp
import zlib

from collections import Counter
from collections.abc import Callable
from typing import Any, cast, overload

import torch
from rdkit.Chem import ChemicalFeatures, RDConfig
from torch import Tensor
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from ehfnet.data.datasets.layout import (
    ESM_CACHE_VERSION_TAG,
    GRAPH_CACHE_DIRNAME,
    GRAPH_CACHE_SCHEMA_TAG,
    PREPROCESS_METADATA_DIRNAME,
    PREPROCESS_SUMMARY_FILENAME,
    esm_cache_paths,
    ligand_path,
    load_index,
    protein_path,
)
from ehfnet.data.preprocess import (
    ensure_context_features,
    extract_ligand_sanitize_metadata,
    get_esm_model,
    normalize_ligand_sanitize_mode,
    prepare_graph_sample,
)
from ehfnet.data.datasets.ligand_sanitize import LigandSanitizationError
from ehfnet.data.datasets.pose_initialization import (
    build_start_positions,
)
from ehfnet.graph import ESMEmbeddingFiller, GraphBuilder, build_graph_cost_profile


logger = logging.getLogger(__name__)
GRAPH_COST_PROFILE_VERSION = 1


class ProteinLigandDataset(Dataset):
    """
    蛋白配体复合物数据集。

    封装索引加载、缓存访问、预处理触发与样本读取逻辑，
    为训练、验证和评估流程提供统一的数据集接口。
    """

    def __init__(
        self,
        root: str,
        index_file: str,
        *,
        esm_root: str | None = None,
        esm: str,
        esm_model_name: str,
        esm_device: str | torch.device | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        r_cutoff_intra: float,
        max_neighbors_intra: int,
        atom_neighbor_cap: int,
        residue_neighbor_cap: int,
        residue_radius_scale: float,
        residue_radius_bias: float,
        ligand_atom_fallback_k: int,
        protein_atom_fallback_k: int,
        protein_residue_fallback_k: int,
        interaction_profile: str,
        force_reprocess: bool = False,
        esm_dim: int,
    ) -> None:
        """
        初始化蛋白配体数据集。

        Args:
            root: 数据集根目录。
            index_file: 数据索引文件路径。
            esm_root: ESM 缓存根目录。
            esm: ESM 处理模式或缓存策略。
            esm_model_name: ESM 主干模型名称。
            esm_device: 执行 ESM 推理时使用的设备。
            transform: 样本读取完成后应用的在线变换函数。
            pre_transform: 样本写入缓存前执行的一次性预处理函数。
            pre_filter: 样本写入缓存前执行的过滤函数。
            r_cutoff_intra: 图内边构建的距离截断半径。
            max_neighbors_intra: 图内边构建时每类节点允许的最大邻居数。
            atom_neighbor_cap: 原子层图内边的邻居上限。
            residue_neighbor_cap: 残基层图内边的邻居上限。
            residue_radius_scale: 残基层邻域半径相对原子半径的缩放系数。
            residue_radius_bias: 残基层邻域半径的额外偏置。
            ligand_atom_fallback_k: 配体原子图内边回退到 kNN 时的邻居数。
            protein_atom_fallback_k: 蛋白原子图内边回退到 kNN 时的邻居数。
            protein_residue_fallback_k: 蛋白残基层图内边回退到 kNN 时的邻居数。
            interaction_profile: 跨图交互拓扑配置。
            force_reprocess: 是否忽略已有缓存并强制重新预处理。
            esm_dim: ESM 残基嵌入维度。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """
        required_args = {
            "esm_model_name": esm_model_name,
            "r_cutoff_intra": r_cutoff_intra,
            "max_neighbors_intra": max_neighbors_intra,
            "atom_neighbor_cap": atom_neighbor_cap,
            "residue_neighbor_cap": residue_neighbor_cap,
            "residue_radius_scale": residue_radius_scale,
            "residue_radius_bias": residue_radius_bias,
            "ligand_atom_fallback_k": ligand_atom_fallback_k,
            "protein_atom_fallback_k": protein_atom_fallback_k,
            "protein_residue_fallback_k": protein_residue_fallback_k,
            "interaction_profile": interaction_profile,
            "esm_dim": esm_dim,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "ProteinLigandDataset is missing required explicit configuration "
                f"values: {missing_args}."
            )
        self.index_file = index_file
        self.esm_root = esm_root
        self.esm = esm
        self.esm_model_name = str(esm_model_name)
        self.esm_device = None if esm_device is None else str(esm_device)
        self.force_reprocess = force_reprocess
        self.esm_dim = int(esm_dim)

        self.index_df = load_index(index_file)
        self.cleaned_dir = osp.join(root, "cleaned")
        if not osp.exists(self.cleaned_dir) and not self.index_df.empty:
            first_pdb = self.index_df.iloc[0]["pdb_id"]
            if osp.exists(osp.join(root, first_pdb)):
                logger.warning(
                    "Data found in %s, but expected in %s. Please move data to 'cleaned' subdirectory.",
                    root,
                    self.cleaned_dir,
                )

        fdef_path = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.feature_factory = cast(
            Any, ChemicalFeatures
        ).BuildFeatureFactory(fdef_path)

        esm_filler = ESMEmbeddingFiller(
            embed_dim=self.esm_dim,
            fill_strategy="zeros",
        )
        self.graph_builder = GraphBuilder(
            r_cutoff_intra=float(r_cutoff_intra),
            max_neighbors_intra=int(max_neighbors_intra),
            atom_neighbor_cap=int(atom_neighbor_cap),
            residue_neighbor_cap=int(residue_neighbor_cap),
            residue_radius_scale=float(residue_radius_scale),
            residue_radius_bias=float(residue_radius_bias),
            ligand_atom_fallback_k=int(ligand_atom_fallback_k),
            protein_atom_fallback_k=int(protein_atom_fallback_k),
            protein_residue_fallback_k=int(protein_residue_fallback_k),
            esm_filler=esm_filler,
            interaction_profile=str(interaction_profile),
        )

        self._esm_model = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self._build_valid_index()
        self.affinity_stats = self.compute_affinity_stats()

    def compute_affinity_stats(
        self,
        indices: list[int] | None = None,
    ) -> dict[str, float]:
        """
        计算亲和力标签的均值和标准差。

        Args:
            indices: 待处理样本的索引集合。

        Returns:
            dict[str, float]: 包含亲和力均值与标准差的统计字典。
        """

        if self.index_df.empty:
            return {"mean": 0.0, "std": 1.0}

        if indices is None:
            pdb_ids = self._valid_pdb_ids
        else:
            if not indices:
                return {"mean": 0.0, "std": 1.0}
            pdb_ids = [self._valid_pdb_ids[int(i)] for i in indices]

        affinities = self.index_df[self.index_df["pdb_id"].isin(pdb_ids)][
            "affinity"
        ].to_numpy(dtype=float)
        if affinities.size == 0:
            return {"mean": 0.0, "std": 1.0}

        mean = float(affinities.mean())
        std = max(float(affinities.std()), 1e-3)
        return {"mean": mean, "std": std}

    def set_affinity_stats(self, stats: dict[str, float]) -> None:
        """
        更新运行期使用的 affinity 归一化统计。

        Args:
            stats: 待写入或使用的统计量对象。
        """

        self.affinity_stats = {
            "mean": float(stats.get("mean", 0.0)),
            "std": max(float(stats.get("std", 1.0)), 1e-3),
        }

    @overload
    def denormalize_affinity(self, val: Tensor) -> Tensor: ...

    @overload
    def denormalize_affinity(self, val: float) -> float: ...

    def denormalize_affinity(self, val: float | Tensor) -> float | Tensor:
        """
        将归一化的亲和力值还原为 pKd。

        Args:
            val: 待反归一化或处理的亲和力数值。

        Returns:
            float | Tensor: 返回还原到原始 pKd 标度后的亲和力数值。
        """

        return val * self.affinity_stats["std"] + self.affinity_stats["mean"]

    @property
    def raw_dir(self) -> str:
        return self.cleaned_dir

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, GRAPH_CACHE_DIRNAME)

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{pdb_id}.pt" for pdb_id in self.index_df["pdb_id"]]

    @property
    def preprocess_metadata_dir(self) -> str:
        return osp.join(self.processed_dir, PREPROCESS_METADATA_DIRNAME)

    @property
    def preprocess_summary_path(self) -> str:
        return osp.join(self.processed_dir, PREPROCESS_SUMMARY_FILENAME)

    def _preprocess_metadata_path(self, pdb_id: str) -> str:
        return osp.join(self.preprocess_metadata_dir, f"{pdb_id}.json")

    def _write_preprocess_metadata(self, pdb_id: str, metadata: dict[str, Any]) -> None:
        os.makedirs(self.preprocess_metadata_dir, exist_ok=True)
        payload = dict(metadata)
        payload.setdefault("graph_mode", "blind")
        payload.setdefault("cache_dir", GRAPH_CACHE_DIRNAME)
        payload.setdefault("esm_model_name", self.esm_model_name)
        payload.setdefault("esm_dim", self.esm_dim)
        if "graph_cost_profile" in payload:
            payload["graph_cost_profile_version"] = GRAPH_COST_PROFILE_VERSION
        with open(self._preprocess_metadata_path(pdb_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)

    def _load_cached_preprocess_metadata(self, pdb_id: str) -> dict[str, Any] | None:
        metadata_path = self._preprocess_metadata_path(pdb_id)
        if not osp.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to read preprocess metadata for %s: %s",
                pdb_id,
                exc,
            )
            return None

        payload["ligand_sanitize_mode"] = normalize_ligand_sanitize_mode(
            payload.get("ligand_sanitize_mode")
        )
        payload.setdefault("graph_mode", "blind")
        payload.setdefault("cache_dir", GRAPH_CACHE_DIRNAME)
        payload.setdefault("esm_model_name", None)
        payload.setdefault("esm_dim", None)
        payload.setdefault("graph_cost_profile", None)
        payload.setdefault("graph_cost_profile_version", 0)
        return payload

    def _is_cached_preprocess_metadata_compatible(self, metadata: dict[str, Any]) -> bool:
        """
        判断缓存样本是否与当前 ESM 配置一致。

        Returns:
            bool: 返回布尔判断结果。
        """

        return (
            metadata.get("esm_model_name") == self.esm_model_name
            and int(metadata.get("esm_dim", -1)) == self.esm_dim
        )

    def _load_or_recover_preprocess_metadata(
        self,
        pdb_id: str,
        graph_path: str,
    ) -> dict[str, Any]:
        cached = self._load_cached_preprocess_metadata(pdb_id)
        if cached is not None:
            return cached

        data = cast(
            HeteroData,
            torch.load(graph_path, map_location="cpu", weights_only=False),
        )
        metadata = extract_ligand_sanitize_metadata(data)
        metadata["esm_model_name"] = None
        metadata["esm_dim"] = None
        metadata["graph_cost_profile"] = build_graph_cost_profile(data)
        metadata["graph_cost_profile_version"] = GRAPH_COST_PROFILE_VERSION
        self._write_preprocess_metadata(pdb_id, metadata)
        return metadata

    def get_graph_cost_profile(self, idx: int) -> dict[str, int]:
        """
        读取或恢复指定样本的图成本画像。

        Args:
            idx: 当前访问的样本索引。

        Returns:
            dict[str, int]: 包含节点、边和扭转规模信息的成本画像。

        Raises:
            IndexError: 当索引超出有效样本范围时抛出。
        """
        if idx < 0 or idx >= len(self._valid_pdb_ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self._valid_pdb_ids)})")

        pdb_id = self._valid_pdb_ids[idx]
        graph_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")
        metadata = self._load_or_recover_preprocess_metadata(pdb_id, graph_path)
        cached_profile = metadata.get("graph_cost_profile")
        cached_version = int(metadata.get("graph_cost_profile_version", 0))
        if isinstance(cached_profile, dict) and cached_version == GRAPH_COST_PROFILE_VERSION:
            return {str(key): int(value) for key, value in cached_profile.items()}

        data = cast(HeteroData, torch.load(graph_path, map_location="cpu", weights_only=False))
        graph_cost_profile = build_graph_cost_profile(data)
        metadata["graph_cost_profile"] = graph_cost_profile
        metadata["graph_cost_profile_version"] = GRAPH_COST_PROFILE_VERSION
        self._write_preprocess_metadata(pdb_id, metadata)
        return graph_cost_profile

    def _write_preprocess_summary(self, summary: dict[str, Any]) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.preprocess_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

    def download(self) -> None:
        return None

    def process(self) -> None:
        """
        处理并缓存全部样本到 processed_dir。
        """

        if self.force_reprocess:
            logger.info("Force reprocess enabled - overwriting existing cache")

        total = len(self.index_df)
        logger.info("Processing %d complexes...", total)

        success_count = 0
        skip_count = 0
        filtered_count = 0
        error_count = 0
        other_failure_count = 0
        sanitize_counts: Counter[str] = Counter(
            {
                "full": 0,
                "partial": 0,
                "rejected": 0,
                "unknown": 0,
            }
        )

        for _, row in tqdm(self.index_df.iterrows(), total=total, desc="Processing"):
            pdb_id = row["pdb_id"]
            affinity = float(row["affinity"])
            out_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

            if not self.force_reprocess and osp.exists(out_path):
                should_skip_cached_sample = False
                try:
                    metadata = self._load_or_recover_preprocess_metadata(
                        pdb_id,
                        out_path,
                    )
                    if self._is_cached_preprocess_metadata_compatible(metadata):
                        sanitize_counts[
                            normalize_ligand_sanitize_mode(
                                metadata.get("ligand_sanitize_mode")
                            )
                        ] += 1
                        should_skip_cached_sample = True
                    else:
                        logger.info(
                            "Rebuilding cached sample %s because ESM config changed "
                            "(model=%s, dim=%s).",
                            pdb_id,
                            self.esm_model_name,
                            self.esm_dim,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to recover sanitize metadata for cached sample %s: %s",
                        pdb_id,
                        exc,
                    )
                    sanitize_counts["unknown"] += 1
                    should_skip_cached_sample = True
                if should_skip_cached_sample:
                    skip_count += 1
                    continue

            try:
                data = self._process_one(pdb_id, affinity)
                if data is None:
                    sanitize_counts["unknown"] += 1
                    error_count += 1
                    other_failure_count += 1
                    continue

                metadata = extract_ligand_sanitize_metadata(data)
                sanitize_counts[metadata["ligand_sanitize_mode"]] += 1

                if self.pre_filter is not None and not self.pre_filter(data):
                    filtered_count += 1
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                metadata["graph_cost_profile"] = build_graph_cost_profile(data)
                metadata["graph_cost_profile_version"] = GRAPH_COST_PROFILE_VERSION
                torch.save(data, out_path)

                # 预处理阶段同步生成 start_pos 缓存，保证训练时所有有效样本均有该属性。
                # 此时原始配体文件必然存在（刚被 _process_one 使用），是生成的最可靠时机。
                if hasattr(data["ligand_atom"], "pos"):
                    expected_num_atoms = int(data["ligand_atom"].pos.size(0))
                    start_pos = self._load_or_build_start_pos(pdb_id, expected_num_atoms)
                    if start_pos is None:
                        os.remove(out_path)
                        logger.warning(
                            "Skipping %s: start_pos generation failed, graph cache removed.",
                            pdb_id,
                        )
                        error_count += 1
                        continue

                self._write_preprocess_metadata(pdb_id, metadata)
                success_count += 1

            except LigandSanitizationError as exc:
                self._write_preprocess_metadata(
                    pdb_id,
                    {
                        "ligand_sanitize_mode": "rejected",
                        "ligand_partial_sanitize": False,
                        "ligand_full_sanitize_flag": exc.full_flag,
                        "ligand_partial_sanitize_flag": (
                            -1 if exc.partial_flag is None else exc.partial_flag
                        ),
                    },
                )
                logger.warning("Ligand sanitize rejected %s: %s", pdb_id, exc)
                sanitize_counts["rejected"] += 1
                error_count += 1
            except Exception as exc:
                logger.warning("Error processing %s: %s", pdb_id, exc)
                sanitize_counts["unknown"] += 1
                error_count += 1
                other_failure_count += 1

        summary = {
            "graph_cache_schema": GRAPH_CACHE_SCHEMA_TAG,
            "graph_mode": "blind",
            "cache_dir": GRAPH_CACHE_DIRNAME,
            "esm_cache_version": ESM_CACHE_VERSION_TAG,
            "esm_model_name": self.esm_model_name,
            "esm_dim": self.esm_dim,
            "processed_dir": osp.abspath(self.processed_dir),
            "index_file": osp.abspath(self.index_file),
            "force_reprocess": bool(self.force_reprocess),
            "totals": {
                "indexed": total,
                "success": success_count,
                "cached": skip_count,
                "filtered": filtered_count,
                "errors": error_count,
                "other_failures": other_failure_count,
            },
            "ligand_sanitize": {
                "full": int(sanitize_counts["full"]),
                "partial": int(sanitize_counts["partial"]),
                "rejected": int(sanitize_counts["rejected"]),
                "unknown": int(sanitize_counts["unknown"]),
            },
        }
        self._write_preprocess_summary(summary)

        logger.info(
            "Processing complete: %d success, %d cached, %d filtered, %d errors",
            success_count,
            skip_count,
            filtered_count,
            error_count,
        )
        logger.info(
            "Ligand sanitize summary: full=%d partial=%d rejected=%d unknown=%d",
            sanitize_counts["full"],
            sanitize_counts["partial"],
            sanitize_counts["rejected"],
            sanitize_counts["unknown"],
        )
        logger.info("Preprocess summary written to %s", self.preprocess_summary_path)

        if self._esm_model is not None:
            logger.info("Releasing ESM model from GPU memory...")
            del self._esm_model
            self._esm_model = None
            torch.cuda.empty_cache()

    def _process_one(self, pdb_id: str, affinity: float) -> HeteroData | None:
        pdb_dir = osp.join(self.raw_dir, pdb_id)
        lig = ligand_path(pdb_id, pdb_dir)
        pro = protein_path(pdb_id, pdb_dir)
        if lig is None or pro is None:
            return None

        esm_cache_path, esm_cache_write_path = esm_cache_paths(
            pdb_id=pdb_id,
            pdb_dir=pdb_dir,
            esm_root=self.esm_root,
            esm_model_name=self.esm_model_name,
        )
        if self.esm == "auto" and esm_cache_path is None and self._esm_model is None:
            self._esm_model = get_esm_model(
                model_name=self.esm_model_name,
                device=self.esm_device,
            )

        return prepare_graph_sample(
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
            esm_device=self.esm_device,
        )

    def _build_valid_index(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        processed_files = os.listdir(self.processed_dir)
        allowed = set(self.index_df["pdb_id"].tolist())
        candidates = sorted(
            [
                f.replace("data_", "").replace(".pt", "")
                for f in processed_files
                if f.startswith("data_")
                and f.endswith(".pt")
                and f.replace("data_", "").replace(".pt", "") in allowed
            ]
        )

        valid_pdb_ids: list[str] = []
        excluded = 0
        for pdb_id in candidates:
            cache_path = self._diffdock_like_cache_path(pdb_id)
            if osp.exists(cache_path):
                valid_pdb_ids.append(pdb_id)
                continue

            # start_pos 缓存缺失（来自旧版预处理）：尝试即时生成并过滤失败样本。
            file_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")
            try:
                data = torch.load(file_path, weights_only=False)
            except Exception as e:
                logger.warning("Skipping %s: failed to load graph cache: %s", pdb_id, e)
                excluded += 1
                continue

            if not hasattr(data["ligand_atom"], "pos"):
                logger.warning("Skipping %s: ligand_atom.pos missing in graph cache.", pdb_id)
                excluded += 1
                continue

            expected_num_atoms = int(data["ligand_atom"].pos.size(0))
            start_pos = self._load_or_build_start_pos(pdb_id, expected_num_atoms)
            if start_pos is None:
                logger.warning("Skipping %s: start_pos unavailable.", pdb_id)
                excluded += 1
            else:
                valid_pdb_ids.append(pdb_id)

        if excluded:
            logger.warning(
                "%d samples excluded: start_pos unavailable. "
                "Re-run preprocessing to rebuild missing caches.",
                excluded,
            )
        self._valid_pdb_ids = valid_pdb_ids
        self._pdb_to_idx = {pdb: i for i, pdb in enumerate(valid_pdb_ids)}
        logger.info("Dataset ready: %d valid samples", len(valid_pdb_ids))

    def _diffdock_like_cache_path(self, pdb_id: str) -> str:
        return osp.join(
            self.root,
            "candidates",
            "diffdock_like_init",
            pdb_id,
            "poses.pt",
        )

    def _load_or_build_start_pos(
        self,
        pdb_id: str,
        expected_num_atoms: int,
    ) -> torch.Tensor | None:
        cache_path = self._diffdock_like_cache_path(pdb_id)
        if osp.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning("Corrupted start_pos cache for %s, will rebuild: %s", pdb_id, e)
                cached = None
            if cached is not None:
                start_pos = (
                    cached.get("ligand_start_pos") if isinstance(cached, dict) else cached
                )
                if (
                    isinstance(start_pos, torch.Tensor)
                    and start_pos.ndim == 2
                    and start_pos.size(1) == 3
                    and int(start_pos.size(0)) == expected_num_atoms
                ):
                    return start_pos.float()

        pdb_dir = osp.join(self.raw_dir, pdb_id)
        lig_path = ligand_path(pdb_id, pdb_dir)
        if lig_path is None:
            return None

        seed = zlib.adler32(pdb_id.encode("utf-8")) & 0xFFFFFFFF
        try:
            start_pos_np = build_start_positions(
                lig_path,
                random_seed=seed,
            )
        except Exception as e:
            logger.warning("build_start_positions failed for %s: %s", pdb_id, e)
            return None
        start_pos = torch.as_tensor(start_pos_np, dtype=torch.float32)
        if start_pos.ndim != 2 or start_pos.size(1) != 3:
            logger.warning(
                "Start pose has unexpected shape for %s: %s.",
                pdb_id,
                tuple(start_pos.shape),
            )
            return None
        if int(start_pos.size(0)) != expected_num_atoms:
            logger.warning(
                "Start pose atom count mismatch for %s: expected %d, got %d.",
                pdb_id,
                expected_num_atoms,
                int(start_pos.size(0)),
            )
            return None

        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save({"ligand_start_pos": start_pos}, cache_path)
        return start_pos

    def len(self) -> int:
        return len(self._valid_pdb_ids)

    def get(self, idx: int) -> HeteroData:
        """
        公开接口说明。

        Args:
            idx: 当前访问的样本索引。

        Returns:
            HeteroData: 指定索引对应的缓存图样本。

        Raises:
            IndexError: 当索引超出有效样本范围时抛出。
            RuntimeError: 当缓存样本损坏且无法恢复时抛出。
        """
        if idx < 0 or idx >= len(self._valid_pdb_ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self._valid_pdb_ids)})")

        pdb_id = self._valid_pdb_ids[idx]
        file_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")
        data = cast(HeteroData, torch.load(file_path, weights_only=False))

        if hasattr(data, "y_energy"):
            raw_val = data.y_energy
            if not isinstance(raw_val, torch.Tensor):
                raw_val = torch.tensor(raw_val, dtype=torch.float32)
            data.y_energy_raw = raw_val
            mean = self.affinity_stats["mean"]
            std = self.affinity_stats["std"]
            data.y_energy = (raw_val - mean) / std

        if hasattr(data["ligand_atom"], "pos"):
            expected_num_atoms = int(data["ligand_atom"].pos.size(0))
            start_pos = self._load_or_build_start_pos(pdb_id, expected_num_atoms)
            # _build_valid_index 已保证 start_pos 缓存存在；此处 pos 回退仅作防御性兜底，
            # 语义上等价于 flow_matcher 中 seed_pos=None 的行为（同样以 x_ref 为基准）。
            data["ligand_atom"]["start_pos"] = (
                start_pos if start_pos is not None else data["ligand_atom"].pos.clone()
            )

        data = ensure_context_features(data, self.graph_builder)
        data.dataset_index = int(idx)
        data.dataset_pdb_id = str(pdb_id)
        return data
