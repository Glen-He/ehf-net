"""
数据预处理与图构建

提供从原始文件（配体/蛋白）读取、ESM embedding 获取，以及构建 HeteroData 的工具函数。
"""

import numpy as np

from typing import Any, cast

import torch
import MDAnalysis as mda

from esm.models.esmc import ESMC
from torch_geometric.data import HeteroData

from ehfnet.encoders import LigandEncoder, ProteinEncoder
from ehfnet.datasets.ligand_sanitize import load_ligand_mol
from ehfnet.encoders.esm_embedding import load_or_compute_esm_embeddings
from ehfnet.graph import GraphBuilder


def load_ligand(ligand_path: str):
    """
    读取配体分子，并确保包含 3D conformer。

    Args:
        ligand_path: 配体文件路径（支持 .sdf/.mol2）

    Returns:
        RDKit Mol 对象（去氢后）
    """

    return load_ligand_mol(ligand_path, remove_hs=True, require_conformer=True)


def load_protein(protein_path: str) -> mda.Universe:
    """
    读取蛋白质结构文件为 MDAnalysis Universe。

    Args:
        protein_path: 蛋白质 PDB 文件路径

    Returns:
        MDAnalysis Universe 对象
    """

    return mda.Universe(protein_path)


def get_esm_model(model_name: str = "esmc_300m") -> ESMC:
    """
    加载 ESM 模型并设置为 eval 模式。

    Args:
        model_name: 预训练模型名称

    Returns:
        已加载并放置到可用设备上的 ESMC 模型
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return cast(ESMC, ESMC.from_pretrained(model_name).to(device).eval())


def prepare_graph(
    *,
    pdb_id: str,
    ligand_path: str,
    protein_path: str,
    affinity: float | None,
    feature_factory: Any,
    graph_builder: GraphBuilder,
    esm_cache_path: str | None,
    esm_cache_write_path: str | None,
    esm: str,
    esm_model: ESMC | None = None,
    esm_model_name: str = "esmc_300m",
    pocket_radius: float | None = None,
) -> HeteroData:
    """
    从原始文件构建单个复合物的图数据。

    Args:
        pdb_id: 复合物 ID
        ligand_path: 配体文件路径
        protein_path: 蛋白文件路径
        affinity: 亲和力标签（可选）
        feature_factory: RDKit ChemicalFeatures 工厂（用于配体特征提取）
        graph_builder: 图构建器
        esm_cache_path: 可读取的 ESM 缓存路径（可选）
        esm_cache_write_path: 可写入的 ESM 缓存路径（可选）
        esm: ESM 模式（如 "off"/"file"/"auto"）
        esm_model: 已加载的 ESM 模型实例（可选，用于 auto 推理）
        esm_model_name: ESM 模型名称（esm_model 为空时使用）
        pocket_radius: 口袋提取半径 (Å)。如果提供，则仅保留该半径内的残基。

    Returns:
        构建完成的 HeteroData
    """

    mol = load_ligand(ligand_path)
    universe = load_protein(protein_path)

    ligand_encoder = LigandEncoder(feature_factory=feature_factory)
    protein_encoder = ProteinEncoder()

    esm_embeddings = None

    if esm == "off":
        esm_embeddings = None

    elif esm == "file":

        if esm_cache_path is None:
            raise FileNotFoundError(f"ESM cache not found for {pdb_id}")

        esm_embeddings = load_or_compute_esm_embeddings(
            universe=universe,
            esm_model=None,
            cache_path=esm_cache_path,
            force_recompute=False,
        )

    else:

        if esm_cache_path is not None:
            esm_embeddings = load_or_compute_esm_embeddings(
                universe=universe,
                esm_model=None,
                cache_path=esm_cache_path,
                force_recompute=False,
            )

        else:

            if esm_cache_write_path is None:
                raise ValueError(f"ESM cache_write_path is required for esm=auto ({pdb_id})")

            if esm_model is None:
                esm_model = get_esm_model(model_name=esm_model_name)

            esm_embeddings = load_or_compute_esm_embeddings(
                universe=universe,
                esm_model=esm_model,
                cache_path=esm_cache_write_path,
                force_recompute=False,
            )

    ligand_result = ligand_encoder.encode(mol, strict_torsion=True)
    
    # 如果指定了 pocket_radius，获取配体坐标用于裁剪
    # 将 list 转换为 np.ndarray 以匹配 ProteinEncoder 的类型要求
    ligand_positions = None
    
    if pocket_radius is not None:
        ligand_positions = np.array(ligand_result["positions"], dtype=np.float32)

    protein_result = protein_encoder.encode(
        universe,
        esm_embeddings=esm_embeddings,
        pocket_radius=pocket_radius,
        ligand_positions=ligand_positions,
    )

    data = graph_builder.build(ligand_result, protein_result)
    data.pdb_id = pdb_id

    if affinity is not None:
        data.y_energy = torch.tensor([affinity], dtype=torch.float32)
        
    return data
