"""
数据预处理与图构建

提供从原始文件（配体/蛋白）读取、ESM embedding 获取，以及构建 HeteroData 的工具函数。
"""

import os

from typing import Any, cast

import torch
from rdkit import Chem
import MDAnalysis as mda

from esm.models.esmc import ESMC
from torch_geometric.data import HeteroData

from ehfnet.encoders import LigandEncoder, ProteinEncoder
from ehfnet.encoders.esm_embedding import load_or_compute_esm_embeddings
from ehfnet.graph import GraphBuilder


def load_ligand(ligand_path: str) -> Chem.Mol:
    """
    读取配体分子，并确保包含 3D conformer。

    Args:
        ligand_path: 配体文件路径（支持 .sdf/.mol2）

    Returns:
        RDKit Mol 对象（去氢后）
    """

    if ligand_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(ligand_path, sanitize=False)

    else:
        suppl = Chem.SDMolSupplier(ligand_path, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 else None

    if mol is None:
        raise ValueError(f"Failed to load ligand: {ligand_path}")

    try:
        Chem.SanitizeMol(mol)

    except Exception:
        mol.UpdatePropertyCache(strict=False)

    mol = Chem.RemoveHs(mol)

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Ligand has no conformer: {ligand_path}")

    return mol


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

    device = (
        "cuda"
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available()
        else "cpu"
    )

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

    protein_result = protein_encoder.encode(
        universe,
        esm_embeddings=esm_embeddings,
        esm_embedding_file=None,
    )

    data = graph_builder.build(ligand_result, protein_result)
    data.pdb_id = pdb_id

    if affinity is not None:
        data.y_energy = torch.tensor([affinity], dtype=torch.float32)
        
    return data
