"""
ESM 嵌入工具。

负责蛋白序列提取、ESM 推理、缓存读写和嵌入加载，
服务预处理阶段的残基级表示构建。
"""


import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from MDAnalysis import Universe

from ehfnet.data.featurizers.chemistry import resolve_esm_residue_type
from ehfnet.data.featurizers.protein_segments import segment_residues_by_continuity

if TYPE_CHECKING:
    from esm.models.esmc import ESMC


logger = logging.getLogger(__name__)


def extract_protein_chain_sequences(universe: Universe) -> list[tuple[str, str, list[int]]]:
    """
    提取蛋白链段序列。

    按真实连续肽链切分蛋白残基，并返回序列与残基索引映射，
    为 ESM 嵌入计算准备输入链段。

    Args:
        universe: MDAnalysis Universe 蛋白结构对象。

    Returns:
        list[tuple[str, str, list[int]]]: 返回按连续链段切分后的序列、序列字符串和残基索引列表。
    """

    protein_residues = list(universe.select_atoms("protein").residues)
    if not protein_residues:
        return []

    out: list[tuple[str, str, list[int]]] = []

    for segment in segment_residues_by_continuity(protein_residues):
        seq_list: list[str] = []
        residue_ixs: list[int] = []

        for res in segment.residues:
            resolution = resolve_esm_residue_type(res.resname)

            if resolution.source == "unknown":
                logger.warning(
                    f"Unknown residue '{res.resname}' at ix={res.ix} in segment {segment.key}, "
                    f"mapped to UNK ('{resolution.residue_type.one_letter}')"
                )
            elif resolution.source == "alias":
                logger.debug(
                    "Canonicalized residue '%s' at ix=%s in segment %s to %s for ESM sequence.",
                    resolution.original_resname,
                    res.ix,
                    segment.key,
                    resolution.normalized_resname,
                )

            seq_list.append(resolution.residue_type.one_letter)
            residue_ixs.append(int(res.ix))

        out.append((segment.key, "".join(seq_list), residue_ixs))

    return out


def embeddings_to_residue_ix_map(
    residue_ixs: list[int],
    embeddings: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    回填 ESM 嵌入到残基索引。

    将链段级 ESM 输出重新映射回原始残基 `ix`，
    使后续蛋白编码能够按残基位置直接读取嵌入。

    Args:
        residue_ixs: 与嵌入结果对应的残基索引列表。
        embeddings: 待保存或处理的嵌入结果。

    Returns:
        dict[int, np.ndarray]: 返回以残基 `ix` 为键、对应嵌入向量为值的映射字典。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [L, D], got {embeddings.shape}")

    if len(residue_ixs) != embeddings.shape[0]:
        raise ValueError(
            f"Alignment error: Residue indices count ({len(residue_ixs)}) "
            f"does not match embedding count ({embeddings.shape[0]})."
        )

    return {ix: embeddings[i].copy() for i, ix in enumerate(residue_ixs)}


def compute_esm_embeddings(
    universe: Universe,
    esm_model: "ESMC",
) -> dict[int, np.ndarray]:
    """
    计算 ESM 残基嵌入。

    对提取出的蛋白链段执行模型推理，
    并生成可按残基索引访问的嵌入结果。

    Args:
        universe: MDAnalysis Universe 蛋白结构对象。
        esm_model: 已加载的 ESM 模型实例。

    Returns:
        dict[int, np.ndarray]: 返回当前结构中全部残基的 ESM 嵌入映射。
    """

    from esm.sdk.api import ESMProtein, LogitsConfig

    esm_map: dict[int, np.ndarray] = {}
    chains = extract_protein_chain_sequences(universe)

    if not chains:
        logger.warning("No protein segments found in universe")
        return esm_map

    device = next(esm_model.parameters()).device

    with torch.no_grad():
        for seg_idx, (segid, sequence, residue_ixs) in enumerate(chains):
            if not sequence:
                continue

            try:
                protein_for_esm = ESMProtein(sequence=sequence)
                protein_tensor = esm_model.encode(protein_for_esm)

                if hasattr(protein_tensor, "to"):
                    protein_tensor = protein_tensor.to(device)

                logits_output = esm_model.logits(
                    protein_tensor,
                    LogitsConfig(return_embeddings=True),
                )

                embeddings_tensor = logits_output.embeddings
                if embeddings_tensor is None:
                    raise RuntimeError(
                        f"ESM model returned None embeddings for segment {seg_idx} "
                        f"(segid={segid})."
                    )
                raw_embeddings = embeddings_tensor.squeeze(0)

                expected_len_with_tokens = len(residue_ixs) + 2
                expected_len_without = len(residue_ixs)

                if raw_embeddings.shape[0] == expected_len_with_tokens:
                    residue_embeddings = raw_embeddings[1:-1]
                elif raw_embeddings.shape[0] == expected_len_without:
                    residue_embeddings = raw_embeddings
                else:
                    logger.error(
                        f"ESM embedding shape mismatch for segment {seg_idx}: "
                        f"expected {expected_len_with_tokens} or {expected_len_without}, "
                        f"got {raw_embeddings.shape[0]}"
                    )
                    continue

                cpu_embeddings = residue_embeddings.float().cpu().numpy()
                segment_map = embeddings_to_residue_ix_map(
                    residue_ixs,
                    cpu_embeddings,
                )
                esm_map.update(segment_map)

                logger.info(
                    f"Computed ESM embeddings for segment {seg_idx} "
                    f"({len(residue_ixs)} residues, segid={segid})"
                )
            except Exception as exc:
                logger.error(
                    f"ESM inference failed for segment {seg_idx} (segid={segid}): "
                    f"{type(exc).__name__}: {exc}",
                    exc_info=True,
                )

    return esm_map


def save_esm_embeddings(
    embeddings: dict[int, np.ndarray],
    output_path: str | Path,
) -> None:
    """
    保存 ESM 嵌入缓存。

    将残基嵌入按约定格式写入 `.npz` 文件，
    供后续预处理和训练阶段重复读取。

    Args:
        embeddings: 待保存或处理的嵌入结果。
        output_path: 输出文件路径。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_key_dict = {str(k): v for k, v in embeddings.items()}
    np.savez_compressed(str(output_path), **cast(dict[str, Any], str_key_dict))
    logger.info(f"Saved {len(embeddings)} ESM embeddings to {output_path}")


def load_esm_embeddings(input_path: str | Path) -> dict[int, np.ndarray]:
    """
    加载 ESM 嵌入缓存。

    从 `.npz` 文件恢复残基级嵌入字典，
    为蛋白编码和图构建阶段提供缓存结果。

    Args:
        input_path: 输入文件路径。

    Returns:
        dict[int, np.ndarray]: 返回从缓存文件恢复出的残基嵌入映射。

    Raises:
        FileNotFoundError: 当依赖文件不存在时抛出。
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"ESM embedding file not found: {input_path}")

    with np.load(input_path) as data:
        embeddings = {int(k): data[k].copy() for k in data.files}

    logger.info(f"Loaded {len(embeddings)} ESM embeddings from {input_path}")
    return embeddings


def cache_esm_embeddings(
    *,
    universe: Universe,
    esm_model: "ESMC",
    output_path: str | Path,
) -> dict[int, np.ndarray]:
    """
    计算并写入 ESM 缓存。

    串联嵌入计算与缓存保存流程，
    用于预处理阶段一次性生成可复用的 ESM 文件。

    Args:
        universe: MDAnalysis Universe 蛋白结构对象。
        esm_model: 已加载的 ESM 模型实例。
        output_path: 输出文件路径。

    Returns:
        dict[int, np.ndarray]: 返回新计算并写入缓存的残基嵌入映射。
    """

    embeddings = compute_esm_embeddings(universe, esm_model)
    save_esm_embeddings(embeddings, output_path)
    return embeddings


def load_or_compute_esm_embeddings(
    *,
    universe: Universe,
    esm_model: "ESMC | None",
    cache_path: str | Path,
    force_recompute: bool = False,
) -> dict[int, np.ndarray]:
    """
    加载或计算 ESM 嵌入。

    优先复用已存在的缓存，不存在时再执行模型推理，
    在速度与一致性之间提供统一入口。

    Args:
        universe: MDAnalysis Universe 蛋白结构对象。
        esm_model: 已加载的 ESM 模型实例。
        cache_path: 缓存文件路径。
        force_recompute: 是否忽略已有缓存并强制重新计算 ESM 嵌入。

    Returns:
        dict[int, np.ndarray]: 返回最终可供后续编码复用的残基嵌入映射。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        try:
            return load_esm_embeddings(cache_path)
        except Exception as exc:
            logger.warning(
                f"Failed to load cached embeddings from {cache_path}: {exc}. Will recompute."
            )

    if esm_model is None:
        raise ValueError(
            f"ESM model required to compute embeddings, but got None. "
            f"Cache file {cache_path} does not exist or force_recompute=True."
        )

    logger.info(f"Computing ESM embeddings (cache: {cache_path})")
    return cache_esm_embeddings(
        universe=universe,
        esm_model=esm_model,
        output_path=cache_path,
    )
