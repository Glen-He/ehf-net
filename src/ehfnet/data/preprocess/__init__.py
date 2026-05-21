"""
预处理模块入口。

导出图构建、上下文修复和元数据工具，
统一预处理侧公开组件的访问方式。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ehfnet.data.preprocess.build_graph_sample import get_esm_model, prepare_graph_sample
    from ehfnet.data.preprocess.context_repair import ensure_context_features
    from ehfnet.data.preprocess.filters import LigandGeometryPreFilter
    from ehfnet.data.preprocess.hf_runtime import configure_hf_cache_env, resolve_esm_device
    from ehfnet.data.preprocess.metadata import extract_ligand_sanitize_metadata, normalize_ligand_sanitize_mode


__all__ = [
    "configure_hf_cache_env",
    "ensure_context_features",
    "extract_ligand_sanitize_metadata",
    "get_esm_model",
    "LigandGeometryPreFilter",
    "normalize_ligand_sanitize_mode",
    "prepare_graph_sample",
    "resolve_esm_device",
]


def __getattr__(name: str):
    """
    按名称返回公开对象。

    仅在首次访问时执行真实导入，
    用于避免包初始化阶段触发重模块加载或循环依赖。

    Args:
        name: 请求访问或解析的公开对象名称。

    Returns:
        object: 返回与名称对应的惰性导出对象。

    Raises:
        AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
    """

    if name in {"get_esm_model", "prepare_graph_sample"}:
        from ehfnet.data.preprocess.build_graph_sample import (
            get_esm_model,
            prepare_graph_sample,
        )

        exports = {
            "get_esm_model": get_esm_model,
            "prepare_graph_sample": prepare_graph_sample,
        }
        return exports[name]
    if name == "ensure_context_features":
        from ehfnet.data.preprocess.context_repair import ensure_context_features

        return ensure_context_features
    if name == "LigandGeometryPreFilter":
        from ehfnet.data.preprocess.filters import LigandGeometryPreFilter

        return LigandGeometryPreFilter
    if name in {"configure_hf_cache_env", "resolve_esm_device"}:
        from ehfnet.data.preprocess.hf_runtime import (
            configure_hf_cache_env,
            resolve_esm_device,
        )

        exports = {
            "configure_hf_cache_env": configure_hf_cache_env,
            "resolve_esm_device": resolve_esm_device,
        }
        return exports[name]
    if name in {"extract_ligand_sanitize_metadata", "normalize_ligand_sanitize_mode"}:
        from ehfnet.data.preprocess.metadata import (
            extract_ligand_sanitize_metadata,
            normalize_ligand_sanitize_mode,
        )

        exports = {
            "extract_ligand_sanitize_metadata": extract_ligand_sanitize_metadata,
            "normalize_ligand_sanitize_mode": normalize_ligand_sanitize_mode,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
