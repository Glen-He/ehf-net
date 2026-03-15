"""
契约模块入口。

导出缓存、checkpoint 与 blind pool 相关常量，
统一外部访问这些契约定义的导入路径。
"""


from importlib import import_module

__all__ = [
    "BLIND_POOL_SCHEMA_TAG",
    "CHECKPOINT_FEATURE_SCHEMA_TAG",
    "ESM_CACHE_VERSION_TAG",
    "GRAPH_CACHE_DIRNAME",
    "GRAPH_CACHE_SCHEMA_TAG",
    "PREPROCESS_METADATA_DIRNAME",
    "PREPROCESS_SUMMARY_FILENAME",
    "build_blind_pool_signature",
    "build_feature_signature",
    "build_model_config",
    "validate_checkpoint_contract",
]

_EXPORT_MAP = {
    "BLIND_POOL_SCHEMA_TAG": ("ehfnet.contracts.blind_pool", "BLIND_POOL_SCHEMA_TAG"),
    "CHECKPOINT_FEATURE_SCHEMA_TAG": (
        "ehfnet.contracts.checkpoint",
        "CHECKPOINT_FEATURE_SCHEMA_TAG",
    ),
    "ESM_CACHE_VERSION_TAG": ("ehfnet.contracts.cache", "ESM_CACHE_VERSION_TAG"),
    "GRAPH_CACHE_DIRNAME": ("ehfnet.contracts.cache", "GRAPH_CACHE_DIRNAME"),
    "GRAPH_CACHE_SCHEMA_TAG": ("ehfnet.contracts.cache", "GRAPH_CACHE_SCHEMA_TAG"),
    "PREPROCESS_METADATA_DIRNAME": (
        "ehfnet.contracts.cache",
        "PREPROCESS_METADATA_DIRNAME",
    ),
    "PREPROCESS_SUMMARY_FILENAME": (
        "ehfnet.contracts.cache",
        "PREPROCESS_SUMMARY_FILENAME",
    ),
    "build_blind_pool_signature": (
        "ehfnet.contracts.blind_pool",
        "build_blind_pool_signature",
    ),
    "build_feature_signature": (
        "ehfnet.contracts.checkpoint",
        "build_feature_signature",
    ),
    "build_model_config": ("ehfnet.contracts.checkpoint", "build_model_config"),
    "validate_checkpoint_contract": (
        "ehfnet.contracts.checkpoint",
        "validate_checkpoint_contract",
    ),
}


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

    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
