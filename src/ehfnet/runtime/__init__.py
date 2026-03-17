"""
运行时模块入口。

导出配置、工厂和日志工具，
统一训练与推理侧公共能力的访问方式。
"""


from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ehfnet.runtime.factories import (
        build_dataset,
        build_dataset_from_model_config,
        build_model,
        build_model_from_config,
    )
    from ehfnet.runtime.logging import build_run_suffix, configure_text_logging
    from ehfnet.runtime.config import (
        flatten_config,
        get_configured_device,
        get_configured_smoke,
        load_flattened_toml_config,
        load_train_defaults,
        resolve_interaction_profile,
    )

__all__ = [
    "build_run_suffix",
    "build_dataset",
    "build_dataset_from_model_config",
    "build_model",
    "build_model_from_config",
    "configure_text_logging",
    "flatten_config",
    "get_configured_device",
    "get_configured_smoke",
    "load_flattened_toml_config",
    "load_train_defaults",
    "resolve_interaction_profile",
]

_EXPORT_MAP = {
    "build_dataset": ("ehfnet.runtime.factories", "build_dataset"),
    "build_dataset_from_model_config": (
        "ehfnet.runtime.factories",
        "build_dataset_from_model_config",
    ),
    "build_model": ("ehfnet.runtime.factories", "build_model"),
    "build_model_from_config": ("ehfnet.runtime.factories", "build_model_from_config"),
    "build_run_suffix": ("ehfnet.runtime.logging", "build_run_suffix"),
    "configure_text_logging": ("ehfnet.runtime.logging", "configure_text_logging"),
    "flatten_config": ("ehfnet.runtime.config", "flatten_config"),
    "get_configured_device": ("ehfnet.runtime.config", "get_configured_device"),
    "get_configured_smoke": ("ehfnet.runtime.config", "get_configured_smoke"),
    "load_flattened_toml_config": (
        "ehfnet.runtime.config",
        "load_flattened_toml_config",
    ),
    "load_train_defaults": ("ehfnet.runtime.config", "load_train_defaults"),
    "resolve_interaction_profile": (
        "ehfnet.runtime.config",
        "resolve_interaction_profile",
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
