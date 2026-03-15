"""
运行时配置工具。

负责加载 TOML 配置、展开嵌套字段并解析共享运行参数，
为各入口脚本提供统一的配置读取逻辑。
"""


import tomllib

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def flatten_config(
    config: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, object]:
    """
    展开嵌套配置字典。

    将 TOML 读取后的层级配置拍平成单层键值映射，
    便于命令行默认值和运行时工厂统一消费。

    Args:
        config: 待展开或转换的配置字典。
        prefix: 递归展开配置时附加在键名前的前缀。

    Returns:
        dict[str, object]: 返回按点号路径展开后的单层配置字典。
    """

    flat: dict[str, object] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def config_to_arg_defaults(config: Mapping[str, Any]) -> dict[str, object]:
    """
    转换 argparse 默认值映射。

    把配置字典整理为命令行参数可直接复用的默认值格式，
    减少入口脚本手动拼装默认值的重复代码。

    Args:
        config: 待展开或转换的配置字典。

    Returns:
        dict[str, object]: 返回可直接作为命令行默认值使用的键值映射。
    """

    flat = flatten_config(config)
    defaults: dict[str, object] = {}
    for key, value in flat.items():
        defaults[key.split(".")[-1]] = value
    return defaults


def load_flattened_toml_config(
    config_path: str | Path | None,
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """
    读取并展开 TOML 配置。

    负责加载 TOML 文件并返回拍平后的参数字典，
    作为各类入口脚本读取项目配置的基础接口。

    Args:
        config_path: 配置文件路径。
        project_root: 项目根目录路径。

    Returns:
        dict[str, object]: 返回从 TOML 文件读取并拍平后的配置字典。
    """

    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_absolute() and project_root is not None:
        path = project_root / path
    if not path.exists():
        return {}

    with path.open("rb") as file:
        raw = tomllib.load(file)
    return config_to_arg_defaults(raw)


def load_train_defaults(
    *,
    config_path: str | Path | None,
    project_root: Path,
) -> dict[str, object]:
    """
    加载训练默认配置。

    合并训练配置与模型配置中的相关字段，
    生成训练入口统一使用的一套默认参数。

    Args:
        config_path: 配置文件路径。
        project_root: 项目根目录路径。

    Returns:
        dict[str, object]: 返回合并训练配置与模型配置后的默认参数字典。
    """

    defaults = load_flattened_toml_config(
        config_path,
        project_root=project_root,
    )
    model_config_path = defaults.get("model_config")
    if isinstance(model_config_path, (str, Path)):
        defaults.update(
            load_flattened_toml_config(
                model_config_path,
                project_root=project_root,
            )
        )
    return defaults


def get_configured_device(
    *,
    config_path: str | Path | None,
    project_root: Path,
    fallback: str | None = None,
) -> str:
    """
    读取共享设备配置。

    从训练配置中解析统一的运行设备设置，
    供训练和预处理脚本复用相同的设备默认值。

    Args:
        config_path: 配置文件路径。
        project_root: 项目根目录路径。
        fallback: 配置缺失时使用的回退值。

    Returns:
        str: 返回当前运行应使用的设备名称。

    Raises:
        RuntimeError: 当运行过程出现不可继续的状态时抛出。
    """

    defaults = load_flattened_toml_config(
        config_path,
        project_root=project_root,
    )
    configured = defaults.get("device")
    if configured is not None:
        return str(configured)
    if fallback is not None:
        return fallback
    raise RuntimeError(
        "Device is not configured. Please set `device` in configs/train.toml "
        "or pass --device explicitly."
    )


def get_configured_smoke(
    *,
    config_path: str | Path | None,
    project_root: Path,
    fallback: bool = False,
) -> bool:
    """
    读取共享 smoke 开关。

    从训练配置中解析 smoke 运行标记，
    供日志目录和运行模式在不同入口间保持一致。

    Args:
        config_path: 配置文件路径。
        project_root: 项目根目录路径。
        fallback: 配置缺失时使用的回退值。

    Returns:
        bool: 返回布尔判断结果。
    """

    defaults = load_flattened_toml_config(
        config_path,
        project_root=project_root,
    )
    configured = defaults.get("smoke")
    if configured is None:
        return fallback
    return bool(configured)


def resolve_interaction_profile(*, ablation_mode: str) -> str:
    """
    解析交互拓扑配置。

    根据消融模式决定使用完整交互还是裁剪后的交互拓扑，
    避免训练入口分散处理这类模式分支。

    Args:
        ablation_mode: 当前训练使用的消融模式名称。

    Returns:
        str: 返回根据当前消融模式解析出的交互拓扑配置名称。
    """

    if ablation_mode == "inter_multiscale_off":
        return "atom_only"
    return "full"
