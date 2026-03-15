"""
HuggingFace 运行时工具。

负责配置缓存目录、解析设备设置，
并避免 ESM 运行时环境初始化顺序问题。
"""


import os

from pathlib import Path

import torch

from ehfnet.runtime import get_configured_device


def configure_hf_cache_env(
    *,
    project_root: str | Path | None = None,
) -> tuple[Path, Path, str]:
    """
    配置 HuggingFace 缓存目录。

    Args:
        project_root: 项目根目录路径。

    Returns:
        tuple[Path, Path, str]: 返回 HuggingFace 主缓存目录、hub 缓存目录及其来源标记。
    """

    existing_hf_home = os.environ.get("HF_HOME")
    existing_hf_hub_cache = os.environ.get("HF_HUB_CACHE")

    if existing_hf_home or existing_hf_hub_cache:
        if existing_hf_home is not None:
            hf_home = Path(existing_hf_home)
        else:
            hf_home = Path(existing_hf_hub_cache).parent

        if existing_hf_hub_cache is not None:
            hub_cache = Path(existing_hf_hub_cache)
        else:
            hub_cache = hf_home / "hub"
            os.environ["HF_HUB_CACHE"] = str(hub_cache)

        return hf_home, hub_cache, "user-env"

    if project_root is None:
        project_root = Path(__file__).resolve().parents[4]

    hf_home = Path(project_root) / ".hf-cache"
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    return hf_home, hub_cache, "project-default"


def resolve_esm_device(
    device: str | torch.device | None,
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> torch.device:
    """
    解析 ESM 推理设备。

    根据配置和当前环境选择可用设备，
    避免预处理阶段在设备配置不一致时静默退化。

    Args:
        device: 运行所用设备，如 CPU 或 CUDA 设备。
        project_root: 项目根目录路径。
        config_path: 配置文件路径。

    Returns:
        torch.device: 返回当前 ESM 推理应使用的具体设备对象。

    Raises:
        RuntimeError: 当运行过程出现不可继续的状态时抛出。
    """

    if device is None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[4]
        if config_path is None:
            config_path = Path(project_root) / "configs" / "train.toml"
        device = get_configured_device(
            config_path=config_path,
            project_root=Path(project_root),
        )

    requested = torch.device(device)
    if requested.type != "cuda":
        return requested

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{requested}', but CUDA is unavailable."
        )

    device_index = 0 if requested.index is None else int(requested.index)
    device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise RuntimeError(
            f"Requested CUDA device index {device_index}, but only "
            f"{device_count} CUDA device(s) are available."
        )

    return torch.device(f"cuda:{device_index}")
