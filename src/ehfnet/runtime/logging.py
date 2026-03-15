"""
运行时日志工具。

负责生成运行后缀、解析日志目录，
并配置终端与文件日志的统一输出格式。
"""


import logging

from datetime import datetime
from pathlib import Path


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
RUN_SUFFIX_FORMAT = "%Y-%m-%d_%H-%M-%S"


def build_run_suffix(requested_suffix: str | None = None) -> str:
    """
    生成运行后缀。

    为一次运行创建统一的时间后缀或校验外部传入后缀，
    用于对齐日志文件名、输出目录和相关产物命名。

    Args:
        requested_suffix: 外部请求的运行后缀；为空时自动生成。

    Returns:
        str: 返回当前运行共用的时间后缀或外部指定后缀。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if requested_suffix is not None:
        suffix = requested_suffix.strip()
        if not suffix:
            raise ValueError("run suffix cannot be empty.")
        return suffix
    return datetime.now().strftime(RUN_SUFFIX_FORMAT)


def resolve_log_root(*, smoke: bool) -> Path:
    """
    解析日志根目录。

    根据 smoke 标记和日志类型确定最终输出目录，
    保证训练、预处理和 smoke 日志按约定分组存放。

    Args:
        smoke: 是否启用 smoke 日志分组。

    Returns:
        Path: 返回对应的路径对象。
    """

    return Path("logs") / "smoke" if smoke else Path("logs")


def configure_text_logging(
    *,
    category: str,
    file_stem: str,
    smoke: bool,
    run_suffix: str | None = None,
) -> tuple[Path, str]:
    """
    配置终端加文件双写的文本日志。

    Args:
        category: 日志类别名称，如 train 或 preprocess。
        file_stem: 日志文件主名，不含扩展名。
        smoke: 是否启用 smoke 日志分组。
        run_suffix: 本次运行使用的统一后缀。

    Returns:
        tuple[Path, str]: `(log_file, resolved_suffix)`。
    """

    resolved_suffix = build_run_suffix(run_suffix)
    log_dir = resolve_log_root(smoke=smoke) / category
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{file_stem}_{resolved_suffix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    return log_file, resolved_suffix
