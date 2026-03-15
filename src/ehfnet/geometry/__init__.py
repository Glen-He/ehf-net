"""
几何模块入口。

导出静态几何与动力学工具，
统一模型几何相关能力的访问方式。
"""


from importlib import import_module

__all__ = [
    "PathInterpolator",
    "PhysicsConstants",
    "PoseUpdater",
    "TangentTargetProjector",
    "calculate_dihedral",
    "compute_center_of_mass",
    "compute_principal_frame",
    "get_moving_atoms",
]

_EXPORT_MAP = {
    "PathInterpolator": ("ehfnet.geometry.dynamics", "PathInterpolator"),
    "PhysicsConstants": ("ehfnet.geometry.dynamics", "PhysicsConstants"),
    "PoseUpdater": ("ehfnet.geometry.dynamics", "PoseUpdater"),
    "TangentTargetProjector": (
        "ehfnet.geometry.dynamics",
        "TangentTargetProjector",
    ),
    "calculate_dihedral": ("ehfnet.geometry.static", "calculate_dihedral"),
    "compute_center_of_mass": ("ehfnet.geometry.dynamics", "compute_center_of_mass"),
    "compute_principal_frame": (
        "ehfnet.geometry.dynamics",
        "compute_principal_frame",
    ),
    "get_moving_atoms": ("ehfnet.geometry.static", "get_moving_atoms"),
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
