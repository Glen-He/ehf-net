"""
几何计算模块

提供静态和动态几何计算功能。
"""

from ehfnet.geometry.static import (
    calculate_dihedral,
    get_moving_atoms,
)
from ehfnet.geometry.dynamics import (
    PhysicsConstants,
    PoseUpdater,
    PathInterpolator,
    compute_center_of_mass,
    compute_principal_frame,
    TangentTargetProjector,
)

__all__ = [
    # 静态几何计算（数据预处理）
    "calculate_dihedral",
    "get_moving_atoms",
    # 物理常量
    "PhysicsConstants",
    # 动态几何计算（训练/推理）
    "PoseUpdater",
    "PathInterpolator",
    "compute_center_of_mass",
    "compute_principal_frame",
    "TangentTargetProjector",
]
