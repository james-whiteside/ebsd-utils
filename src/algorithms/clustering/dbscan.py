# -*- coding: utf-8 -*-

from enum import Enum
from typing import Self
from numpy import ndarray
from numba import cuda
from src.algorithms.clustering.dbscan_cpu import _dbscan_cpu
from src.algorithms.clustering.dbscan_gpu import _dbscan_gpu
from src.utilities.config import Config


class ClusterCategory(Enum):
    CORE = 1
    BORDER = 2
    NOISE = 3

    @property
    def code(self) -> str:
        return self.name[0]

    @classmethod
    def from_code(cls, code: str) -> Self:
        for category in ClusterCategory:
            if category.name[0] == code:
                return category

        raise ValueError(f"Value is not a valid cluster category code: {code}")


def dbscan(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
) -> tuple[int, ndarray, ndarray]:
    if Config().use_cuda:
        if cuda.is_available():
            return _dbscan_gpu(
                width,
                height,
                global_phase_id,
                reduced_euler_rotation_matrix,
                core_point_neighbour_threshold,
                neighbourhood_radius
            )
        else:
            print(f"Warning: Config specifies to use CUDA but no compatible CUDA device was detected.")

    return _dbscan_cpu(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        core_point_neighbour_threshold,
        neighbourhood_radius,
    )
