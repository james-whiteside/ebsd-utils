# -*- coding: utf-8 -*-

import math
from enum import Enum
from typing import Self

import numpy
from numba import jit


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


@jit(nopython=True)
def rotation_angle(R: numpy.ndarray) -> float:
    """
    Computes the rotation angle ``θ`` of a rotation matrix ``R``.
    Solves Eqn. 4.1.
    :param R: The rotation matrix ``R``.
    :return: The rotation angle ``θ``.
    """

    if 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) > 1:
        theta = math.acos(1)
    elif 0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1) < -1:
        theta = math.acos(-1)
    else:
        theta = math.acos(0.5 * (abs(R[0][0]) + abs(R[1][1]) + abs(R[2][2]) - 1))

    return theta


@jit(nopython=True)
def misrotation_matrix(R1: numpy.ndarray, R2: numpy.ndarray) -> numpy.ndarray:
    """
    Computes the misrotation matrix ``dR`` between two rotation matrices ``R1`` and ``R2``.
    Solves Eqn. 4.2.
    :param R1: The rotation matrix ``R1``.
    :param R2: The rotation matrix ``R2``.
    :return: The misrotation matrix ``dR``.
    """

    dR = numpy.dot(numpy.linalg.inv(R1), R2)
    return dR


@jit(nopython=True)
def dbscan(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
) -> tuple[int, numpy.ndarray, numpy.ndarray]:
    cluster_count = 0
    category = numpy.zeros((height, width))
    cluster_id = numpy.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            if global_phase_id[y][x] == 0:
                # This point is a noise point.
                category[y][x] = 3
    
    for y0 in range(height):
        for x0 in range(width):
            if category[y0][x0] == 0:
                point_neighbours = 0
    
                for y1 in range(height):
                    for x1 in range(width):
                        if x0 == x1 and y0 == y1:
                            continue
                        elif global_phase_id[y0][x0] != global_phase_id[y1][x1]:
                            continue
                        else:
                            rotation_matrix_0 = reduced_euler_rotation_matrix[y0][x0]
                            rotation_matrix_1 = reduced_euler_rotation_matrix[y1][x1]
                            misrotation_angle = rotation_angle(misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

                            if misrotation_angle <= neighbourhood_radius and x0 != x1 and y0 != y1:
                                point_neighbours += 1
    
                            if point_neighbours >= core_point_neighbour_threshold:
                                # This point is a core point.
                                category[y0][x0] = 1
                                break
    
                    if category[y0][x0] == 1:
                        break
    
    for y0 in range(height):
        for x0 in range(width):
            if category[y0][x0] != 0:
                continue
    
            for y1 in range(height):
                for x1 in range(width):
                    if global_phase_id[y0][x0] == global_phase_id[y1][x1] and category[y1][x1] == 1:
                        rotation_matrix_0 = reduced_euler_rotation_matrix[y0][x0]
                        rotation_matrix_1 = reduced_euler_rotation_matrix[y1][x1]
                        misrotation_angle = rotation_angle(misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

                        if misrotation_angle <= neighbourhood_radius:
                            # This point is a border point.
                            category[y0][x0] = 2
                            break
    
                if category[y0][x0] == 2:
                    break
    
    for y0 in range(height):
        for x0 in range(width):
            if category[y0][x0] == 0:
                # This point is a noise point.
                category[y0][x0] = 3
    
    for y0 in range(height):
        for x0 in range(width):
            if category[y0][x0] == 1 and cluster_id[y0][x0] == 0:
                cluster_count += 1
                cluster_id[y0][x0] = cluster_count
                cluster_core_complete = False
    
                while not cluster_core_complete:
                    cluster_core_complete = True
    
                    for y1 in range(height):
                        for x1 in range(width):
                            if cluster_id[y1][x1] == cluster_count:
                                for y2 in range(height):
                                    for x2 in range(width):
                                        if global_phase_id[y1][x1] == global_phase_id[y2][x2] and category[y2][x2] == 1 and cluster_id[y2][x2] == 0:
                                            rotation_matrix_1 = reduced_euler_rotation_matrix[y1][x1]
                                            rotation_matrix_2 = reduced_euler_rotation_matrix[y2][x2]
                                            misrotation_angle = rotation_angle(misrotation_matrix(rotation_matrix_1, rotation_matrix_2))

                                            if misrotation_angle <= neighbourhood_radius:
                                                cluster_id[y2][x2] = cluster_count
                                                cluster_core_complete = False
    
    for y0 in range(height):
        for x0 in range(width):
            if category[y0][x0] == 2:
                minimum_misrotation_angle = neighbourhood_radius
                cluster_id_of_minimum = 0
    
                for y1 in range(height):
                    for x1 in range(width):
                        if global_phase_id[y0][x0] == global_phase_id[y1][x1] and category[y1][x1] == 1:
                            rotation_matrix_0 = reduced_euler_rotation_matrix[y0][x0]
                            rotation_matrix_1 = reduced_euler_rotation_matrix[y1][x1]
                            misrotation_angle = rotation_angle(misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

                            if misrotation_angle <= minimum_misrotation_angle:
                                minimum_misrotation_angle = misrotation_angle
                                cluster_id_of_minimum = cluster_id[y1][x1]
    
                cluster_id[y0][x0] = cluster_id_of_minimum
    
    return cluster_count, category, cluster_id
