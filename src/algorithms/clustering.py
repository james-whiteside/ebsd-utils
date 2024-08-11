# -*- coding: utf-8 -*-

import math
from enum import Enum
from typing import Self
import numpy
from numba import jit, cuda
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
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
) -> tuple[int, numpy.ndarray, numpy.ndarray]:
    if Config().orientation_clustering_use_cuda:
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


def _dbscan_gpu(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
) -> tuple[int, numpy.ndarray, numpy.ndarray]:
    category = numpy.zeros((height, width))
    cluster_id = numpy.zeros((height, width))
    inverse_euler_rotation_matrix = numpy.zeros((height, width, 3, 3))
    misrotation_matrix_cache = numpy.zeros((height, width, 3, 3))

    _compute_inverted_matrices(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
    )

    global_phase_id_ = cuda.to_device(global_phase_id)
    reduced_euler_rotation_matrix_ = cuda.to_device(reduced_euler_rotation_matrix)
    category_ = cuda.to_device(category)
    cluster_id_ = cuda.to_device(cluster_id)
    inverse_euler_rotation_matrix_ = cuda.to_device(inverse_euler_rotation_matrix)
    misrotation_matrix_cache_ = cuda.to_device(misrotation_matrix_cache)

    block_size = (16, 16)
    grid_size = (math.ceil(width / block_size[0]), math.ceil(height / block_size[1]))

    _assign_unindexed_point_category[grid_size, block_size](
        width,
        height,
        global_phase_id_,
        category_,
    )

    _assign_core_point_category[grid_size, block_size](
        width,
        height,
        global_phase_id_,
        reduced_euler_rotation_matrix_,
        inverse_euler_rotation_matrix_,
        misrotation_matrix_cache_,
        core_point_neighbour_threshold,
        neighbourhood_radius,
        category_,
    )

    _assign_border_point_category[grid_size, block_size](
        width,
        height,
        global_phase_id_,
        reduced_euler_rotation_matrix_,
        inverse_euler_rotation_matrix_,
        misrotation_matrix_cache_,
        neighbourhood_radius,
        category_,
    )

    _assign_noise_point_category[grid_size, block_size](
        width,
        height,
        category_,
    )

    cluster_count = _assign_core_point_cluster_id(
        width,
        height,
        global_phase_id_,
        reduced_euler_rotation_matrix_,
        inverse_euler_rotation_matrix_,
        misrotation_matrix_cache_,
        neighbourhood_radius,
        category_,
        cluster_id_,
        grid_size,
        block_size,
    )

    _assign_border_point_cluster_id[grid_size, block_size](
        width,
        height,
        global_phase_id_,
        reduced_euler_rotation_matrix_,
        inverse_euler_rotation_matrix_,
        misrotation_matrix_cache_,
        neighbourhood_radius,
        category_,
        cluster_id_,
    )

    _reassign_unindexed_point_category[grid_size, block_size](
        width,
        height,
        category_,
    )

    category = category_.copy_to_host()
    cluster_id = cluster_id_.copy_to_host()

    return cluster_count, category, cluster_id


@jit
def _misrotation_angle(inverse_matrix_1: numpy.ndarray, matrix_2: numpy.ndarray, cache: numpy.ndarray) -> float:
    for j in range(3):
        for i in range(3):
            cache[j][i] = inverse_matrix_1[j][0] * matrix_2[0][i]
            cache[j][i] += inverse_matrix_1[j][1] * matrix_2[1][i]
            cache[j][i] += inverse_matrix_1[j][2] * matrix_2[2][i]

    if 0.5 * (abs(cache[0][0]) + abs(cache[1][1]) + abs(cache[2][2]) - 1) > 1:
        angle = math.acos(1)
    elif 0.5 * (abs(cache[0][0]) + abs(cache[1][1]) + abs(cache[2][2]) - 1) < -1:
        angle = math.acos(-1)
    else:
        angle = math.acos(0.5 * (abs(cache[0][0]) + abs(cache[1][1]) + abs(cache[2][2]) - 1))

    return angle


@jit
def _compute_inverted_matrices(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
) -> None:
    for y in range(height):
        for x in range(width):
            if global_phase_id[y][x] == 0:
                continue

            inverse_euler_rotation_matrix[y][x] = numpy.linalg.inv(reduced_euler_rotation_matrix[y][x])


@cuda.jit
def _assign_unindexed_point_category(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    category: numpy.ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if global_phase_id[thread_y][thread_x] == 0:
        category[thread_y][thread_x] = -1


@cuda.jit
def _assign_core_point_category(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
    misrotation_matrix_cache: numpy.ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
    category: numpy.ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] != 0:
        return

    point_neighbours = 0

    for y in range(height):
        for x in range(width):
            if thread_x == x and thread_y == y:
                continue
            elif global_phase_id[thread_y][thread_x] != global_phase_id[y][x]:
                continue
            else:
                misrotation_angle = _misrotation_angle(
                    inverse_euler_rotation_matrix[thread_y][thread_x],
                    reduced_euler_rotation_matrix[y][x],
                    misrotation_matrix_cache[thread_y][thread_x],
                )

                if misrotation_angle <= neighbourhood_radius and thread_x != x and thread_y != y:
                    point_neighbours += 1

                if point_neighbours >= core_point_neighbour_threshold:
                    category[thread_y][thread_x] = 1
                    break

        if category[thread_y][thread_x] == 1:
            break


@cuda.jit
def _assign_border_point_category(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
    misrotation_matrix_cache: numpy.ndarray,
    neighbourhood_radius: float,
    category: numpy.ndarray,
):
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] != 0:
        return

    for y in range(height):
        for x in range(width):
            if global_phase_id[thread_y][thread_x] == global_phase_id[y][x] and category[y][x] == 1:
                misrotation_angle = _misrotation_angle(
                    inverse_euler_rotation_matrix[thread_y][thread_x],
                    reduced_euler_rotation_matrix[y][x],
                    misrotation_matrix_cache[thread_y][thread_x],
                )

                if misrotation_angle <= neighbourhood_radius:
                    category[thread_y][thread_x] = 2
                    break

        if category[thread_y][thread_x] == 2:
            break


@cuda.jit
def _assign_noise_point_category(
    width: int,
    height: int,
    category: numpy.ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] == 0:
        category[thread_y][thread_x] = 3


def _assign_core_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
    misrotation_matrix_cache: numpy.ndarray,
    neighbourhood_radius: float,
    category: numpy.ndarray,
    cluster_id: numpy.ndarray,
    grid_size: tuple[int, int],
    block_size: tuple[int, int],
) -> int:
    cluster_count = 0

    for y in range(height):
        for x in range(width):
            if category[y][x] == 1 and cluster_id[y][x] == 0:
                cluster_count += 1
                cluster_id[y][x] = cluster_count
                cluster_core_complete = cuda.to_device(numpy.array([False]))

                while not cluster_core_complete[0]:
                    cluster_core_complete[0] = True

                    _grow_cluster_core[grid_size, block_size](
                        width,
                        height,
                        global_phase_id,
                        reduced_euler_rotation_matrix,
                        inverse_euler_rotation_matrix,
                        misrotation_matrix_cache,
                        neighbourhood_radius,
                        cluster_count,
                        category,
                        cluster_id,
                        cluster_core_complete,
                    )

    return cluster_count


@cuda.jit
def _grow_cluster_core(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
    misrotation_matrix_cache: numpy.ndarray,
    neighbourhood_radius: float,
    cluster_count: int,
    category: numpy.ndarray,
    cluster_id: numpy.ndarray,
    cluster_core_complete: numpy.ndarray,
):
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if cluster_id[thread_y][thread_x] != 0:
        return

    for y in range(height):
        for x in range(width):
            if cluster_id[y][x] == cluster_count:
                if global_phase_id[y][x] == global_phase_id[thread_y][thread_x] and category[thread_y][thread_x] == 1:
                    misrotation_angle = _misrotation_angle(
                        inverse_euler_rotation_matrix[thread_y][thread_x],
                        reduced_euler_rotation_matrix[y][x],
                        misrotation_matrix_cache[thread_y][thread_x],
                    )

                    if misrotation_angle <= neighbourhood_radius:
                        cluster_id[thread_y][thread_x] = cluster_count

                        if cluster_core_complete[0]:
                            cluster_core_complete[0] = False


@cuda.jit
def _assign_border_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: numpy.ndarray,
    reduced_euler_rotation_matrix: numpy.ndarray,
    inverse_euler_rotation_matrix: numpy.ndarray,
    misrotation_matrix_cache: numpy.ndarray,
    neighbourhood_radius: float,
    category: numpy.ndarray,
    cluster_id: numpy.ndarray,
):
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] != 2:
        return

    minimum_misrotation_angle = neighbourhood_radius
    cluster_id_of_minimum = 0

    for y in range(height):
        for x in range(width):
            if global_phase_id[thread_y][thread_x] == global_phase_id[y][x] and category[y][x] == 1:
                misrotation_angle = _misrotation_angle(
                    inverse_euler_rotation_matrix[thread_y][thread_x],
                    reduced_euler_rotation_matrix[y][x],
                    misrotation_matrix_cache[thread_y][thread_x]
                )

                if misrotation_angle <= minimum_misrotation_angle:
                    minimum_misrotation_angle = misrotation_angle
                    cluster_id_of_minimum = cluster_id[y][x]

    cluster_id[thread_y][thread_x] = cluster_id_of_minimum


@cuda.jit
def _reassign_unindexed_point_category(
    width: int,
    height: int,
    category: numpy.ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] == -1:
        category[thread_y][thread_x] = 0


@jit(nopython=True)
def _rotation_angle(R: numpy.ndarray) -> float:
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
def _misrotation_matrix(R1: numpy.ndarray, R2: numpy.ndarray) -> numpy.ndarray:
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
def _dbscan_cpu(
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
                # This point is an unindexed point.
                category[y][x] = -1

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
                            misrotation_angle = _rotation_angle(_misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

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
                        misrotation_angle = _rotation_angle(_misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

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
                                            misrotation_angle = _rotation_angle(_misrotation_matrix(rotation_matrix_1, rotation_matrix_2))

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
                            misrotation_angle = _rotation_angle(_misrotation_matrix(rotation_matrix_0, rotation_matrix_1))

                            if misrotation_angle <= minimum_misrotation_angle:
                                minimum_misrotation_angle = misrotation_angle
                                cluster_id_of_minimum = cluster_id[y1][x1]

                cluster_id[y0][x0] = cluster_id_of_minimum

    for y in range(height):
        for x in range(width):
            if category[y][x] == -1:
                category[y][x] = 0

    return cluster_count, category, cluster_id
