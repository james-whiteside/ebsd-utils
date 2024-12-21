# -*- coding: utf-8 -*-

from math import acos
from numpy import ndarray, array, zeros, dot
from numpy.linalg import inv
from numba import jit


def _dbscan_cpu(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
) -> tuple[int, ndarray, ndarray]:
    category = zeros((height, width))
    cluster_id = zeros((height, width))
    inverse_euler_rotation_matrix = zeros((height, width, 3, 3))

    _compute_inverted_matrices(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
    )

    _assign_unindexed_point_category(
        width,
        height,
        global_phase_id,
        category,
    )

    _assign_core_point_category(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
        core_point_neighbour_threshold,
        neighbourhood_radius,
        category,
    )

    _assign_border_point_category(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
        neighbourhood_radius,
        category,
    )

    _assign_noise_point_category(
        width,
        height,
        category,
    )

    cluster_count = _assign_core_point_cluster_id(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
        neighbourhood_radius,
        category,
        cluster_id,
    )

    _assign_border_point_cluster_id(
        width,
        height,
        global_phase_id,
        reduced_euler_rotation_matrix,
        inverse_euler_rotation_matrix,
        neighbourhood_radius,
        category,
        cluster_id,
    )

    _reassign_unindexed_point_category(
        width,
        height,
        category,
    )

    return cluster_count, category, cluster_id


@jit
def _misrotation_angle(inverse_matrix_1: ndarray, matrix_2: ndarray) -> float:
    matrix = dot(inverse_matrix_1, matrix_2)

    if 0.5 * (abs(matrix[0][0]) + abs(matrix[1][1]) + abs(matrix[2][2]) - 1) > 1:
        angle = acos(1)
    elif 0.5 * (abs(matrix[0][0]) + abs(matrix[1][1]) + abs(matrix[2][2]) - 1) < -1:
        angle = acos(-1)
    else:
        angle = acos(0.5 * (abs(matrix[0][0]) + abs(matrix[1][1]) + abs(matrix[2][2]) - 1))

    return angle


@jit
def _compute_inverted_matrices(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
) -> None:
    for y in range(height):
        for x in range(width):
            if global_phase_id[y][x] == 0:
                continue

            inverse_euler_rotation_matrix[y][x] = inv(reduced_euler_rotation_matrix[y][x])


@jit
def _assign_unindexed_point_category(
    width: int,
    height: int,
    global_phase_id: ndarray,
    category: ndarray,
) -> None:
    for iter_y in range(height):
        for iter_x in range(width):
            if global_phase_id[iter_y][iter_x] == 0:
                category[iter_y][iter_x] = -1


@jit
def _assign_core_point_category(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
    category: ndarray,
) -> None:
    for iter_y in range(height):
        for iter_x in range(width):
            if category[iter_y][iter_x] != 0:
                continue

            point_neighbours = 0

            for y in range(height):
                for x in range(width):
                    if iter_x == x and iter_y == y:
                        continue
                    elif global_phase_id[iter_y][iter_x] != global_phase_id[y][x]:
                        continue
                    else:
                        misrotation_angle = _misrotation_angle(
                            inverse_euler_rotation_matrix[iter_y][iter_x],
                            reduced_euler_rotation_matrix[y][x],
                        )

                        if misrotation_angle <= neighbourhood_radius and iter_x != x and iter_y != y:
                            point_neighbours += 1

                        if point_neighbours >= core_point_neighbour_threshold:
                            category[iter_y][iter_x] = 1
                            break

                if category[iter_y][iter_x] == 1:
                    break


@jit
def _assign_border_point_category(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
):
    for iter_y in range(height):
        for iter_x in range(width):
            if category[iter_y][iter_x] != 0:
                continue

            for y in range(height):
                for x in range(width):
                    if global_phase_id[iter_y][iter_x] == global_phase_id[y][x] and category[y][x] == 1:
                        misrotation_angle = _misrotation_angle(
                            inverse_euler_rotation_matrix[iter_y][iter_x],
                            reduced_euler_rotation_matrix[y][x],
                        )

                        if misrotation_angle <= neighbourhood_radius:
                            category[iter_y][iter_x] = 2
                            break

                if category[iter_y][iter_x] == 2:
                    break


@jit
def _assign_noise_point_category(
    width: int,
    height: int,
    category: ndarray,
) -> None:
    for iter_y in range(height):
        for iter_x in range(width):
            if category[iter_y][iter_x] == 0:
                category[iter_y][iter_x] = 3


@jit
def _assign_core_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
    cluster_id: ndarray,
) -> int:
    cluster_count = 0

    for y in range(height):
        for x in range(width):
            if category[y][x] == 1 and cluster_id[y][x] == 0:
                cluster_count += 1
                cluster_id[y][x] = cluster_count
                cluster_core_complete = array([False])

                while not cluster_core_complete[0]:
                    cluster_core_complete[0] = True

                    _grow_cluster_core(
                        width,
                        height,
                        global_phase_id,
                        reduced_euler_rotation_matrix,
                        inverse_euler_rotation_matrix,
                        neighbourhood_radius,
                        cluster_count,
                        category,
                        cluster_id,
                        cluster_core_complete,
                    )

    return cluster_count


@jit
def _grow_cluster_core(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    neighbourhood_radius: float,
    cluster_count: int,
    category: ndarray,
    cluster_id: ndarray,
    cluster_core_complete: ndarray,
):
    for iter_y in range(height):
        for iter_x in range(width):
            if cluster_id[iter_y][iter_x] != 0:
                continue

            for y in range(height):
                for x in range(width):
                    if cluster_id[y][x] == cluster_count:
                        if global_phase_id[y][x] == global_phase_id[iter_y][iter_x] and category[iter_y][iter_x] == 1:
                            misrotation_angle = _misrotation_angle(
                                inverse_euler_rotation_matrix[iter_y][iter_x],
                                reduced_euler_rotation_matrix[y][x],
                            )

                            if misrotation_angle <= neighbourhood_radius:
                                cluster_id[iter_y][iter_x] = cluster_count

                                if cluster_core_complete[0]:
                                    cluster_core_complete[0] = False


@jit
def _assign_border_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
    cluster_id: ndarray,
):
    for iter_y in range(height):
        for iter_x in range(width):
            if category[iter_y][iter_x] != 2:
                continue

            minimum_misrotation_angle = neighbourhood_radius
            cluster_id_of_minimum = 0

            for y in range(height):
                for x in range(width):
                    if global_phase_id[iter_y][iter_x] == global_phase_id[y][x] and category[y][x] == 1:
                        misrotation_angle = _misrotation_angle(
                            inverse_euler_rotation_matrix[iter_y][iter_x],
                            reduced_euler_rotation_matrix[y][x],
                        )

                        if misrotation_angle <= minimum_misrotation_angle:
                            minimum_misrotation_angle = misrotation_angle
                            cluster_id_of_minimum = cluster_id[y][x]

            cluster_id[iter_y][iter_x] = cluster_id_of_minimum


@jit
def _reassign_unindexed_point_category(
    width: int,
    height: int,
    category: ndarray,
) -> None:
    for iter_y in range(height):
        for iter_x in range(width):
            if category[iter_y][iter_x] == -1:
                category[iter_y][iter_x] = 0
