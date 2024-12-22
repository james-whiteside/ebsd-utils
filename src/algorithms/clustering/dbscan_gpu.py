# -*- coding: utf-8 -*-

from math import ceil, acos
from numpy import ndarray, array, zeros
from numpy.linalg import inv
from numba import cuda, jit


def _dbscan_gpu(
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
    misrotation_matrix_diagonal_cache = zeros((height, width, 3))

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
    misrotation_matrix_diagonal_cache_ = cuda.to_device(misrotation_matrix_diagonal_cache)

    block_size = (16, 16)
    grid_size = (ceil(width / block_size[0]), ceil(height / block_size[1]))

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
        misrotation_matrix_diagonal_cache_,
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
        misrotation_matrix_diagonal_cache_,
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
        misrotation_matrix_diagonal_cache_,
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
        misrotation_matrix_diagonal_cache_,
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
def _misrotation_angle(inverse_matrix_1: ndarray, matrix_2: ndarray, diagonal_cache: ndarray) -> float:
    for i in range(3):
        diagonal_cache[i] = inverse_matrix_1[i][0] * matrix_2[0][i]
        diagonal_cache[i] += inverse_matrix_1[i][1] * matrix_2[1][i]
        diagonal_cache[i] += inverse_matrix_1[i][2] * matrix_2[2][i]

    trace_metric = 0.5 * (abs(diagonal_cache[0]) + abs(diagonal_cache[1]) + abs(diagonal_cache[2]) - 1)

    if trace_metric > 1:
        angle = acos(1)
    elif trace_metric < -1:
        angle = acos(-1)
    else:
        angle = acos(trace_metric)

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


@cuda.jit
def _assign_unindexed_point_category(
    width: int,
    height: int,
    global_phase_id: ndarray,
    category: ndarray,
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
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    misrotation_matrix_diaognal_cache: ndarray,
    core_point_neighbour_threshold: int,
    neighbourhood_radius: float,
    category: ndarray,
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
                    misrotation_matrix_diaognal_cache[thread_y][thread_x],
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
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    misrotation_matrix_diagonal_cache: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
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
                    misrotation_matrix_diagonal_cache[thread_y][thread_x],
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
    category: ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] == 0:
        category[thread_y][thread_x] = 3


def _assign_core_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    misrotation_matrix_diagonal_cache: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
    cluster_id: ndarray,
    grid_size: tuple[int, int],
    block_size: tuple[int, int],
) -> int:
    cluster_count = 0

    for y in range(height):
        for x in range(width):
            if category[y][x] == 1 and cluster_id[y][x] == 0:
                cluster_count += 1
                cluster_id[y][x] = cluster_count
                cluster_core_complete = cuda.to_device(array([False]))

                while not cluster_core_complete[0]:
                    cluster_core_complete[0] = True

                    _grow_cluster_core[grid_size, block_size](
                        width,
                        height,
                        global_phase_id,
                        reduced_euler_rotation_matrix,
                        inverse_euler_rotation_matrix,
                        misrotation_matrix_diagonal_cache,
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
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    misrotation_matrix_diagonal_cache: ndarray,
    neighbourhood_radius: float,
    cluster_count: int,
    category: ndarray,
    cluster_id: ndarray,
    cluster_core_complete: ndarray,
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
                        misrotation_matrix_diagonal_cache[thread_y][thread_x],
                    )

                    if misrotation_angle <= neighbourhood_radius:
                        cluster_id[thread_y][thread_x] = cluster_count

                        if cluster_core_complete[0]:
                            cluster_core_complete[0] = False


@cuda.jit
def _assign_border_point_cluster_id(
    width: int,
    height: int,
    global_phase_id: ndarray,
    reduced_euler_rotation_matrix: ndarray,
    inverse_euler_rotation_matrix: ndarray,
    misrotation_matrix_diagonal_cache: ndarray,
    neighbourhood_radius: float,
    category: ndarray,
    cluster_id: ndarray,
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
                    misrotation_matrix_diagonal_cache[thread_y][thread_x]
                )

                if misrotation_angle <= minimum_misrotation_angle:
                    minimum_misrotation_angle = misrotation_angle
                    cluster_id_of_minimum = cluster_id[y][x]

    cluster_id[thread_y][thread_x] = cluster_id_of_minimum


@cuda.jit
def _reassign_unindexed_point_category(
    width: int,
    height: int,
    category: ndarray,
) -> None:
    thread_x, thread_y = cuda.grid(2)

    if thread_x >= width or thread_y >= height:
        return

    if category[thread_y][thread_x] == -1:
        category[thread_y][thread_x] = 0
