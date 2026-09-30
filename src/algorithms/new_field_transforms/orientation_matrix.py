# -*- coding: utf-8 -*-

from numba import njit, prange
from numpy import ndarray, full, nan, float64, array_equal, isnan

from src.data_structures.field import FieldLike
from src.data_structures.fields.matrix_field import MatrixField
from src.data_structures.fields.vector_3d_field import Vector3DField
from src.utilities.geometry import AxisSet, euler_rotation_matrix


def orientation_matrix(
        axis_set: AxisSet,
        euler_angle_field: Vector3DField,
        parallel: bool = True,
) -> MatrixField:
    input_fields = [euler_angle_field]
    width, height, nullable = FieldLike.get_combined_params(input_fields)

    if parallel:
        values = _orientation_matrix_parallel(axis_set.vectors, euler_angle_field.array, euler_angle_field.null_value)
    else:
        values = _orientation_matrix(axis_set.vectors, euler_angle_field.array, euler_angle_field.null_value)

    return MatrixField.from_array(width, height, values, nullable)


@njit
def _orientation_matrix(axis_set: tuple[ndarray, ndarray, ndarray], euler_angle_field: ndarray, null_value: ndarray) -> ndarray:
    width, height, _ = euler_angle_field.shape
    output = full((width, height, 3, 3), nan, dtype=float64)

    for x in range(width):
        for y in range(height):
            euler_angles = euler_angle_field[x, y]

            if isnan(euler_angles).any() or array_equal(euler_angles, null_value):
                pass
            else:
                output[x, y] = euler_rotation_matrix(axis_set, euler_angles)

    return output


@njit(parallel=True)
def _orientation_matrix_parallel(axis_set: tuple[ndarray, ndarray, ndarray], euler_angle_field: ndarray, null_value: ndarray) -> ndarray:
    width, height, _ = euler_angle_field.shape
    output = full((width, height, 3, 3), nan, dtype=float64)

    for x in prange(width):
        for y in range(height):
            euler_angles = euler_angle_field[x, y]

            if isnan(euler_angles).any() or array_equal(euler_angles, null_value):
                pass
            else:
                output[x, y] = euler_rotation_matrix(axis_set, euler_angles)

    return output
