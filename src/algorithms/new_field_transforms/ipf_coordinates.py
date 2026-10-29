# -*- coding: utf-8 -*-

from numba import njit, prange
from numpy import ndarray, full, nan, float64, array_equal, isnan, dot

from src.data_structures.field import FieldLike
from src.data_structures.fields.enum_field import EnumField
from src.data_structures.fields.matrix_field import MatrixField
from src.data_structures.fields.vector_2d_field import Vector2DField
from src.data_structures.phase import CrystalFamily, vector_ipf_coordinates, SymmetryNotImplementedError
from src.utilities.geometry import Axis


_NO_UNIMPLEMENTED_SYMMETRY = -1


def ipf_coordinates(
        axis: Axis,
        reduced_matrix_field: MatrixField,
        crystal_family_field: EnumField[CrystalFamily],
        parallel: bool = True,
) -> Vector2DField:
    input_fields = [reduced_matrix_field, crystal_family_field]
    width, height, nullable = FieldLike.get_combined_params(input_fields)

    if parallel:
        values, unimplemented_symmetry = _ipf_coordinates_parallel(
            axis.vector,
            reduced_matrix_field.array,
            crystal_family_field.array,
            reduced_matrix_field.null_value,
            crystal_family_field.null_value,
        )
    else:
        values, unimplemented_symmetry = _ipf_coordinates(
            axis.vector,
            reduced_matrix_field.array,
            crystal_family_field.array,
            reduced_matrix_field.null_value,
            crystal_family_field.null_value,
        )

    if unimplemented_symmetry == _NO_UNIMPLEMENTED_SYMMETRY:
        return Vector2DField.from_array(width, height, values, nullable)
    else:
        raise SymmetryNotImplementedError(CrystalFamily(unimplemented_symmetry))


@njit
def _ipf_coordinates(
        axis: ndarray,
        reduced_matrix_field: ndarray,
        crystal_family_field: ndarray,
        reduced_matrix_null_value: ndarray,
        crystal_family_null_value: ndarray,
) -> tuple[ndarray, int]:
    width, height, _, _ = reduced_matrix_field.shape
    output = full((width, height, 2), nan, dtype=float64)
    unimplemented_symmetry = _NO_UNIMPLEMENTED_SYMMETRY

    for x in range(width):
        for y in range(height):
            reduced_matrix = reduced_matrix_field[x, y]
            crystal_family = crystal_family_field[x, y]

            if isnan(reduced_matrix).any() or array_equal(reduced_matrix, reduced_matrix_null_value):
                pass
            elif array_equal(crystal_family, crystal_family_null_value):
                pass
            else:
                vector = dot(reduced_matrix, axis)
                output[x, y], symmetry_not_implemented = vector_ipf_coordinates(int(crystal_family), vector)

                if symmetry_not_implemented:
                    unimplemented_symmetry = int(crystal_family)

    return output, unimplemented_symmetry


@njit(parallel=True)
def _ipf_coordinates_parallel(
        axis: ndarray,
        reduced_matrix_field: ndarray,
        crystal_family_field: ndarray,
        reduced_matrix_null_value: ndarray,
        crystal_family_null_value: ndarray,
) -> tuple[ndarray, int]:
    width, height, _, _ = reduced_matrix_field.shape
    output = full((width, height, 2), nan, dtype=float64)
    unimplemented_symmetry = _NO_UNIMPLEMENTED_SYMMETRY

    for x in prange(width):
        for y in range(height):
            reduced_matrix = reduced_matrix_field[x, y]
            crystal_family = crystal_family_field[x, y]

            if isnan(reduced_matrix).any() or array_equal(reduced_matrix, reduced_matrix_null_value):
                pass
            elif array_equal(crystal_family, crystal_family_null_value):
                pass
            else:
                vector = dot(reduced_matrix, axis)
                output[x, y], symmetry_not_implemented = vector_ipf_coordinates(int(crystal_family), vector)

                if symmetry_not_implemented:
                    unimplemented_symmetry = int(crystal_family)

    return output, unimplemented_symmetry
