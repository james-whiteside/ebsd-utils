# -*- coding: utf-8 -*-

from numba import njit, prange
from numpy import ndarray, full, nan, float64, array_equal, isnan

from src.data_structures.field import FieldLike
from src.data_structures.fields.enum_field import EnumField
from src.data_structures.fields.matrix_field import MatrixField
from src.data_structures.phase import CrystalFamily, reduce_matrix, SymmetryNotImplementedError


_NO_UNIMPLEMENTED_SYMMETRY = -1


def reduced_matrix(
        orientation_matrix_field: MatrixField,
        crystal_family_field: EnumField[CrystalFamily],
        parallel: bool = True,
) -> MatrixField:
    input_fields = [orientation_matrix_field, crystal_family_field]
    width, height, nullable = FieldLike.get_combined_params(input_fields)

    if parallel:
        values, unimplemented_symmetry = _reduced_matrix_parallel(
            orientation_matrix_field.array,
            crystal_family_field.array,
            orientation_matrix_field.null_value,
            crystal_family_field.null_value
        )
    else:
        values, unimplemented_symmetry = _reduced_matrix(
            orientation_matrix_field.array,
            crystal_family_field.array,
            orientation_matrix_field.null_value,
            crystal_family_field.null_value
        )

    if unimplemented_symmetry == _NO_UNIMPLEMENTED_SYMMETRY:
        return MatrixField.from_array(width, height, values, nullable)
    else:
        raise SymmetryNotImplementedError(CrystalFamily(unimplemented_symmetry))


@njit
def _reduced_matrix(
        orientation_matrix_field: ndarray,
        crystal_family_field: ndarray,
        orientation_matrix_null_value: ndarray,
        crystal_family_null_value: ndarray,
) -> tuple[ndarray, int]:
    width, height, _, _ = orientation_matrix_field.shape
    output = full((width, height, 3, 3), nan, dtype=float64)
    unimplemented_symmetry = _NO_UNIMPLEMENTED_SYMMETRY

    for y in range(height):
        for x in range(width):
            orientation_matrix = orientation_matrix_field[x, y]
            crystal_family = crystal_family_field[x, y]

            if isnan(orientation_matrix).any() or array_equal(orientation_matrix, orientation_matrix_null_value):
                pass
            elif array_equal(crystal_family, crystal_family_null_value):
                pass
            else:
                output[x, y], symmetry_not_implemented = reduce_matrix(int(crystal_family), orientation_matrix)

                if symmetry_not_implemented:
                     unimplemented_symmetry = int(crystal_family)

    return output, unimplemented_symmetry


@njit(parallel=True)
def _reduced_matrix_parallel(
        orientation_matrix_field: ndarray,
        crystal_family_field: ndarray,
        orientation_matrix_null_value: ndarray,
        crystal_family_null_value: ndarray,
) -> tuple[ndarray, int]:
    width, height, _, _ = orientation_matrix_field.shape
    output = full((width, height, 3, 3), nan, dtype=float64)
    unimplemented_symmetry = _NO_UNIMPLEMENTED_SYMMETRY

    for y in prange(height):
        for x in range(width):
            orientation_matrix = orientation_matrix_field[x, y]
            crystal_family = crystal_family_field[x, y]

            if isnan(orientation_matrix).any() or array_equal(orientation_matrix, orientation_matrix_null_value):
                pass
            elif array_equal(crystal_family, crystal_family_null_value):
                pass
                # TODO: eliminate temporary array creation during field transforms
            else:
                output[x, y], symmetry_not_implemented = reduce_matrix(int(crystal_family), orientation_matrix)

                if symmetry_not_implemented:
                     unimplemented_symmetry = int(crystal_family)

    return output, unimplemented_symmetry
