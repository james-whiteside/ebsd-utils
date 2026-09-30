# -*- coding: utf-8 -*-

from numba import njit, prange
from numpy import ndarray, full, array_equal, int32

from src.data_structures.field import FieldLike
from src.data_structures.fields.enum_field import EnumField
from src.data_structures.phase import BravaisLattice, CrystalFamily


def crystal_family(
        lattice_type_field: EnumField[BravaisLattice],
        parallel: bool = True,
) -> EnumField[CrystalFamily]:
    input_fields = [lattice_type_field]
    width, height, nullable = FieldLike.get_combined_params(input_fields)
    crystal_families = full(max(lattice_type.value for lattice_type in BravaisLattice) + 1, -1, dtype=int32)

    for lattice_type in BravaisLattice:
        crystal_families[lattice_type.value] = lattice_type.family.value

    if parallel:
        values = _crystal_family_parallel(lattice_type_field.array, crystal_families, lattice_type_field.null_value)
    else:
        values = _crystal_family(lattice_type_field.array, crystal_families, lattice_type_field.null_value)

    return EnumField.from_array(width, height, BravaisLattice, values, nullable)


@njit
def _crystal_family(lattice_type_field: ndarray, crystal_families: ndarray, null_value: ndarray) -> ndarray:
    width, height = lattice_type_field.shape
    output = full((width, height), -1, dtype=int32)

    for x in range(width):
        for y in range(height):
            lattice_type = lattice_type_field[x, y]

            if array_equal(lattice_type, null_value):
                pass
            else:
                output[x, y] = crystal_families[lattice_type]

    return output


@njit(parallel=True)
def _crystal_family_parallel(lattice_type_field: ndarray, crystal_families: ndarray, null_value: ndarray) -> ndarray:
    width, height = lattice_type_field.shape
    output = full((width, height), -1, dtype=int32)

    for x in prange(width):
        for y in range(height):
            lattice_type = lattice_type_field[x, y]

            if array_equal(lattice_type, null_value):
                pass
            else:
                output[x, y] = crystal_families[lattice_type]

    return output
