# -*- coding: utf-8 -*-

from numba import njit, prange
from numpy import ndarray, full, array_equal, int32

from src.data_structures.field import FieldLike
from src.data_structures.fields.discrete_field import DiscreteField
from src.data_structures.fields.enum_field import EnumField
from src.data_structures.phase import Phase, BravaisLattice


def lattice_type(
        phase_id_field: DiscreteField,
        phases: dict[int, Phase],
        parallel: bool = True,
) -> EnumField[BravaisLattice]:
    input_fields = [phase_id_field]
    width, height, nullable = FieldLike.get_combined_params(input_fields)
    lattice_types = full(max(phases.keys()) + 1, -1, dtype=int32)

    for phase_id, phase in phases.items():
        lattice_types[phase_id] = phase.lattice_type.value

    if parallel:
        values = _lattice_type_parallel(phase_id_field.array, lattice_types, phase_id_field.null_value)
    else:
        values = _lattice_type(phase_id_field.array, lattice_types, phase_id_field.null_value)

    return EnumField.from_array(width, height, BravaisLattice, values, nullable)


@njit
def _lattice_type(phase_id_field: ndarray, lattice_types: ndarray, null_value: ndarray) -> ndarray:
    width, height = phase_id_field.shape
    output = full((width, height), -1, dtype=int32)

    for x in range(width):
        for y in range(height):
            phase_id = phase_id_field[x, y]

            if array_equal(phase_id, null_value):
                pass
            else:
                output[x, y] = lattice_types[phase_id]

    return output


@njit(parallel=True)
def _lattice_type_parallel(phase_id_field: ndarray, lattice_types: ndarray, null_value: ndarray) -> ndarray:
    width, height = phase_id_field.shape
    output = full((width, height), -1, dtype=int32)

    for x in prange(width):
        for y in range(height):
            phase_id = phase_id_field[x, y]

            if array_equal(phase_id, null_value):
                pass
            else:
                output[x, y] = lattice_types[phase_id]

    return output
