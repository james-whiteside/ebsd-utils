# -*- coding: utf-8 -*-

from math import sqrt
from random import Random
from numpy import ndarray, array, dot, zeros
from src.algorithms.channelling import load_crit_data, fraction
from src.algorithms.clustering.dbscan import dbscan
from src.data_structures.field import FieldLike, FieldNullError, FieldType, Field
from src.data_structures.phase import Phase, CrystalFamily
from src.utilities.geometry import (
    Axis,
    AxisSet,
    euler_rotation_matrix,
    misrotation_matrix,
    rotation_angle,
    misrotation_tensor as misrotation_tensor_,
    reduce_vector,
)
from src.utilities.utils import maximise_brightness


def orientation_matrix(
    axis_set: AxisSet,
    euler_angle_field: FieldLike[tuple[float, float, float]],
) -> Field[ndarray]:
    input_fields = [euler_angle_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.MATRIX, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            try:
                euler_angles = euler_angle_field.get_value_at(x, y)
            except FieldNullError:
                continue

            value = euler_rotation_matrix(axis_set, euler_angles)
            output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def reduced_matrix(
    orientation_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[ndarray]:
    input_fields = [orientation_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.MATRIX, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            try:
                orientation_matrix = orientation_matrix_field.get_value_at(x, y)
                crystal_family = phase_field.get_value_at(x, y).lattice_type.family
            except FieldNullError:
                continue

            value = crystal_family.reduce_matrix(orientation_matrix)
            output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def ipf_coordinates(
    axis: Axis,
    reduced_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[tuple[float, float]]:
    input_fields = [reduced_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.VECTOR_2D, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            try:
                reduced_matrix = reduced_matrix_field.get_value_at(x, y)
                crystal_family = phase_field.get_value_at(x, y).lattice_type.family
            except FieldNullError:
                continue

            vector = dot(reduced_matrix, array(axis.vector)).tolist()
            value = crystal_family.ipf_coordinates(vector)
            output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def average_misorientation(
    reduced_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[float]:
    input_fields = [reduced_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.SCALAR, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            kernel = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
            total = 0.0
            count = 0

            try:
                rotation_matrix_1 = reduced_matrix_field.get_value_at(x, y)
            except FieldNullError:
                continue

            for dx, dy in kernel:
                try:
                    if phase_field.get_value_at(x, y) == phase_field.get_value_at(x + dx, y + dy):
                        rotation_matrix_2 = reduced_matrix_field.get_value_at(x + dx, y + dy)
                        total += rotation_angle(misrotation_matrix(rotation_matrix_1, rotation_matrix_2))
                        count += 1
                except (IndexError, FieldNullError):
                    continue

            if count == 0:
                continue
            else:
                value = total / count
                output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def misrotation_tensor(
    axis: Axis,
    pixel_size: float,
    reduced_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[ndarray]:
    input_fields = [reduced_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.MATRIX, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            match axis:
                case Axis.X:
                    kernel = [(-1, 0), (+1, 0)]
                case Axis.Y:
                    kernel = [(0, -1), (0, +1)]
                case Axis.Z:
                    raise ValueError("Misrotation data not available for z-axis intervals.")
                case _:
                    raise ValueError("Non-Cartesian axes are not valid for misrotation data.")

            total = zeros((3, 3))
            count = 0

            try:
                rotation_matrix_1 = reduced_matrix_field.get_value_at(x, y)
            except FieldNullError:
                continue

            for dx, dy in kernel:
                try:
                    if phase_field.get_value_at(x, y) == phase_field.get_value_at(x + dx, y + dy):
                        rotation_matrix_2 = reduced_matrix_field.get_value_at(x + dx, y + dy)
                        total += misrotation_tensor_(misrotation_matrix(rotation_matrix_1, rotation_matrix_2), pixel_size)
                        count += 1
                except (IndexError, FieldNullError):
                    continue

            if count == 0:
                continue
            else:
                value = total / count
                output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def nye_tensor(
    misrotation_x_tensor_field: FieldLike[ndarray],
    misrotation_y_tensor_field: FieldLike[ndarray],
) -> Field[ndarray]:
    input_fields = [misrotation_x_tensor_field, misrotation_y_tensor_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.MATRIX, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            value = zeros((3, 3))
            count = 0

            try:
                value += array((
                    (0.0, misrotation_x_tensor_field.get_value_at(x, y)[2][0], -misrotation_x_tensor_field.get_value_at(x, y)[1][0]),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, -misrotation_x_tensor_field.get_value_at(x, y)[1][2])
                ))

                count += 1
            except FieldNullError:
                pass

            try:
                value += array((
                    (0.0, 0.0, 0.0),
                    (-misrotation_y_tensor_field.get_value_at(x, y)[2][1], 0.0, misrotation_y_tensor_field.get_value_at(x, y)[0][1]),
                    (0.0, 0.0, misrotation_y_tensor_field.get_value_at(x, y)[0][2])
                ))

                count += 1
            except FieldNullError:
                pass

            if count == 0:
                continue
            else:
                output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def gnd_density(
    corrective_factor: float,
    nye_tensor_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[float]:
    input_fields = [nye_tensor_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.SCALAR, default_value=None, nullable=True)

    for y in range(height):
        for x in range(width):
            try:
                nye_tensor_norm = sum(abs(element) for row in nye_tensor_field.get_value_at(x, y).tolist() for element in row)
                close_pack_distance = phase_field.get_value_at(x, y).close_pack_distance_nm
            except FieldNullError:
                continue

            value = (corrective_factor / close_pack_distance) * nye_tensor_norm
            output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def channelling_fraction(
    beam_atomic_number: int,
    beam_energy: float,
    beam_vector: tuple[float, float, float],
    phases: dict[int, Phase],
    orientation_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
    random_source: Random,
    use_cache: bool,
    cache_dir: str,
    phase_dir: str,
    phase_database_path: str,
) -> Field[float]:
    input_fields = [orientation_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.SCALAR, default_value=None, nullable=True)

    channel_data = {
        phase.global_id: load_crit_data(
            beam_atomic_number=beam_atomic_number,
            target_id=phase.global_id,
            beam_energy=beam_energy,
            random_source=random_source,
            use_cache=use_cache,
            cache_dir=cache_dir,
            phase_dir=phase_dir,
            phase_database_path=phase_database_path,
        ) for local_id, phase in phases.items() if phase.global_id != Phase.UNINDEXED_ID
    }

    for y in range(height):
        for x in range(width):
            try:
                rotation_matrix = orientation_matrix_field.get_value_at(x, y)
                phase_data = channel_data[phase_field.get_value_at(x, y).global_id]
            except FieldNullError:
                continue

            effective_beam_vector = dot(rotation_matrix, beam_vector).tolist()
            value = fraction(effective_beam_vector, phase_data)
            output_field.set_value_at(x, y, value)

    output_field.nullable = nullable
    return output_field


def orientation_cluster(
    core_point_threshold: int,
    neighbourhood_radius: float,
    phase_field: FieldLike[Phase],
    reduced_matrix_field: FieldLike[ndarray],
    use_cuda: bool,
) -> tuple[int, Field[int], Field[int]]:
    input_fields = [phase_field, reduced_matrix_field]
    width, height, nullable = FieldLike.get_params(input_fields)

    phase = zeros((height, width))
    reduced_euler_rotation_matrix = zeros((height, width, 3, 3))

    for y in range(height):
        for x in range(width):
            try:
                phase[y][x] = phase_field.get_value_at(x, y).global_id
                reduced_euler_rotation_matrix[y][x] = reduced_matrix_field.get_value_at(x, y)
            except FieldNullError:
                pass

    cluster_count, category_id_array, cluster_id_array = dbscan(
        width,
        height,
        phase,
        reduced_euler_rotation_matrix,
        core_point_threshold,
        neighbourhood_radius,
        use_cuda,
    )

    category_id_values = category_id_array.astype(int).tolist()
    cluster_id_values = cluster_id_array.astype(int).tolist()

    for y in range(height):
        for x in range(width):
            if category_id_values[y][x] == 0:
                category_id_values[y][x] = None

            if cluster_id_values[y][x] == 0:
                cluster_id_values[y][x] = None

    cluster_count_result = cluster_count
    clustering_category_id = Field.from_array(width, height, FieldType.DISCRETE, category_id_values, nullable)
    orientation_cluster_id = Field.from_array(width, height, FieldType.DISCRETE, cluster_id_values, nullable=True)
    return cluster_count_result, clustering_category_id, orientation_cluster_id


def euler_angle_colours(
    euler_angle_field: FieldLike[tuple[float, float, float]],
    phase_field: FieldLike[Phase],
) -> Field[tuple[float, float, float]]:
    input_fields = [euler_angle_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

    for y in range(height):
        for x in range(width):
            try:
                euler_angles = euler_angle_field.get_value_at(x, y)
                max_euler_angles = phase_field.get_value_at(x, y).lattice_type.family.max_euler_angles
            except FieldNullError:
                continue

            value = (
                euler_angles[0] / max_euler_angles[0],
                euler_angles[1] / max_euler_angles[1],
                euler_angles[2] / max_euler_angles[2],
            )

            output_field.set_value_at(x, y, value)

    return output_field


def ipf_colours(
    axis: Axis,
    reduced_matrix_field: FieldLike[ndarray],
    phase_field: FieldLike[Phase],
) -> Field[tuple[float, float, float]]:
    input_fields = [reduced_matrix_field, phase_field]
    width, height, nullable = FieldLike.get_params(input_fields)
    output_field = Field(width, height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

    for y in range(height):
        for x in range(width):
            try:
                rotation_matrix = reduced_matrix_field.get_value_at(x, y)
                crystal_family = phase_field.get_value_at(x, y).lattice_type.family
            except FieldNullError:
                continue

            match crystal_family:
                case CrystalFamily.C:
                    vector = dot(rotation_matrix, array(axis.vector)).tolist()
                    u, v, w = reduce_vector(vector)
                    r, g, b = w - v, (v - u) * sqrt(2), u * sqrt(3)
                    value = maximise_brightness((r, g, b))
                case _:
                    raise NotImplementedError()

            output_field.set_value_at(x, y, value)

    return output_field
