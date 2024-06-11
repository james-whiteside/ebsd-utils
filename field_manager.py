# -*- coding: utf-8 -*-

from numpy import ndarray, array, zeros, eye, dot
from field import FieldType, Field, DiscreteFieldMapper, FunctionalFieldMapper
from geometry import (
    Axis,
    euler_rotation_matrix,
    reduce_matrix,
    rotation_angle,
    misrotation_matrix,
    misrotation_tensor,
    forward_stereographic,
)
from phase import Phase, UNINDEXED_PHASE_ID
from channelling import load_crit_data, fraction
from clustering import ClusterCategory, dbscan
from parameter_groups import ScaleParameters, ChannellingParameters, ClusteringParameters, ScanParameters
from utilities import tuple_degrees, tuple_radians, float_degrees, float_radians, log_or_zero


GND_DENSITY_CORRECTIVE_FACTOR = 3.6


class FieldManager:
    def __init__(
        self,
        scan_parameters: ScanParameters,
        scale_parameters: ScaleParameters,
        channelling_parameters: ChannellingParameters,
        clustering_parameters: ClusteringParameters,
        phase_id_values: list[list[int]],
        euler_angle_degrees_values: list[list[tuple[float, float, float]]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
    ):
        self._scan_parameters = scan_parameters
        self._scale_parameters = scale_parameters
        self._channelling_parameters = channelling_parameters
        self._clustering_parameters = clustering_parameters
        self._phase_id = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.DISCRETE, values=phase_id_values)
        self.euler_angles: Field[tuple[float, float, float]] = None
        self.euler_angles_degrees = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.VECTOR_3D, values=euler_angle_degrees_values)
        self.pattern_quality = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.SCALAR, values=pattern_quality_values)
        self.index_quality = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.SCALAR, values=index_quality_values)
        self._reduced_euler_rotation_matrix: Field[ndarray] = None
        self._inverse_x_pole_figure_coordinates: Field[tuple[float, float]] = None
        self._inverse_y_pole_figure_coordinates: Field[tuple[float, float]] = None
        self._inverse_z_pole_figure_coordinates: Field[tuple[float, float]] = None
        self._kernel_average_misorientation: Field[float] = None
        self._misrotation_x_tensor: Field[ndarray] = None
        self._misrotation_y_tensor: Field[ndarray] = None
        self._nye_tensor: Field[ndarray] = None
        self._geometrically_necessary_dislocation_density: Field[float] = None
        self._channelling_fraction: Field[float] = None
        self._cluster_count: int = None
        self._orientation_clustering_category_id: Field[int] = None
        self._orientation_cluster_id: Field[int] = None

    @property
    def phase(self) -> DiscreteFieldMapper[Phase]:
        return DiscreteFieldMapper(FieldType.OBJECT, self._phase_id, self._scan_parameters.phases)

    @property
    def euler_angles_degrees(self) -> FunctionalFieldMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalFieldMapper(FieldType.VECTOR_3D, self.euler_angles, tuple_degrees, tuple_radians)

    @euler_angles_degrees.setter
    def euler_angles_degrees(self, value: Field[tuple[float, float, float]]) -> None:
        euler_angle_values = list()

        for y in range(self._scan_parameters.height):
            euler_angle_values.append(list())

            for x in range(self._scan_parameters.width):
                euler_angle_values[y].append(tuple_radians(value.get_value_at(x, y)))

        self.euler_angles = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.VECTOR_3D, values=euler_angle_values)

    @property
    def reduced_euler_rotation_matrix(self) -> Field[ndarray]:
        if self._reduced_euler_rotation_matrix is None:
            self._init_reduced_euler_rotation_matrices()

        return self._reduced_euler_rotation_matrix

    def inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if None in (
            self._inverse_x_pole_figure_coordinates,
            self._inverse_y_pole_figure_coordinates,
            self._inverse_z_pole_figure_coordinates,
        ):
            self._init_inverse_pole_figure_coordinates()

        match axis:
            case Axis.X:
                return self._inverse_x_pole_figure_coordinates
            case Axis.Y:
                return self._inverse_y_pole_figure_coordinates
            case Axis.Z:
                return self._inverse_z_pole_figure_coordinates

    @property
    def kernel_average_misorientation(self) -> Field[float]:
        if self._kernel_average_misorientation is None:
            self._init_kernel_average_misorientation()

        return self._kernel_average_misorientation

    @property
    def kernel_average_misorientation_degrees(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.kernel_average_misorientation, float_degrees, float_radians)

    def misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        if None in (self._misrotation_x_tensor, self._misrotation_y_tensor):
            self._init_misrotation_tensors()

        match axis:
            case Axis.X:
                return self._misrotation_x_tensor
            case Axis.Y:
                return self._misrotation_y_tensor
            case Axis.Z:
                raise ValueError("Misrotation data not available for z-axis intervals.")

    @property
    def nye_tensor(self) -> Field[ndarray]:
        if self._nye_tensor is None:
            self._init_nye_tensor()

        return self._nye_tensor

    @property
    def geometrically_necessary_dislocation_density(self) -> Field[float]:
        if self._geometrically_necessary_dislocation_density is None:
            self._init_geometrically_necessary_dislocation_density()

        return self._geometrically_necessary_dislocation_density

    @property
    def geometrically_necessary_dislocation_density_logarithmic(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.geometrically_necessary_dislocation_density, log_or_zero)

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            self._init_channelling_fraction()

        return self._channelling_fraction

    def _get_cluster_count(self) -> int:
        if self._cluster_count is None:
            self._init_orientation_cluster()

        return self._cluster_count

    @property
    def orientation_clustering_category(self) -> DiscreteFieldMapper[ClusterCategory]:
        if self._orientation_clustering_category_id is None:
            self._init_orientation_cluster()

        mapping = {category.value: category for category in ClusterCategory}
        return DiscreteFieldMapper(FieldType.OBJECT, self._orientation_clustering_category_id, mapping)

    @property
    def orientation_cluster_id(self) -> Field[int]:
        if self._orientation_cluster_id is None:
            self._init_orientation_cluster()

        return self._orientation_cluster_id

    def _init_reduced_euler_rotation_matrices(self) -> None:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.MATRIX, default_value=eye(3))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                axis_set = self._scan_parameters.axis_set
                euler_angles = self.euler_angles.get_value_at(x, y)
                crystal_family = self.phase.get_value_at(x, y).lattice_type.get_family()
                value = reduce_matrix(euler_rotation_matrix(axis_set, euler_angles), crystal_family)
                field.set_value_at(x, y, value)

        self._reduced_euler_rotation_matrix = field

    def _gen_inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.VECTOR_2D, default_value=(0.0, 0.0))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    value = forward_stereographic(*dot(rotation_matrix, array(axis.value)).tolist())
                    field.set_value_at(x, y, value)

        return field

    def _init_inverse_pole_figure_coordinates(self) -> None:
        self._inverse_x_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.X)
        self._inverse_y_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Y)
        self._inverse_z_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Z)

    def _init_kernel_average_misorientation(self) -> None:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.SCALAR, default_value=0.0)

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    total = 0.0
                    count = 4
                    kernel = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
                    rotation_matrix_1 = self.reduced_euler_rotation_matrix.get_value_at(x, y)

                    for dx, dy in kernel:
                        try:
                            if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                                rotation_matrix_2 = self.reduced_euler_rotation_matrix.get_value_at(x + dx, y + dy)
                                total += rotation_angle(misrotation_matrix(rotation_matrix_1, rotation_matrix_2))
                            else:
                                count -= 1
                        except IndexError:
                            count -= 1

                    if count == 0:
                        continue
                    else:
                        value = total / count
                        field.set_value_at(x, y, value)

        self._kernel_average_misorientation = field

    def _gen_misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.MATRIX, default_value=zeros((3, 3)))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    total = zeros((3, 3))
                    count = 2

                    match axis:
                        case Axis.X:
                            kernel = [(-1, 0), (+1, 0)]
                        case Axis.Y:
                            kernel = [(0, -1), (0, +1)]
                        case Axis.Z:
                            raise ValueError("Misrotation data not available for z-axis intervals.")

                    rotation_matrix_1 = self.reduced_euler_rotation_matrix.get_value_at(x, y)

                    for dx, dy in kernel:
                        try:
                            if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                                rotation_matrix_2 = self.reduced_euler_rotation_matrix.get_value_at(x + dx, y + dy)
                                total += misrotation_tensor(misrotation_matrix(rotation_matrix_1, rotation_matrix_2), self._scale_parameters.pixel_size)
                            else:
                                count -= 1
                        except IndexError:
                            count -= 1

                    if count == 0:
                        continue
                    else:
                        value = total / count
                        field.set_value_at(x, y, value)

        return field

    def _init_misrotation_tensors(self) -> None:
        self._misrotation_x_tensor = self._gen_misrotation_tensor(Axis.X)
        self._misrotation_y_tensor = self._gen_misrotation_tensor(Axis.Y)

    def _init_nye_tensor(self) -> None:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.MATRIX, default_value=zeros((3, 3)))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    value = array((
                        (0.0, self.misrotation_tensor(Axis.X).get_value_at(x, y)[2][0], -self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][0]),
                        (-self.misrotation_tensor(Axis.Y).get_value_at(x, y)[2][1], 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][1]),
                        (0.0, 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][2] - self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][2])
                    ))

                    field.set_value_at(x, y, value)

        self._nye_tensor = field

    def _init_geometrically_necessary_dislocation_density(self) -> None:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.SCALAR, default_value=0.0)

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    nye_tensor_norm = sum(abs(element) for row in self.nye_tensor.get_value_at(x, y).tolist() for element in row)
                    close_pack_distance = self.phase.get_value_at(x, y).close_pack_distance
                    value = (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm
                    # value = 0.25 * (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm ** 2
                    field.set_value_at(x, y, value)

        self._geometrically_necessary_dislocation_density = field

    def _init_channelling_fraction(self) -> None:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.SCALAR, default_value=0.0)

        channel_data = {
            local_id: load_crit_data(self._channelling_parameters.beam_atomic_number, phase.global_id, self._channelling_parameters.beam_energy)
            for local_id, phase in self._scan_parameters.phases.items() if phase.global_id != UNINDEXED_PHASE_ID
        }

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    effective_beam_vector = dot(rotation_matrix, self._channelling_parameters.beam_vector).tolist()
                    value = fraction(effective_beam_vector, channel_data[self._phase_id.get_value_at(x, y)])
                    field.set_value_at(x, y, value)

        self._channelling_fraction = field

    def _init_orientation_cluster(self) -> None:
        phase = zeros((self._scan_parameters.height, self._scan_parameters.width))
        reduced_euler_rotation_matrix = zeros((self._scan_parameters.height, self._scan_parameters.width, 3, 3))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                phase[y][x] = self.phase.get_value_at(x, y).global_id
                reduced_euler_rotation_matrix[y][x] = self.reduced_euler_rotation_matrix.get_value_at(x, y)

        cluster_count, category_id_array, cluster_id_array = dbscan(
            self._scan_parameters.width,
            self._scan_parameters.height,
            phase,
            reduced_euler_rotation_matrix,
            self._clustering_parameters.core_point_neighbour_threshold,
            self._clustering_parameters.neighbourhood_radius
        )

        self._cluster_count = cluster_count
        self._orientation_clustering_category_id = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.DISCRETE, values=category_id_array.astype(int).tolist())
        self._orientation_cluster_id = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.DISCRETE, values=cluster_id_array.astype(int).tolist())
