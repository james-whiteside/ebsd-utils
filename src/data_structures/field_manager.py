# -*- coding: utf-8 -*-

from numpy import ndarray, array, zeros, dot
from src.utilities.config import Config
from src.data_structures.field import FieldType, Field, DiscreteFieldMapper, FunctionalFieldMapper, FieldNullError
from src.utilities.geometry import Axis, euler_rotation_matrix, rotation_angle, misrotation_matrix, misrotation_tensor
from src.data_structures.phase import Phase
from src.algorithms.channelling import load_crit_data, fraction
from src.algorithms.clustering.dbscan import ClusterCategory, dbscan
from src.data_structures.parameter_groups import ScanParams
from src.utilities.utilities import tuple_degrees, tuple_radians, float_degrees, float_radians, log_or_zero


class FieldManager:
    def __init__(
        self,
        scan_params: ScanParams,
        phase_id_values: list[list[int | None]],
        euler_angle_values: list[list[tuple[float, float, float] | None]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
        config: Config,
    ):
        self._scan_params = scan_params
        self._config = config
        self._phase_id: Field[int] = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.DISCRETE, phase_id_values, nullable=True)
        self.euler_angles_rad: Field[tuple[float, float, float]] = None
        self.euler_angles_deg: Field[tuple[float, float, float]] = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.VECTOR_3D, euler_angle_values, nullable=True)
        self.pattern_quality: Field[float] = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.SCALAR, pattern_quality_values)
        self.index_quality: Field[float] = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.SCALAR, index_quality_values)
        self._orientation_matrix: Field[ndarray] = None
        self._reduced_matrix: Field[ndarray] = None
        self._ipf_coordinates: dict[Axis, Field[tuple[float, float]]] = dict()
        self._average_misorientation_rad: Field[float] = None
        self._misrotation_tensor: dict[Axis, Field[ndarray]] = dict()
        self._nye_tensor: Field[ndarray] = None
        self._gnd_density_lin: Field[float] = None
        self._channelling_fraction: Field[float] = None
        self._cluster_count_result: int = None
        self._clustering_category_id: Field[int] = None
        self._orientation_cluster_id: Field[int] = None

    @property
    def phase(self) -> DiscreteFieldMapper[Phase]:
        return DiscreteFieldMapper(FieldType.OBJECT, self._phase_id, self._scan_params.phases)

    @property
    def euler_angles_deg(self) -> FunctionalFieldMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalFieldMapper(FieldType.VECTOR_3D, self.euler_angles_rad, tuple_degrees, tuple_radians)

    @euler_angles_deg.setter
    def euler_angles_deg(self, value: Field[tuple[float, float, float]]) -> None:
        self.euler_angles_rad = Field(self._scan_params.width, self._scan_params.height, FieldType.VECTOR_3D, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    self.euler_angles_rad.set_value_at(x, y, tuple_radians(value.get_value_at(x, y)))
                except FieldNullError:
                    continue

    @property
    def orientation_matrix(self) -> Field[ndarray]:
        if self._orientation_matrix is None:
            self._init_orientation_matrix()

        return self._orientation_matrix

    @property
    def reduced_matrix(self) -> Field[ndarray]:
        if self._reduced_matrix is None:
            self._init_reduced_matrix()

        return self._reduced_matrix

    def ipf_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if axis not in self._ipf_coordinates:
            self._init_ipf_coordinates(axis)

        return self._ipf_coordinates[axis]

    @property
    def average_misorientation_rad(self) -> Field[float]:
        if self._average_misorientation_rad is None:
            self._init_average_misorientation_rad()

        return self._average_misorientation_rad

    @property
    def average_misorientation_deg(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.average_misorientation_rad, float_degrees, float_radians)

    def misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        if axis not in self._misrotation_tensor:
            self._init_misrotation_tensor(axis)

        return self._misrotation_tensor[axis]

    @property
    def nye_tensor(self) -> Field[ndarray]:
        if self._nye_tensor is None:
            self._init_nye_tensor()

        return self._nye_tensor

    @property
    def gnd_density_lin(self) -> Field[float]:
        if self._gnd_density_lin is None:
            self._init_gnd_density_lin()

        return self._gnd_density_lin

    @property
    def gnd_density_log(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.gnd_density_lin, log_or_zero)

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            self._init_channelling_fraction()

        return self._channelling_fraction

    @property
    def _cluster_count(self) -> int:
        if self._cluster_count_result is None:
            self._init_orientation_cluster()

        return self._cluster_count_result

    @property
    def clustering_category(self) -> DiscreteFieldMapper[ClusterCategory]:
        if self._clustering_category_id is None:
            self._init_orientation_cluster()

        mapping = {category.value: category for category in ClusterCategory}
        return DiscreteFieldMapper(FieldType.OBJECT, self._clustering_category_id, mapping)

    @property
    def orientation_cluster_id(self) -> Field[int]:
        if self._orientation_cluster_id is None:
            self._init_orientation_cluster()

        return self._orientation_cluster_id

    def _init_orientation_matrix(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.MATRIX, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    euler_angles = self.euler_angles_rad.get_value_at(x, y)
                except FieldNullError:
                    continue

                axis_set = self._scan_params.axis_set
                value = euler_rotation_matrix(axis_set, euler_angles)
                field.set_value_at(x, y, value)

        self._orientation_matrix = field

    def _init_reduced_matrix(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.MATRIX, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    orientation_matrix = self.orientation_matrix.get_value_at(x, y)
                    crystal_family = self.phase.get_value_at(x, y).lattice_type.family
                except FieldNullError:
                    continue

                value = crystal_family.reduce_matrix(orientation_matrix)
                field.set_value_at(x, y, value)

        self._reduced_matrix = field

    def _init_ipf_coordinates(self, axis: Axis) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.VECTOR_2D, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    reduced_matrix = self.reduced_matrix.get_value_at(x, y)
                    crystal_family = self.phase.get_value_at(x, y).lattice_type.family
                except FieldNullError:
                    continue

                vector = dot(reduced_matrix, array(axis.vector)).tolist()
                value = crystal_family.ipf_coordinates(vector)
                field.set_value_at(x, y, value)

        self._ipf_coordinates[axis] = field

    def _init_average_misorientation_rad(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.SCALAR, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                kernel = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
                total = 0.0
                count = 0

                try:
                    rotation_matrix_1 = self.reduced_matrix.get_value_at(x, y)
                except FieldNullError:
                    continue

                for dx, dy in kernel:
                    try:
                        if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                            rotation_matrix_2 = self.reduced_matrix.get_value_at(x + dx, y + dy)
                            total += rotation_angle(misrotation_matrix(rotation_matrix_1, rotation_matrix_2))
                            count += 1
                    except (IndexError, FieldNullError):
                        continue

                if count == 0:
                    continue
                else:
                    value = total / count
                    field.set_value_at(x, y, value)

        self._average_misorientation_rad = field

    def _init_misrotation_tensor(self, axis: Axis) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.MATRIX, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
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
                    rotation_matrix_1 = self.reduced_matrix.get_value_at(x, y)
                except FieldNullError:
                    continue

                for dx, dy in kernel:
                    try:
                        if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                            rotation_matrix_2 = self.reduced_matrix.get_value_at(x + dx, y + dy)
                            total += misrotation_tensor(misrotation_matrix(rotation_matrix_1, rotation_matrix_2), self._config.pixel_size)
                            count += 1
                    except (IndexError, FieldNullError):
                        continue

                if count == 0:
                    continue
                else:
                    value = total / count
                    field.set_value_at(x, y, value)

        self._misrotation_tensor[axis] = field

    def _init_nye_tensor(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.MATRIX, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                value = zeros((3, 3))
                count = 0

                try:
                    value += array((
                        (0.0, self.misrotation_tensor(Axis.X).get_value_at(x, y)[2][0], -self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][0]),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, -self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][2])
                    ))

                    count += 1
                except FieldNullError:
                    pass

                try:
                    value += array((
                        (0.0, 0.0, 0.0),
                        (-self.misrotation_tensor(Axis.Y).get_value_at(x, y)[2][1], 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][1]),
                        (0.0, 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][2])
                    ))

                    count += 1
                except FieldNullError:
                    pass

                if count == 0:
                    continue
                else:
                    field.set_value_at(x, y, value)

        self._nye_tensor = field

    def _init_gnd_density_lin(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.SCALAR, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    nye_tensor_norm = sum(abs(element) for row in self.nye_tensor.get_value_at(x, y).tolist() for element in row)
                    close_pack_distance = self.phase.get_value_at(x, y).close_pack_distance
                except FieldNullError:
                    continue

                value = (self._config.gnd_corrective_factor / close_pack_distance) * nye_tensor_norm
                # value = 0.25 * (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm ** 2
                field.set_value_at(x, y, value)

        self._gnd_density_lin = field

    def _init_channelling_fraction(self) -> None:
        field = Field(self._scan_params.width, self._scan_params.height, FieldType.SCALAR, default_value=None, nullable=True)

        channel_data = {
            local_id: load_crit_data(self._config.beam_atomic_number, phase.global_id, self._config.beam_energy, self._config.materials_file, self._config.channelling_cache_dir)
            for local_id, phase in self._scan_params.phases.items() if phase.global_id != Phase.UNINDEXED_ID
        }

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    rotation_matrix = self.orientation_matrix.get_value_at(x, y)
                    phase_data = channel_data[self._phase_id.get_value_at(x, y)]
                except FieldNullError:
                    continue

                effective_beam_vector = dot(rotation_matrix, self._config.beam_vector).tolist()
                value = fraction(effective_beam_vector, phase_data)
                field.set_value_at(x, y, value)

        self._channelling_fraction = field

    def _init_orientation_cluster(self) -> None:
        phase = zeros((self._scan_params.height, self._scan_params.width))
        reduced_euler_rotation_matrix = zeros((self._scan_params.height, self._scan_params.width, 3, 3))

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    phase[y][x] = self.phase.get_value_at(x, y).global_id
                    reduced_euler_rotation_matrix[y][x] = self.reduced_matrix.get_value_at(x, y)
                except FieldNullError:
                    pass

        cluster_count, category_id_array, cluster_id_array = dbscan(
            self._scan_params.width,
            self._scan_params.height,
            phase,
            reduced_euler_rotation_matrix,
            self._config.core_point_threshold,
            self._config.neighbourhood_radius_rad,
            self._config.use_cuda,
        )

        category_id_values = category_id_array.astype(int).tolist()
        cluster_id_values = cluster_id_array.astype(int).tolist()

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                if category_id_values[y][x] == 0:
                    category_id_values[y][x] = None

                if cluster_id_values[y][x] == 0:
                    cluster_id_values[y][x] = None

        self._cluster_count_result = cluster_count
        self._clustering_category_id = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.DISCRETE, category_id_values, nullable=True)
        self._orientation_cluster_id = Field.from_array(self._scan_params.width, self._scan_params.height, FieldType.DISCRETE, cluster_id_values, nullable=True)
