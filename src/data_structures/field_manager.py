# -*- coding: utf-8 -*-

from random import Random
from numpy import ndarray
from src.algorithms.field_transforms import (
    orientation_matrix,
    reduced_matrix,
    ipf_coordinates,
    average_misorientation,
    misrotation_tensor,
    nye_tensor,
    gnd_density,
    channelling_fraction,
    orientation_cluster,
)
from src.utilities.config import Config
from src.data_structures.field import FieldType, Field, DiscreteFieldMapper, FunctionalFieldMapper
from src.utilities.exception import FieldNullError
from src.utilities.geometry import Axis
from src.data_structures.phase import Phase
from src.algorithms.clustering.dbscan import ClusterCategory
from src.data_structures.parameter_groups import ScanParams
from src.utilities.utils import tuple_degrees, tuple_radians, float_degrees, float_radians, log_or_zero


class FieldManager:
    def __init__(
        self,
        scan_params: ScanParams,
        phase_id_values: list[list[int | None]],
        euler_angle_values: list[list[tuple[float, float, float] | None]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
        config: Config,
        random_source: Random,
    ):
        self._scan_params = scan_params
        self._config = config
        self._random_source = random_source
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
            self._orientation_matrix = orientation_matrix(self._config.data.euler_axis_set, self.euler_angles_rad)

        return self._orientation_matrix

    @property
    def reduced_matrix(self) -> Field[ndarray]:
        if self._reduced_matrix is None:
            self._reduced_matrix = reduced_matrix(self.orientation_matrix, self.phase)

        return self._reduced_matrix

    def ipf_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if axis not in self._ipf_coordinates:
            self._ipf_coordinates[axis] = ipf_coordinates(axis, self.reduced_matrix, self.phase)

        return self._ipf_coordinates[axis]

    @property
    def average_misorientation_rad(self) -> Field[float]:
        if self._average_misorientation_rad is None:
            self._average_misorientation_rad = average_misorientation(self.reduced_matrix, self.phase)

        return self._average_misorientation_rad

    @property
    def average_misorientation_deg(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.average_misorientation_rad, float_degrees, float_radians)

    def misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        if axis not in self._misrotation_tensor:
            self._misrotation_tensor[axis] = misrotation_tensor(axis, self._scan_params.pixel_size, self.reduced_matrix, self.phase)

        return self._misrotation_tensor[axis]

    @property
    def nye_tensor(self) -> Field[ndarray]:
        if self._nye_tensor is None:
            self._nye_tensor = nye_tensor(self.misrotation_tensor(Axis.X), self.misrotation_tensor(Axis.Y))

        return self._nye_tensor

    @property
    def gnd_density_lin(self) -> Field[float]:
        if self._gnd_density_lin is None:
            self._gnd_density_lin = gnd_density(self._config.dislocation.corrective_factor, self.nye_tensor, self.phase)

        return self._gnd_density_lin

    @property
    def gnd_density_log(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.gnd_density_lin, log_or_zero)

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            random_source = Random(self._random_source.random())

            self._channelling_fraction = channelling_fraction(
                self._config.channelling.beam_atomic_number,
                self._config.channelling.beam_energy,
                self._config.channelling.beam_vector,
                self._scan_params.phases,
                self.orientation_matrix,
                self.phase,
                random_source,
                self._config.analysis.use_cache,
                self._config.project.channelling_cache_dir,
            )

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

    def _init_orientation_cluster(self) -> None:
        self._cluster_count_result, self._clustering_category_id, self._orientation_cluster_id = orientation_cluster(
            self._config.clustering.core_point_threshold,
            self._config.clustering.neighbourhood_radius_rad,
            self.phase,
            self.reduced_matrix,
            self._config.analysis.use_cuda,
        )
