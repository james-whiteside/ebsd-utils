# -*- coding: utf-8 -*-

from random import Random
from numpy import ndarray
from src.algorithms.field_transforms import (
    ipf_coordinates,
    average_misorientation,
    misrotation_tensor,
    nye_tensor,
    gnd_density,
    channelling_fraction,
    orientation_cluster,
)
from src.algorithms.new_field_transforms.crystal_family import crystal_family
from src.algorithms.new_field_transforms.lattice_type import lattice_type
from src.algorithms.new_field_transforms.orientation_matrix import orientation_matrix
from src.algorithms.new_field_transforms.reduced_matrix import reduced_matrix
from src.data_structures.fields.discrete_field import DiscreteField
from src.data_structures.fields.enum_field import EnumField
from src.data_structures.fields.matrix_field import MatrixField
from src.data_structures.fields.scalar_field import ScalarField
from src.data_structures.fields.vector_2d_field import Vector2DField
from src.data_structures.fields.vector_3d_field import Vector3DField
from src.utilities.config import Config
from src.data_structures.field import FieldType, Field, DiscreteFieldMapper, FunctionalFieldMapper, FieldNullError
from src.utilities.geometry import Axis
from src.data_structures.phase import Phase, BravaisLattice, CrystalFamily
from src.algorithms.clustering.dbscan import ClusterCategory
from src.data_structures.parameter_groups import ScanParams
from src.utilities.logging import Logger
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
        logger: Logger,
        random_source: Random,
    ):
        self._scan_params = scan_params
        self._config = config
        self._logger = logger
        self._random_source = random_source
        self.phase_id: DiscreteField = DiscreteField.from_list(self._scan_params.width, self._scan_params.height, phase_id_values, nullable=True)
        self.euler_angles_rad: Vector3DField | None = None
        self.euler_angles_deg: Vector3DField = Vector3DField.from_list(self._scan_params.width, self._scan_params.height, euler_angle_values, nullable=True)
        self.pattern_quality: ScalarField = ScalarField.from_list(self._scan_params.width, self._scan_params.height, pattern_quality_values)
        self.index_quality: ScalarField = ScalarField.from_list(self._scan_params.width, self._scan_params.height, index_quality_values)
        self._orientation_matrix: MatrixField | None = None
        self._lattice_type: EnumField[BravaisLattice] | None = None
        self._crystal_family: EnumField[CrystalFamily] | None = None
        self._reduced_matrix: MatrixField | None = None
        self._ipf_coordinates: dict[Axis, Vector2DField] = dict()
        self._average_misorientation_rad: ScalarField  | None = None
        self._misrotation_tensor: dict[Axis, MatrixField] = dict()
        self._nye_tensor: MatrixField | None = None
        self._gnd_density_lin: ScalarField | None = None
        self._channelling_fraction: ScalarField | None = None
        self._cluster_count_result: int | None = None
        self._clustering_category_id: DiscreteField | None = None
        self._orientation_cluster_id: DiscreteField | None = None

    @property
    def phase(self) -> DiscreteFieldMapper[Phase]:
        return DiscreteFieldMapper(FieldType.OBJECT, self.phase_id, self._scan_params.phases)

    @property
    def euler_angles_deg(self) -> FunctionalFieldMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalFieldMapper(FieldType.VECTOR_3D, self.euler_angles_rad, tuple_degrees, tuple_radians)

    @euler_angles_deg.setter
    def euler_angles_deg(self, value: Field[tuple[float, float, float]]) -> None:
        self.euler_angles_rad = Vector3DField(self._scan_params.width, self._scan_params.height, default_value=None, nullable=True)

        for y in range(self._scan_params.height):
            for x in range(self._scan_params.width):
                try:
                    self.euler_angles_rad.set_value_at(x, y, tuple_radians(value.get_value_at(x, y)))
                except FieldNullError:
                    continue

    @property
    def orientation_matrix(self) -> MatrixField:
        if self._orientation_matrix is None:
            self._logger.debug("Generating orientation matrix field...")
            self._orientation_matrix = orientation_matrix(self._config.data.euler_axis_set, self.euler_angles_rad)

        return self._orientation_matrix

    @property
    def lattice_type(self) -> EnumField[BravaisLattice]:
        if self._lattice_type is None:
            self._logger.debug("Generating lattice type field...")
            self._lattice_type = lattice_type(self.phase_id, self._scan_params.phases)

        return self._lattice_type

    @property
    def crystal_family(self) -> EnumField[CrystalFamily]:
        if self._crystal_family is None:
            self._logger.debug("Generating crystal family field...")
            self._crystal_family = crystal_family(self.lattice_type)

        return self._crystal_family

    @property
    def reduced_matrix(self) -> MatrixField:
        if self._reduced_matrix is None:
            self._logger.debug("Generating reduced matrix field...")
            self._reduced_matrix = reduced_matrix(self.orientation_matrix, self.crystal_family)

        return self._reduced_matrix

    def ipf_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if axis not in self._ipf_coordinates:
            self._logger.debug("Generating IPF coordinate field...")
            self._ipf_coordinates[axis] = ipf_coordinates(axis, self.reduced_matrix, self.phase)

        return self._ipf_coordinates[axis]

    @property
    def average_misorientation_rad(self) -> Field[float]:
        if self._average_misorientation_rad is None:
            self._logger.debug("Generating average misorientation field...")
            self._average_misorientation_rad = average_misorientation(self.reduced_matrix, self.phase)

        return self._average_misorientation_rad

    @property
    def average_misorientation_deg(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.average_misorientation_rad, float_degrees, float_radians)

    def misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        if axis not in self._misrotation_tensor:
            self._logger.debug(f"Generating misrotation tensor field for {axis.name} axis...")
            self._misrotation_tensor[axis] = misrotation_tensor(axis, self._scan_params.pixel_size, self.reduced_matrix, self.phase)

        return self._misrotation_tensor[axis]

    @property
    def nye_tensor(self) -> Field[ndarray]:
        if self._nye_tensor is None:
            self._logger.debug("Generating Nye tensor field...")
            self._nye_tensor = nye_tensor(self.misrotation_tensor(Axis.X), self.misrotation_tensor(Axis.Y))

        return self._nye_tensor

    @property
    def gnd_density_lin(self) -> Field[float]:
        if self._gnd_density_lin is None:
            self._logger.debug("Generating GND density field...")
            self._gnd_density_lin = gnd_density(self._config.dislocation.corrective_factor, self.nye_tensor, self.phase)

        return self._gnd_density_lin

    @property
    def gnd_density_log(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(FieldType.SCALAR, self.gnd_density_lin, log_or_zero)

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            self._logger.debug("Generating channelling fraction field...")
            random_source = Random(self._random_source.random())

            self._channelling_fraction = channelling_fraction(
                self._config.channelling.beam_atomic_number,
                self._config.channelling.beam_energy,
                self._config.channelling.beam_axis,
                self._scan_params.phases,
                self.orientation_matrix,
                self.phase,
                random_source,
                self._config.analysis.use_cache,
                self._config.project.channelling_cache_dir,
                self._logger,
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
        self._logger.debug("Generating orientation cluster field...")

        self._cluster_count_result, self._clustering_category_id, self._orientation_cluster_id = orientation_cluster(
            self._config.clustering.core_point_threshold,
            self._config.clustering.neighbourhood_radius_rad,
            self.phase,
            self.reduced_matrix,
            self._config.analysis.use_cuda,
            self._logger,
        )
