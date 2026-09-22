# -*- coding: utf-8 -*-

from src.algorithms.field_transforms import euler_angle_colours, ipf_colours
from src.data_structures.field import DiscreteFieldMapper, FieldType, Field, FunctionalFieldMapper
from src.data_structures.field_manager import FieldManager
from src.utilities.geometry import Axis
from src.data_structures.map import Map
from src.utilities.logging import Logger


class MapManager:
    def __init__(self, field_manager: FieldManager, logger: Logger):
        self._field_manager = field_manager
        self._logger = logger

    @property
    def _euler_angle_colours(self) -> Field[tuple[float, float, float]]:
        return euler_angle_colours(self._field_manager.euler_angles_rad, self._field_manager.phase)

    def _ipf_colours(self, axis: Axis) -> Field[tuple[float, float, float]]:
        return ipf_colours(axis, self._field_manager.reduced_matrix, self._field_manager.phase)

    @property
    def phase(self) -> Map:
        self._logger.debug("Generating phase map...")
        sorted_phases = sorted(self._field_manager._scan_params.phases.items(), key=lambda item: item[1].global_id)
        sorted_local_ids = [local_id for local_id, phase in sorted_phases]
        mapping = {local_id: index for index, local_id in enumerate(sorted_local_ids)}
        value_field = DiscreteFieldMapper(FieldType.DISCRETE, self._field_manager.phase_id, mapping)

        return Map(
            value_field=value_field,
            max_value=len(mapping),
            min_value=0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def euler_angle(self) -> Map:
        self._logger.debug("Generating Euler angle map...")

        return Map(
            value_field=self._euler_angle_colours,
            max_value=(1.0, 1.0, 1.0),
            min_value=(0.0, 0.0, 0.0),
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def pattern_quality(self) -> Map:
        self._logger.debug("Generating pattern quality map...")

        return Map(
            value_field=self._field_manager.pattern_quality,
            max_value=100.0,
            min_value=0.0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def index_quality(self) -> Map:
        self._logger.debug("Generating index quality map...")

        return Map(
            value_field=self._field_manager.index_quality,
            max_value=100.0,
            min_value=0.0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    def orientation(self, axis: Axis) -> Map:
        self._logger.debug(f"Generating orientation map for {axis.name} axis...")

        return Map(
            value_field=self._ipf_colours(axis),
            max_value=(1.0, 1.0, 1.0),
            min_value=(0.0, 0.0, 0.0),
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def average_misorientation(self) -> Map:
        self._logger.debug("Generating average orientation map...")

        return Map(
            value_field=self._field_manager.average_misorientation_rad,
            min_value=0.0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def gnd_density(self) -> Map:
        self._logger.debug("Generating GND density map...")

        return Map(
            value_field=self._field_manager.gnd_density_log,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def channelling_fraction(self) -> Map:
        self._logger.debug("Generating channelling fraction map...")

        return Map(
            value_field=self._field_manager.channelling_fraction,
            max_value=100.0,
            min_value=0.0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    @property
    def orientation_cluster(self) -> Map:
        self._logger.debug("Generating orientation cluster map...")
        value_field = FunctionalFieldMapper(FieldType.DISCRETE, self._field_manager.orientation_cluster_id, lambda id: id - 1, lambda id: id + 1)

        return Map(
            value_field=value_field,
            max_value=self._field_manager._cluster_count,
            min_value=0,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )

    def single_orientation_cluster(self, cluster_id: int) -> Map:
        self._logger.debug(f"Generating single orientation cluster map for cluster {cluster_id}...")
        value_field = FunctionalFieldMapper(FieldType.BOOLEAN, self._field_manager.orientation_cluster_id, lambda id: id == cluster_id)

        return Map(
            value_field=value_field,
            upscale_factor=self._field_manager._config.maps.upscale_factor,
        )
