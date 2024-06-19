# -*- coding: utf-8 -*-

import math

from clustering import ClusterCategory
from field import DiscreteFieldMapper, FieldType, Field, FunctionalFieldMapper
from field_manager import FieldManager
from geometry import Axis, inverse_stereographic
from map import Map, MapType
from parameter_groups import ScanParameters, ScaleParameters, ChannellingParameters, ClusteringParameters
from phase import UNINDEXED_PHASE_ID, CrystalFamily, Phase


class MapManager:
    def __init__(
        self,
        scan_parameters: ScanParameters,
        scale_parameters: ScaleParameters,
        channelling_parameters: ChannellingParameters,
        clustering_parameters: ClusteringParameters,
        field_manager: FieldManager
    ):
        self._scan_parameters = scan_parameters
        self._scale_parameters = scale_parameters
        self._channelling_parameters = channelling_parameters
        self._clustering_parameters = clustering_parameters
        self._field_manager = field_manager

    @property
    def _indexed_phase_filter(self) -> FunctionalFieldMapper[Phase, bool]:
        return FunctionalFieldMapper(FieldType.BOOLEAN, self._field_manager.phase, lambda phase: phase.global_id != UNINDEXED_PHASE_ID)

    @property
    def _core_point_filter(self) -> FunctionalFieldMapper[ClusterCategory, bool]:
        return FunctionalFieldMapper(FieldType.BOOLEAN, self._field_manager.orientation_clustering_category, lambda category: category is not ClusterCategory.NOISE)

    @property
    def _populated_kernel_filter(self) -> Field[bool]:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.BOOLEAN, default_value=False)

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self._field_manager.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    kernel = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
                    count = len(kernel)

                    for dx, dy in kernel:
                        try:
                            if self._field_manager.phase.get_value_at(x, y) != self._field_manager.phase.get_value_at(x + dx, y + dy):
                                count -= 1
                        except IndexError:
                            count -= 1

                    if count == 0:
                        continue
                    else:
                        field.set_value_at(x, y, True)

        return field

    @property
    def _euler_angle_colours(self) -> Field[tuple[float, float, float]]:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self._field_manager.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    euler_angles = self._field_manager.euler_angles.get_value_at(x, y)
                    max_euler_angles = self._field_manager.phase.get_value_at(x, y).max_euler_angles

                    value = (
                        euler_angles[0] / max_euler_angles[0],
                        euler_angles[1] / max_euler_angles[1],
                        euler_angles[2] / max_euler_angles[2],
                    )

                    field.set_value_at(x, y, value)

        return field

    def _inverse_pole_figure_colours(self, axis: Axis) -> Field[tuple[float, float, float]]:
        field = Field(self._scan_parameters.width, self._scan_parameters.height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

        for y in range(self._scan_parameters.height):
            for x in range(self._scan_parameters.width):
                if self._field_manager.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    a, b, c = inverse_stereographic(*self._field_manager.inverse_pole_figure_coordinates(axis).get_value_at(x, y))
                    crystal_family = self._field_manager.phase.get_value_at(x, y).lattice_type.family

                    match crystal_family:
                        case CrystalFamily.C:
                            c, b, a = sorted((-abs(a), -abs(b), -abs(c)))
                            a, b, c = c - b, (b - a) * math.sqrt(2), a * math.sqrt(3)

                            value = (
                                abs(a) / max(abs(a), abs(b), abs(c)),
                                abs(b) / max(abs(a), abs(b), abs(c)),
                                abs(c) / max(abs(a), abs(b), abs(c)),
                            )
                        case _:
                            raise NotImplementedError()

                    field.set_value_at(x, y, value)

        return field

    def get(self, map_type: MapType) -> Map:
        match map_type:
            case MapType.P:
                return self.phase
            case MapType.EA:
                return self.euler_angle
            case MapType.PQ:
                return self.pattern_quality
            case MapType.IQ:
                return self.index_quality
            case MapType.OX:
                return self.orientation(Axis.X)
            case MapType.OY:
                return self.orientation(Axis.Y)
            case MapType.OZ:
                return self.orientation(Axis.Z)
            case MapType.KAM:
                return self.kernel_average_misorientation
            case MapType.GND:
                return self.geometrically_necessary_dislocation_density
            case MapType.CF:
                return self.channelling_fraction
            case MapType.OC:
                return self.orientation_cluster

    @property
    def phase(self) -> Map:
        sorted_phases = sorted(self._scan_parameters.phases.items(), key=lambda item: item[1].global_id)
        sorted_local_ids = [local_id for local_id, phase in sorted_phases if phase.global_id != UNINDEXED_PHASE_ID]
        mapping = {local_id: index for index, local_id in enumerate(sorted_local_ids)}
        value_field = DiscreteFieldMapper(FieldType.DISCRETE, self._field_manager._phase_id, mapping)

        return Map(
            map_type=MapType.P,
            value_field=value_field,
            filter_field=self._indexed_phase_filter,
            max_value=len(mapping),
            min_value=0,
        )

    @property
    def euler_angle(self) -> Map:
        return Map(
            map_type=MapType.EA,
            value_field=self._euler_angle_colours,
            filter_field=self._indexed_phase_filter,
            max_value=(1.0, 1.0, 1.0),
            min_value=(0.0, 0.0, 0.0),
        )

    @property
    def pattern_quality(self) -> Map:
        return Map(
            map_type=MapType.PQ,
            value_field=self._field_manager.pattern_quality,
            max_value=100.0,
            min_value=0.0,
        )

    @property
    def index_quality(self) -> Map:
        return Map(
            map_type=MapType.IQ,
            value_field=self._field_manager.index_quality,
            max_value=100.0,
            min_value=0.0,
        )

    def orientation(self, axis: Axis) -> Map:
        match axis:
            case Axis.X:
                map_type = MapType.OX
            case Axis.Y:
                map_type = MapType.OY
            case Axis.Z:
                map_type = MapType.OZ

        return Map(
            map_type=map_type,
            value_field=self._inverse_pole_figure_colours(axis),
            filter_field=self._indexed_phase_filter,
            max_value=(1.0, 1.0, 1.0),
            min_value=(0.0, 0.0, 0.0),
        )

    @property
    def kernel_average_misorientation(self) -> Map:
        return Map(
            map_type=MapType.KAM,
            value_field=self._field_manager.kernel_average_misorientation,
            filter_field=self._populated_kernel_filter,
            min_value=0.0,
        )

    @property
    def geometrically_necessary_dislocation_density(self) -> Map:
        return Map(
            map_type=MapType.GND,
            value_field=self._field_manager.geometrically_necessary_dislocation_density_logarithmic,
            filter_field=self._populated_kernel_filter,
        )

    @property
    def channelling_fraction(self) -> Map:
        return Map(
            map_type=MapType.CF,
            value_field=self._field_manager.channelling_fraction,
            filter_field=self._indexed_phase_filter,
            max_value=100.0,
            min_value=0.0,
        )

    @property
    def orientation_cluster(self) -> Map:
        value_field = FunctionalFieldMapper(FieldType.DISCRETE, self._field_manager.orientation_cluster_id, lambda id: id - 1, lambda id: id + 1)

        return Map(
            map_type=MapType.OC,
            value_field=value_field,
            filter_field=self._core_point_filter,
            max_value=self._field_manager._get_cluster_count(),
            min_value=0,
        )
