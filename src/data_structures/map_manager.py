# -*- coding: utf-8 -*-

import math
from numpy import array, dot
from src.data_structures.field import DiscreteFieldMapper, FieldType, Field, FunctionalFieldMapper, FieldNullError
from src.data_structures.field_manager import FieldManager
from src.utilities.geometry import Axis, reduce_vector
from src.data_structures.map import Map, MapType
from src.data_structures.phase import CrystalFamily, Phase
from src.utilities.utilities import maximise_brightness


class MapManager:
    def __init__(self, field_manager: FieldManager):
        self._field_manager = field_manager

    @property
    def _euler_angle_colours(self) -> Field[tuple[float, float, float]]:
        field = Field(self._field_manager._scan_params.width, self._field_manager._scan_params.height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

        for y in range(self._field_manager._scan_params.height):
            for x in range(self._field_manager._scan_params.width):
                try:
                    euler_angles = self._field_manager.euler_angles_rad.get_value_at(x, y)
                    max_euler_angles = self._field_manager.phase.get_value_at(x, y).lattice_type.family.max_euler_angles
                except FieldNullError:
                    continue

                value = (
                    euler_angles[0] / max_euler_angles[0],
                    euler_angles[1] / max_euler_angles[1],
                    euler_angles[2] / max_euler_angles[2],
                )

                field.set_value_at(x, y, value)

        return field

    def _ipf_colours(self, axis: Axis) -> Field[tuple[float, float, float]]:
        field = Field(self._field_manager._scan_params.width, self._field_manager._scan_params.height, FieldType.VECTOR_3D, default_value=(0.0, 0.0, 0.0))

        for y in range(self._field_manager._scan_params.height):
            for x in range(self._field_manager._scan_params.width):
                try:
                    rotation_matrix = self._field_manager.reduced_matrix.get_value_at(x, y)
                    crystal_family = self._field_manager.phase.get_value_at(x, y).lattice_type.family
                except FieldNullError:
                    continue

                match crystal_family:
                    case CrystalFamily.C:
                        vector = dot(rotation_matrix, array(axis.vector)).tolist()
                        u, v, w = reduce_vector(vector)
                        r, g, b = w - v, (v - u) * math.sqrt(2), u * math.sqrt(3)
                        value = maximise_brightness((r, g, b))
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
            case MapType.OB:
                return self.orientation(self._field_manager._channelling_params.beam_axis)
            case MapType.KAM:
                return self.average_misorientation
            case MapType.GND:
                return self.gnd_density
            case MapType.CF:
                return self.channelling_fraction
            case MapType.OC:
                return self.orientation_cluster

    @property
    def phase(self) -> Map:
        sorted_phases = sorted(self._field_manager._scan_params.phases.items(), key=lambda item: item[1].global_id)
        sorted_local_ids = [local_id for local_id, phase in sorted_phases if phase.global_id != Phase.UNINDEXED_ID]
        mapping = {local_id: index for index, local_id in enumerate(sorted_local_ids)}
        value_field = DiscreteFieldMapper(FieldType.DISCRETE, self._field_manager._phase_id, mapping)

        return Map(
            map_type=MapType.P,
            value_field=value_field,
            max_value=len(mapping),
            min_value=0,
        )

    @property
    def euler_angle(self) -> Map:
        return Map(
            map_type=MapType.EA,
            value_field=self._euler_angle_colours,
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
            case self._field_manager._channelling_params.beam_axis:
                map_type = MapType.OB

        return Map(
            map_type=map_type,
            value_field=self._ipf_colours(axis),
            max_value=(1.0, 1.0, 1.0),
            min_value=(0.0, 0.0, 0.0),
        )

    @property
    def average_misorientation(self) -> Map:
        return Map(
            map_type=MapType.KAM,
            value_field=self._field_manager.average_misorientation_rad,
            min_value=0.0,
        )

    @property
    def gnd_density(self) -> Map:
        return Map(
            map_type=MapType.GND,
            value_field=self._field_manager.gnd_density_log,
        )

    @property
    def channelling_fraction(self) -> Map:
        return Map(
            map_type=MapType.CF,
            value_field=self._field_manager.channelling_fraction,
            max_value=100.0,
            min_value=0.0,
        )

    @property
    def orientation_cluster(self) -> Map:
        value_field = FunctionalFieldMapper(FieldType.DISCRETE, self._field_manager.orientation_cluster_id, lambda id: id - 1, lambda id: id + 1)

        return Map(
            map_type=MapType.OC,
            value_field=value_field,
            max_value=self._field_manager._cluster_count,
            min_value=0,
        )
