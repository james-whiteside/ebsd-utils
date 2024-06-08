# -*- coding: utf-8 -*-

from field_manager import FieldManager
from map import Map, MapType


class MapManager:
    def __init__(self, field_manager: FieldManager):
        self._field_manager = field_manager

    def get(self, map_type: MapType) -> Map:
        match map_type:
            case MapType.P:
                return self.phase
            case MapType.PQ:
                return self.pattern_quality
            case MapType.IQ:
                return self.index_quality

    @property
    def phase(self) -> Map:



        return Map(
            map_type=MapType.P,
            value_field=self._field_manager._phase_id,
        )

    @property
    def euler_angle(self) -> Map:
        raise NotImplementedError()

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
