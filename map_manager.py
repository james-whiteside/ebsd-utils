# -*- coding: utf-8 -*-

from field import DiscreteFieldMapper, FieldType
from field_manager import FieldManager
from map import Map, MapType
from parameter_groups import ScanParameters, ScaleParameters, ChannellingParameters, ClusteringParameters
from phase import UNINDEXED_PHASE_ID


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
        sorted_phases = sorted(self._scan_parameters.phases.items(), key=lambda item: item[1].global_id)
        sorted_local_ids = [local_id for local_id, phase in sorted_phases if phase.global_id != UNINDEXED_PHASE_ID]
        mapping = {local_id: index for index, local_id in enumerate(sorted_local_ids)}

        return Map(
            map_type=MapType.P,
            value_field=DiscreteFieldMapper(FieldType.DISCRETE, self._field_manager._phase_id, mapping),
            phase_field=self._field_manager.phase,
            max_value=len(mapping),
            min_value=0,
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
