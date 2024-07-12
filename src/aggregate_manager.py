# -*- coding: utf-8 -*-

from aggregate import Aggregate
from src.field_manager import FieldManager


class AggregateManager:
    def __init__(self, field_manager: FieldManager):
        self._field_manager = field_manager
        self._cluster_count = self._field_manager._get_cluster_count()

    @property
    def phase_id(self) -> Aggregate[int]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager._phase_id,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def pattern_quality(self) -> Aggregate[float]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager.pattern_quality,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def index_quality(self) -> Aggregate[float]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager.index_quality,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def kernel_average_misorientation(self) -> Aggregate[float]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager.kernel_average_misorientation,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def geometrically_necessary_dislocation_density(self) -> Aggregate[float]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager.geometrically_necessary_dislocation_density,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def channelling_fraction(self) -> Aggregate[float]:
        return Aggregate(
            cluster_count=self._cluster_count,
            value_field=self._field_manager.channelling_fraction,
            cluster_id_field=self._field_manager.orientation_cluster_id,
        )
