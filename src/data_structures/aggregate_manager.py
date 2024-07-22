# -*- coding: utf-8 -*-

from functools import partial
from numpy import ndarray, dot, array
from aggregate import Aggregate, DiscreteAggregateMapper, FunctionalAggregateMapper
from src.data_structures.field import FieldType, FieldLike
from src.data_structures.field_manager import FieldManager
from src.utilities.geometry import euler_angles, Axis, forward_stereographic
from src.data_structures.phase import Phase
from src.utilities.utilities import tuple_degrees


class AggregateManager:
    def __init__(self, field_manager: FieldManager, group_id_field: FieldLike[int]):
        self._field_manager = field_manager
        self._group_id_field = group_id_field

    @property
    def phase_id(self) -> Aggregate[int]:
        return Aggregate(
            value_field=self._field_manager._phase_id,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def phase(self) -> DiscreteAggregateMapper[Phase]:
        return DiscreteAggregateMapper(FieldType.OBJECT, self.phase_id, self._field_manager._scan_parameters.phases)

    @property
    def reduced_euler_rotation_matrix(self) -> Aggregate[ndarray]:
        return Aggregate(
            value_field=self._field_manager.reduced_euler_rotation_matrix,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def euler_angles(self) -> FunctionalAggregateMapper[ndarray, tuple[float, float, float]]:
        axis_set = self._field_manager._scan_parameters.axis_set
        mapping = partial(euler_angles, axis_set=axis_set)
        return FunctionalAggregateMapper(FieldType.VECTOR_3D, self.reduced_euler_rotation_matrix, mapping)

    @property
    def euler_angles_degrees(self) -> FunctionalAggregateMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalAggregateMapper(FieldType.VECTOR_3D, self.euler_angles, tuple_degrees)

    def inverse_pole_figure_coordinates(self, axis: Axis):
        def mapping(rotation_matrix: ndarray) -> tuple[float, float]:
            return forward_stereographic(*dot(rotation_matrix, array(axis.value)).tolist())

        return FunctionalAggregateMapper(FieldType.VECTOR_2D, self.reduced_euler_rotation_matrix, mapping)

    @property
    def pattern_quality(self) -> Aggregate[float]:
        return Aggregate(
            value_field=self._field_manager.pattern_quality,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def index_quality(self) -> Aggregate[float]:
        return Aggregate(
            value_field=self._field_manager.index_quality,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def kernel_average_misorientation(self) -> Aggregate[float]:
        return Aggregate(
            value_field=self._field_manager.kernel_average_misorientation,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def geometrically_necessary_dislocation_density(self) -> Aggregate[float]:
        return Aggregate(
            value_field=self._field_manager.geometrically_necessary_dislocation_density,
            group_id_field=self._field_manager.orientation_cluster_id,
        )

    @property
    def channelling_fraction(self) -> Aggregate[float]:
        return Aggregate(
            value_field=self._field_manager.channelling_fraction,
            group_id_field=self._field_manager.orientation_cluster_id,
        )
