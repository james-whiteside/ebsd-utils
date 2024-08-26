# -*- coding: utf-8 -*-

from collections.abc import Iterator
from functools import partial
from numpy import ndarray, dot, array
from src.data_structures.aggregate import (
    AggregateType,
    CountAggregate,
    CheckAggregate,
    AverageAggregate,
    CustomAggregate,
    DiscreteAggregateMapper,
    FunctionalAggregateMapper,
    AggregateNullError,
)
from src.data_structures.field import FieldType, FieldLike
from src.data_structures.field_manager import FieldManager
from src.utilities.geometry import euler_angles, Axis, inverse_pole_figure_coordinates
from src.data_structures.phase import Phase, CrystalFamily
from src.utilities.utilities import float_degrees, tuple_degrees, log_or_zero


class AggregateManager:
    def __init__(self, field_manager: FieldManager, group_id_field: FieldLike[int]):
        self._field_manager = field_manager
        self._group_id_field = group_id_field

    @property
    def group_ids(self) -> Iterator[int]:
        return self.count.group_ids

    @property
    def count(self) -> CountAggregate:
        return CountAggregate(
            group_id_field=self._group_id_field,
        )

    @property
    def _phase_id(self) -> CheckAggregate[int]:
        return CheckAggregate(
            value_field=self._field_manager._phase_id,
            group_id_field=self._group_id_field,
        )

    @property
    def phase(self) -> DiscreteAggregateMapper[Phase]:
        return DiscreteAggregateMapper(FieldType.OBJECT, self._phase_id, self._field_manager._scan_parameters.phases)

    @property
    def reduced_euler_rotation_matrix(self) -> AverageAggregate[ndarray]:
        return AverageAggregate(
            value_field=self._field_manager.reduced_euler_rotation_matrix,
            group_id_field=self._group_id_field,
        )

    @property
    def euler_angles(self) -> FunctionalAggregateMapper[ndarray, tuple[float, float, float]]:
        axis_set = self._field_manager._scan_parameters.axis_set
        mapping = partial(euler_angles, axis_set=axis_set)
        return FunctionalAggregateMapper(FieldType.VECTOR_3D, self.reduced_euler_rotation_matrix, mapping)

    @property
    def euler_angles_degrees(self) -> FunctionalAggregateMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalAggregateMapper(FieldType.VECTOR_3D, self.euler_angles, tuple_degrees)

    def inverse_pole_figure_coordinates(self, axis: Axis) -> CustomAggregate[tuple[float, float]]:
        values: dict[int, tuple[float, float] | None] = dict()

        for id in self.group_ids:
            try:
                rotation_matrix = self.reduced_euler_rotation_matrix.get_value_for(id)
                crystal_family = self.phase.get_value_for(id).lattice_type.family
            except AggregateNullError:
                values[id] = None
                continue

            match crystal_family:
                case CrystalFamily.C:
                    vector = dot(rotation_matrix, array(axis.vector)).tolist()
                    value = inverse_pole_figure_coordinates(vector, crystal_family)
                case _:
                    raise NotImplementedError()

            values[id] = value

        return CustomAggregate(
            aggregate_type=AggregateType.AVERAGE,
            values=values,
            field_type=FieldType.VECTOR_2D,
            group_id_field=self._group_id_field,
            nullable=True,
        )

    @property
    def pattern_quality(self) -> AverageAggregate[float]:
        return AverageAggregate(
            value_field=self._field_manager.pattern_quality,
            group_id_field=self._group_id_field,
        )

    @property
    def index_quality(self) -> AverageAggregate[float]:
        return AverageAggregate(
            value_field=self._field_manager.index_quality,
            group_id_field=self._group_id_field,
        )

    @property
    def kernel_average_misorientation(self) -> AverageAggregate[float]:
        return AverageAggregate(
            value_field=self._field_manager.kernel_average_misorientation,
            group_id_field=self._group_id_field,
        )

    @property
    def kernel_average_misorientation_degrees(self) -> FunctionalAggregateMapper[float, float]:
        return FunctionalAggregateMapper(FieldType.SCALAR, self.kernel_average_misorientation, float_degrees)

    @property
    def geometrically_necessary_dislocation_density(self) -> AverageAggregate[float]:
        return AverageAggregate(
            value_field=self._field_manager.geometrically_necessary_dislocation_density,
            group_id_field=self._group_id_field,
        )

    @property
    def geometrically_necessary_dislocation_density_logarithmic(self) -> FunctionalAggregateMapper[float, float]:
        return FunctionalAggregateMapper(FieldType.SCALAR, self.geometrically_necessary_dislocation_density, log_or_zero)

    @property
    def channelling_fraction(self) -> AverageAggregate[float]:
        return AverageAggregate(
            value_field=self._field_manager.channelling_fraction,
            group_id_field=self._group_id_field,
        )
