# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from enum import Enum
from numpy import zeros, ndarray
from src.data_structures.field import FieldLike, FieldType, FieldNullError
from src.utilities.geometry import orthogonalise_matrix


class AggregateType(Enum):
    COUNT = "count"
    CHECK = "check"
    AVERAGE = "average"


class AggregateNullError(ValueError):
    pass


class AggregateInconsistentError(ValueError):
    pass


class AggregateLike[VALUE_TYPE](ABC):
    def __init__(self, aggregate_type: AggregateType, field_type: FieldType, nullable: bool = False):
        self.aggregate_type = aggregate_type
        self.field_type = field_type
        self.nullable = nullable

    @property
    @abstractmethod
    def group_count(self) -> int:
        ...

    @property
    @abstractmethod
    def group_ids(self) -> Iterator[int]:
        ...

    @abstractmethod
    def get_value_for(self, id: int) -> VALUE_TYPE:
        ...

    def serialize_value_for(self, id: int, null_serialization: str = "") -> list[str]:
        if not self.field_type.serializable:
            raise AttributeError(f"Field type is not serializable: {self.field_type.name}")
        else:
            if self.field_type.size == 1:
                try:
                    return [str(self.get_value_for(id))]
                except AggregateNullError:
                    return [null_serialization]
            else:
                try:
                    return [str(element) for element in self.get_value_for(id)]
                except AggregateNullError:
                    return [null_serialization for _ in range(self.field_type.size)]


class Aggregate[VALUE_TYPE](AggregateLike, ABC):
    def __init__(
        self,
        aggregate_type: AggregateType,
        field_type: FieldType,
        group_id_field: FieldLike[int],
        nullable: bool = False
    ):
        super().__init__(aggregate_type, field_type, nullable)
        self._group_ids = group_id_field
        self._aggregates: dict[int, VALUE_TYPE] = None

    @property
    def group_count(self) -> int:
        if self._aggregates is None:
            self._init_aggregates()

        return len(self._aggregates)

    @property
    def group_ids(self) -> Iterator[int]:
        if self._aggregates is None:
            self._init_aggregates()

        for id in sorted(self._aggregates):
            yield id

    def get_value_for(self, id: int) -> VALUE_TYPE:
        if self._aggregates is None:
            self._init_aggregates()

        if id not in self._aggregates:
            raise KeyError(f"No group with ID {id}.")
        else:
            value = self._aggregates[id]

            if value is None:
                raise AggregateNullError(f"Aggregate is null for group with ID {id}.")
            else:
                return value

    @abstractmethod
    def _init_aggregates(self) -> None:
        ...


class CountAggregate(Aggregate):
    def __init__(self, group_id_field: FieldLike[int]):
        super().__init__(AggregateType.COUNT, FieldType.DISCRETE, group_id_field)

    def _init_aggregates(self):
        self._aggregates: dict[int, int] = dict()
        counts: dict[int, int] = dict()

        for y in range(self._group_ids.height):
            for x in range(self._group_ids.width):
                try:
                    group_id = self._group_ids.get_value_at(x, y)
                except FieldNullError:
                    continue

                if group_id not in counts:
                    counts[group_id] = 0

                counts[group_id] += 1

        for group_id in counts:
            aggregate = counts[group_id]
            self._aggregates[group_id] = aggregate


class CheckAggregate[VALUE_TYPE](Aggregate):
    def __init__(self, value_field: FieldLike[VALUE_TYPE], group_id_field: FieldLike[int]):
        super().__init__(AggregateType.AVERAGE, value_field.field_type, group_id_field, value_field.nullable)
        self._values = value_field

    def _init_aggregates(self) -> None:
        self._aggregates: dict[int, VALUE_TYPE] = dict()
        values: dict[int, VALUE_TYPE | None] = dict()

        for y in range(self._values.height):
            for x in range(self._values.width):
                try:
                    group_id = self._group_ids.get_value_at(x, y)
                except FieldNullError:
                    continue

                if group_id not in values:
                    values[group_id] = None

                try:
                    value = self._values.get_value_at(x, y)
                except FieldNullError:
                    continue

                if values[group_id] is None:
                    values[group_id] = value

                if value != values[group_id]:
                    raise AggregateInconsistentError(
                        f"Value field for check aggregate has inconsistent values {values[group_id]} and {value} for group with ID {group_id}.")
                else:
                    continue

        for group_id in values:
            aggregate = values[group_id]
            self._aggregates[group_id] = aggregate


class AverageAggregate[VALUE_TYPE](Aggregate):
    def __init__(self, value_field: FieldLike[VALUE_TYPE], group_id_field: FieldLike[int]):
        super().__init__(AggregateType.AVERAGE, value_field.field_type, group_id_field, value_field.nullable)

        if not value_field.field_type.averageable:
            raise ValueError(f"Value field is not an averageable field type: {value_field.field_type.name}")
        else:
            self._values = value_field

    def _init_aggregates(self) -> None:
        self._aggregates: dict[int, VALUE_TYPE] = dict()

        match self._values.field_type:
            case FieldType.SCALAR:
                self._init_scalar_aggregates()
            case FieldType.MATRIX:
                self._init_matrix_aggregates()

    def _init_scalar_aggregates(self) -> None:
        totals: dict[int, float] = dict()
        counts: dict[int, int] = dict()

        for y in range(self._values.height):
            for x in range(self._values.width):
                try:
                    group_id = self._group_ids.get_value_at(x, y)
                except FieldNullError:
                    continue

                if group_id not in totals:
                    totals[group_id] = 0.0
                    counts[group_id] = 0

                try:
                    value = self._values.get_value_at(x, y)
                except FieldNullError:
                    continue

                totals[group_id] += value
                counts[group_id] += 1

        for group_id in totals:
            if counts[group_id] == 0:
                aggregate = None
            else:
                aggregate = totals[group_id] / counts[group_id]

            self._aggregates[group_id] = aggregate

    def _init_matrix_aggregates(self) -> None:
        totals: dict[int, ndarray] = dict()
        counts: dict[int, int] = dict()

        for y in range(self._values.height):
            for x in range(self._values.width):
                try:
                    group_id = self._group_ids.get_value_at(x, y)
                except FieldNullError:
                    continue

                if group_id not in totals:
                    totals[group_id] = zeros((3, 3))
                    counts[group_id] = 0

                try:
                    value = self._values.get_value_at(x, y)
                except FieldNullError:
                    continue

                totals[group_id] += value
                counts[group_id] += 1

        for group_id in totals:
            if counts[group_id] == 0:
                aggregate = None
            else:
                aggregate = orthogonalise_matrix(totals[group_id] / counts[group_id])

            self._aggregates[group_id] = aggregate


class DiscreteAggregateMapper[VALUE_TYPE](AggregateLike):
    def __init__(self, field_type: FieldType, discrete_aggregate: AggregateLike[int], mapping: dict[int, VALUE_TYPE]):
        super().__init__(discrete_aggregate.aggregate_type, field_type, discrete_aggregate.nullable)
        self._mapping = mapping
        self._aggregate = discrete_aggregate

    @property
    def group_count(self) -> int:
        return self._aggregate.group_count

    @property
    def group_ids(self) -> Iterator[int]:
        return self._aggregate.group_ids

    def get_value_for(self, id: int) -> VALUE_TYPE:
        key = self._aggregate.get_value_for(id)
        return self._mapping[key]


class FunctionalAggregateMapper[INPUT_TYPE, OUTPUT_TYPE](AggregateLike):
    def __init__(
        self,
        field_type: FieldType,
        aggregate: AggregateLike[INPUT_TYPE],
        forward_mapping: Callable[[INPUT_TYPE], OUTPUT_TYPE],
    ):
        super().__init__(aggregate.aggregate_type, field_type, aggregate.nullable)
        self._forward_mapping = forward_mapping
        self._aggregate = aggregate

    @property
    def group_count(self) -> int:
        return self._aggregate.group_count

    @property
    def group_ids(self) -> Iterator[int]:
        return self._aggregate.group_ids

    def get_value_for(self, id: int) -> OUTPUT_TYPE:
        return self._forward_mapping(self._aggregate.get_value_for(id))
