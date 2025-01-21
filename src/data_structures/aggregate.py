# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from enum import Enum
from numpy import zeros, ndarray
from src.data_structures.field import FieldLike, FieldType, FieldNullError, FieldTypeError
from src.utilities.geometry import orthogonalise_matrix
from src.utilities.utils import format_sig_figs


class AggregateType(Enum):
    COUNT = "count"
    CHECK = "check"
    AVERAGE = "average"


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

    def serialize_value_for(self, id: int, null_serialization: str = "", sig_figs: int = None) -> list[str]:
        def format(value: VALUE_TYPE) -> str:
            if sig_figs is not None and self.field_type.roundable:
                return format_sig_figs(value, sig_figs)
            else:
                return str(value)

        if not self.field_type.serializable:
            raise FieldTypeError.lacks_property(self.field_type, "serializable")
        else:
            if self.field_type.size == 1:
                try:
                    return [format(self.get_value_for(id))]
                except AggregateNullError:
                    return [null_serialization]
            else:
                try:
                    return [format(element) for element in self.get_value_for(id)]
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
            raise AggregateLookupError(id)
        else:
            value = self._aggregates[id]

            if value is None:
                raise AggregateNullError(id)
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

                if values[group_id] != value:
                    raise CheckAggregationError(group_id, (values[group_id], value))
                else:
                    continue

        for group_id in values:
            aggregate = values[group_id]
            self._aggregates[group_id] = aggregate


class AverageAggregate[VALUE_TYPE](Aggregate):
    def __init__(self, value_field: FieldLike[VALUE_TYPE], group_id_field: FieldLike[int]):
        super().__init__(AggregateType.AVERAGE, value_field.field_type, group_id_field, value_field.nullable)

        if not value_field.field_type.averageable:
            raise FieldTypeError.lacks_property(value_field.field_type, "averageable")
        else:
            self._values = value_field

    def _init_aggregates(self) -> None:
        self._aggregates: dict[int, VALUE_TYPE] = dict()

        match self._values.field_type:
            case FieldType.SCALAR: self._init_scalar_aggregates()
            case FieldType.MATRIX: self._init_matrix_aggregates()

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


class CustomAggregate[VALUE_TYPE](Aggregate):
    def __init__(
        self,
        aggregate_type: AggregateType,
        values: dict[int, VALUE_TYPE],
        field_type: FieldType,
        group_id_field: FieldLike[int],
        nullable: bool,
    ):
        super().__init__(aggregate_type, field_type, group_id_field, nullable)
        self._values = values

    def _init_aggregates(self) -> None:
        self._aggregates = self._values


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


class AggregateLookupError(KeyError):
    def __init__(self, id: int):
        """
        Exception raised when attempting to look up an invalid group ID in an aggregate.
        :param id: The group ID.
        """
        self.id = id
        self.message = f"Aggregate does not contain a group with ID: {self.id}"
        super().__init__(self.message)


class AggregateNullError(ValueError):
    def __init__(self, id: int = None):
        """
        Exception raised when a null value in a nullable aggregate is accessed.
        :param id: ID of the group containing the null value.
        """
        self.id = id
        self.message = f"Aggregate has a null value for group {self.id}."
        super().__init__(self.message)


class CheckAggregationError(ValueError):
    def __init__(self, id: int, values: tuple[int, int]):
        """
        Exception raised when a group of points in a check-aggregate have inconsistent values.
        :param id: ID of the group.
        :param values: Inconsistent values.
        """
        self.id = id
        self.values = values
        self.message = f"Value field for check aggregate has inconsistent values {self.values[0]} and {self.values[1]} for group {self.id}."
        super().__init__(self.message)
