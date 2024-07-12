# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections.abc import Callable
from numpy import zeros
from src.field import FieldLike, FieldType, FieldNullError, FunctionalFieldMapper
from src.geometry import orthogonalise_matrix


class AggregateNullError(ValueError):
    pass


class AggregateInconsistentError(ValueError):
    pass


class AggregateLike[VALUE_TYPE](ABC):
    def __init__(self, cluster_count: int, field_type: FieldType, nullable: bool = False):
        self.cluster_count = cluster_count
        self.field_type = field_type
        self.nullable = nullable

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


class Aggregate[VALUE_TYPE](AggregateLike):
    def __init__(self, cluster_count: int, value_field: FieldLike[VALUE_TYPE], cluster_id_field: FieldLike[int]):
        super().__init__(cluster_count, value_field.field_type, value_field.nullable)

        if not value_field.field_type.aggregable:
            raise ValueError(f"Value field is not an aggregable field type: {value_field.field_type.name}")
        else:
            self._values = value_field

        self._cluster_ids = FunctionalFieldMapper(FieldType.DISCRETE, cluster_id_field, lambda id: id - 1, lambda id: id + 1)
        self._aggregates: list[VALUE_TYPE] = None

    def get_value_for(self, id: int) -> VALUE_TYPE:
        index = id - 1

        if self._aggregates is None:
            self._init_aggregates()

        if not 0 <= index < self.cluster_count:
            raise IndexError(f"No cluster with ID {id}.")
        else:
            value = self._aggregates[index]

            if value is None:
                raise AggregateNullError(f"Aggregate is null for cluster with ID {id}.")
            else:
                return value

    def _init_aggregates(self) -> None:
        self._aggregates = list()

        match self._values.field_type:
            case FieldType.DISCRETE:
                values = [None for _ in range(self.cluster_count)]

                for y in range(self._values.height):
                    for x in range(self._values.width):
                        try:
                            cluster_id = self._cluster_ids.get_value_at(x, y)
                            value = self._values.get_value_at(x, y)
                        except FieldNullError:
                            continue

                        if values[cluster_id] is None:
                            values[cluster_id] = value

                        if value != values[cluster_id]:
                            raise AggregateInconsistentError(f"Value field for discrete aggregate has inconsistent values {values[cluster_id]} and {value} for cluster with ID {id}.")
                        else:
                            continue

                for cluster_id in range(self.cluster_count):
                    aggregate = values[cluster_id]
                    self._aggregates.append(aggregate)

            case FieldType.SCALAR:
                totals = [0 for _ in range(self.cluster_count)]
                counts = [0 for _ in range(self.cluster_count)]

                for y in range(self._values.height):
                    for x in range(self._values.width):
                        try:
                            cluster_id = self._cluster_ids.get_value_at(x, y)
                            value = self._values.get_value_at(x, y)
                        except FieldNullError:
                            continue

                        totals[cluster_id] += value
                        counts[cluster_id] += 1

                for cluster_id in range(self.cluster_count):
                    if counts[cluster_id] == 0:
                        aggregate = None
                    else:
                        aggregate = totals[cluster_id] / counts[cluster_id]

                    self._aggregates.append(aggregate)

            case FieldType.MATRIX:
                totals = [zeros((3, 3)) for _ in range(self.cluster_count)]
                counts = [0 for _ in range(self.cluster_count)]

                for y in range(self._values.height):
                    for x in range(self._values.width):
                        try:
                            cluster_id = self._cluster_ids.get_value_at(x, y)
                            value = self._values.get_value_at(x, y)
                        except FieldNullError:
                            continue

                        totals[cluster_id] += value
                        counts[cluster_id] += 1

                for cluster_id in range(self.cluster_count):
                    if counts[cluster_id] == 0:
                        aggregate = None
                    else:
                        aggregate = orthogonalise_matrix(totals[cluster_id] / counts[cluster_id])

                    self._aggregates.append(aggregate)


class DiscreteAggregateMapper[VALUE_TYPE](AggregateLike):
    def __init__(self, field_type: FieldType, discrete_aggregate: AggregateLike[int], mapping: dict[int, VALUE_TYPE]):
        super().__init__(discrete_aggregate.cluster_count, field_type, discrete_aggregate.nullable)
        self._mapping = mapping
        self._aggregate = discrete_aggregate

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
        super().__init__(aggregate.cluster_count, field_type, aggregate.nullable)
        self._forward_mapping = forward_mapping
        self._aggregate = aggregate

    def get_value_for(self, id: int) -> OUTPUT_TYPE:
        return self._forward_mapping(self._aggregate.get_value_for(id))
