from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import Callable
from numpy import ndarray


class FieldType(Enum):
    DISCRETE = int
    SCALAR = float
    VECTOR_2D = tuple
    VECTOR_3D = tuple
    MATRIX = ndarray
    OBJECT = object

    @property
    def comparable(self) -> bool:
        if self in (FieldType.DISCRETE, FieldType.SCALAR):
            return True
        else:
            return False

    @property
    def mappable(self) -> bool:
        if self in (FieldType.DISCRETE, FieldType.SCALAR, FieldType.VECTOR_3D):
            return True
        else:
            return False


class FieldLike[VALUE_TYPE](ABC):
    def __init__(self, width: int, height: int, field_type: FieldType):
        self.width = width
        self.height = height
        self.field_type = field_type

    @abstractmethod
    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        ...

    @abstractmethod
    def set_value_at(self, x: int, y: int, value: VALUE_TYPE) -> None:
        ...

    @property
    def values(self) -> Iterator[VALUE_TYPE]:
        for y in range(self.height):
            for x in range(self.width):
                yield self.get_value_at(x, y)

    @property
    def max_value(self):
        if not self.field_type.comparable:
            raise AttributeError(f"Field type is not comparable: {self.field_type.name}")
        else:
            max_value = None

            for value in self.values:
                if max_value is None or value > max_value:
                    max_value = value

        return max_value

    @property
    def min_value(self):
        if not self.field_type.comparable:
            raise AttributeError(f"Field type is not comparable: {self.field_type.name}")
        else:
            min_value = None

            for value in self.values:
                if min_value is None or value < min_value:
                    min_value = value

        return min_value


class Field[VALUE_TYPE](FieldLike):
    def __init__(
        self,
        width: int,
        height: int,
        field_type: FieldType,
        default_value: VALUE_TYPE = None,
        values: list[list[VALUE_TYPE]] = None,
    ):
        super().__init__(width, height, field_type)

        if default_value is None and values is None:
            raise ValueError(f"Either default field value or array of field values must be provided.")

        if default_value is not None and type(default_value) is not self.field_type.value:
            raise ValueError(f"Type of default field value {type(default_value)} does not match field type {self.field_type.value}.")

        self.default_value = default_value
        self._values = list()

        for y in range(self.height):
            self._values.append(list())

            for x in range(self.width):
                if values is not None:
                    try:
                        value = values[y][x]
                    except IndexError:
                        raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of provided value array.")

                    if type(value) is not self.field_type.value:
                        raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")

                    self._values[y].append(value)
                else:
                    self._values[y].append(self.default_value)

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            return self._values[y][x]

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        elif type(value) is not self.field_type.value:
            raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")
        else:
            self._values[y][x] = value


class DiscreteFieldMapper[VALUE_TYPE](FieldLike):
    def __init__(self, field_type: FieldType, discrete_field: Field[int], mapping: dict[int, VALUE_TYPE]):
        super().__init__(discrete_field.width, discrete_field.height, field_type)
        self._mapping = mapping
        self._field = discrete_field

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        key = self._field.get_value_at(x, y)
        return self._mapping[key]

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE) -> None:
        for key in self._mapping:
            if self._mapping[key] == value:
                self._field.set_value_at(x, y, key)
                return

        raise KeyError(f"Value is not within permitted values of discrete-valued field: {value}")


class FunctionalFieldMapper[INPUT_TYPE, OUTPUT_TYPE](FieldLike):
    def __init__(
        self,
        field_type: FieldType,
        field: Field[INPUT_TYPE],
        forward_mapping: Callable[[INPUT_TYPE], OUTPUT_TYPE],
        reverse_mapping: Callable[[OUTPUT_TYPE], INPUT_TYPE] = None,
    ):
        super().__init__(field.width, field.height, field_type)
        self._forward_mapping = forward_mapping
        self._reverse_mapping = reverse_mapping
        self._field = field

    def get_value_at(self, x: int, y: int) -> OUTPUT_TYPE:
        return self._forward_mapping(self._field.get_value_at(x, y))

    def set_value_at(self, x: int, y: int, value: OUTPUT_TYPE) -> None:
        if self._reverse_mapping is None:
            raise AttributeError("Functional field mapper does not have a reverse mapping defined.")
        else:
            self._field.set_value_at(x, y, self._reverse_mapping(value))
