from enum import Enum
from typing import Callable
from numpy import ndarray


class FieldType(Enum):
    DISCRETE = int
    SCALAR = float
    VECTOR_2D = tuple
    VECTOR_3D = tuple
    MATRIX = ndarray


class Field[VALUE_TYPE]:
    def __init__(
        self,
        width: int,
        height: int,
        field_type: FieldType,
        default_value: VALUE_TYPE = None,
        values: list[list[VALUE_TYPE]] = None,
    ):
        self.width = width
        self.height = height
        self.field_type = field_type

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


class DiscreteFieldMapper[VALUE_TYPE]:
    def __init__(self, mapping: dict[int, VALUE_TYPE], discrete_field: Field[int]):
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


class FunctionalFieldMapper[INPUT_TYPE, OUTPUT_TYPE]:
    def __init__(
        self,
        forward_mapping: Callable[[INPUT_TYPE], OUTPUT_TYPE],
        field: Field[INPUT_TYPE],
        reverse_mapping: Callable[[OUTPUT_TYPE], INPUT_TYPE] = None,
    ):
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
