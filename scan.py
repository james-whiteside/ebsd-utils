from enum import Enum
from typing import Optional
import numpy


class FieldType(Enum):
    SCALAR = float
    VECTOR = tuple[float, float, float]
    MATRIX = numpy.ndarray


class Field[T]:
    def __init__(self, width: int, height: int, field_type: FieldType, default_value: Optional[T], values: Optional[list[list[T]]]):
        self.width = width
        self.height = height
        self.field_type = field_type

        if default_value is None and values is None:
            raise ValueError(f"Either default field value or array of field values must be provided.")

        if default_value is not None and not isinstance(default_value, self.field_type.value):
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

                    if not isinstance(value, self.field_type.value):
                        raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")

                    self._values.append(values[y][x])
                else:
                    self._values.append(self.default_value)

    def get_value(self, x: int, y: int) -> T:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            return self._values[y][x]

    def set_value(self, x: int, y: int, value: T) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        elif not isinstance(value, self.field_type.value):
            raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")
        else:
            self._values[y][x] = value
