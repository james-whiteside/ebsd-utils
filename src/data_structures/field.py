# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import Callable, Self
from PIL.Image import Image, new as new_image
from numpy import ndarray
from src.utilities.exception import FieldNullError, FieldsInconsistentError
from src.utilities.utils import format_sig_figs


class FieldType(Enum):
    BOOLEAN = "boolean"
    DISCRETE = "discrete"
    SCALAR = "scalar"
    VECTOR_2D = "vector 2d"
    VECTOR_3D = "vector 3d"
    MATRIX = "matrix"
    OBJECT = "object"

    @property
    def comparable(self) -> bool:
        return self in (FieldType.DISCRETE, FieldType.SCALAR)

    @property
    def averageable(self) -> bool:
        return self in (FieldType.SCALAR, FieldType.MATRIX)

    @property
    def mappable(self) -> bool:
        return self in (FieldType.DISCRETE, FieldType.SCALAR, FieldType.VECTOR_3D)

    @property
    def serializable(self) -> bool:
        return self in (FieldType.BOOLEAN, FieldType.DISCRETE, FieldType.SCALAR, FieldType.VECTOR_2D, FieldType.VECTOR_3D)

    @property
    def roundable(self) -> bool:
        return self in (FieldType.SCALAR, FieldType.VECTOR_2D, FieldType.VECTOR_3D)

    @property
    def type(self) -> type:
        match self:
            case self.BOOLEAN: return bool
            case self.DISCRETE: return int
            case self.SCALAR: return float
            case self.VECTOR_2D: return tuple
            case self.VECTOR_3D: return tuple
            case self.MATRIX: return ndarray
            case self.OBJECT: return object

    @property
    def size(self) -> int:
        match self:
            case self.BOOLEAN: return 1
            case self.DISCRETE: return 1
            case self.SCALAR: return 1
            case self.VECTOR_2D: return 2
            case self.VECTOR_3D: return 3
            case self.MATRIX: raise AttributeError(f"Field type is not serializable: {self.name}")
            case self.OBJECT: raise AttributeError(f"Field type is not serializable: {self.name}")


class FieldLike[VALUE_TYPE](ABC):
    def __init__(self, width: int, height: int, field_type: FieldType, nullable: bool = False):
        self.width = width
        self.height = height
        self.field_type = field_type
        self._nullable = nullable

    @property
    def nullable(self) -> bool:
        return self._nullable

    @nullable.setter
    def nullable(self, value: bool) -> None:
        if not value and self.has_null_value:
            raise FieldNullError()
        else:
            self._nullable = value

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
                try:
                    yield self.get_value_at(x, y)
                except FieldNullError:
                    continue

    @property
    def has_null_value(self) -> bool:
        for y in range(self.height):
            for x in range(self.width):
                try:
                    self.get_value_at(x, y)
                except FieldNullError:
                    return True

        return False

    @property
    def max_value(self) -> VALUE_TYPE:
        if not self.field_type.comparable:
            raise AttributeError(f"Field type is not comparable: {self.field_type.name}")
        else:
            return max(self.values)

    @property
    def min_value(self) -> VALUE_TYPE:
        if not self.field_type.comparable:
            raise AttributeError(f"Field type is not comparable: {self.field_type.name}")
        else:
            return min(self.values)

    def serialize_value_at(self, x: int, y: int, null_serialization: str = "", sig_figs: int = None) -> list[str]:
        def format(value: VALUE_TYPE) -> str:
            if sig_figs is not None and self.field_type.roundable:
                return format_sig_figs(value, sig_figs)
            else:
                return str(value)

        if not self.field_type.serializable:
            raise AttributeError(f"Field type is not serializable: {self.field_type.name}")
        else:
            if self.field_type.size == 1:
                try:
                    return [format(self.get_value_at(x, y))]
                except FieldNullError:
                    return [null_serialization]
            else:
                try:
                    return [format(element) for element in self.get_value_at(x, y)]
                except FieldNullError:
                    return [null_serialization for _ in range(self.field_type.size)]

    @classmethod
    def get_params(cls, fields: list[Self]) -> tuple[int, int, bool]:
        if len(fields) == 0:
            raise ValueError("List must contain at least one field.")

        width = fields[0].width
        height = fields[0].height

        for field in fields:
            if field.width != width:
                raise FieldsInconsistentError.width((field.width, width))

            if field.height != height:
                raise FieldsInconsistentError.height((field.height, height))

        nullable = any(field.nullable for field in fields)
        return width, height, nullable


class Field[VALUE_TYPE](FieldLike):
    def __init__(
        self,
        width: int,
        height: int,
        field_type: FieldType,
        default_value: VALUE_TYPE | None,
        nullable: bool = False,
    ):
        super().__init__(width, height, field_type, nullable)
        self._assert_value_permitted(default_value)
        self.default_value = default_value
        self._values = list()

        for y in range(self.height):
            self._values.append(list())

            for x in range(self.width):
                self._values[y].append(self.default_value)

    @classmethod
    def from_array(
        cls,
        width: int,
        height: int,
        field_type: FieldType,
        values: list[list[VALUE_TYPE | None]],
        nullable: bool = False,
    ) -> Self:
        field = Field(width, height, field_type, default_value=None, nullable=True)

        for y in range(field.height):
            for x in range(field.width):
                try:
                    value = values[y][x]
                except IndexError:
                    raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of provided value array.")

                if not nullable and value is None:
                    raise ValueError(f"Provided value is None but field is not nullable.")
                else:
                    field.set_value_at(x, y, value)

        field.nullable = nullable
        return field

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            value = self._values[y][x]

            if value is None:
                raise FieldNullError((x, y))
            else:
                return value

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE | None) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            self._assert_value_permitted(value)
            self._values[y][x] = value

    def _assert_value_permitted(self, value: VALUE_TYPE | None) -> None:
        if value is None:
            if self.nullable:
                return
            else:
                raise ValueError(f"Provided value is None but field is not nullable.")

        if type(value) is not self.field_type.type:
            raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.type}.")

        if self.field_type.type is tuple and len(value) != self.field_type.size:
            raise ValueError(f"Length of provided tuple value is {len(value)} and does not match field type size {self.field_type.size}.")


class DiscreteFieldMapper[VALUE_TYPE](FieldLike):
    def __init__(self, field_type: FieldType, discrete_field: FieldLike[int], mapping: dict[int, VALUE_TYPE]):
        super().__init__(discrete_field.width, discrete_field.height, field_type, discrete_field.nullable)
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
        field: FieldLike[INPUT_TYPE],
        forward_mapping: Callable[[INPUT_TYPE], OUTPUT_TYPE],
        reverse_mapping: Callable[[OUTPUT_TYPE], INPUT_TYPE] = None,
    ):
        super().__init__(field.width, field.height, field_type, field.nullable)
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


class MapField(Field):
    def __init__(
        self,
        width: int,
        height: int,
        default_value: tuple[float, float, float],
        upscale_factor: int = 1,
    ):
        super().__init__(width, height, FieldType.VECTOR_3D, default_value, nullable=False)
        self.upscale_factor = upscale_factor

    def _assert_value_permitted(self, value: tuple[float, float, float]) -> None:
        super()._assert_value_permitted(value)

        if not all(0.0 <= item <= 1.0 for item in value):
            raise ValueError(f"Map field may only take tuples of values between 0.0 and 1.0. Provided tuple: {value}")

    def to_image(self) -> Image:
        size = self.upscale_factor * self.width, self.upscale_factor * self.height
        image = new_image("RGB", size)

        for y in range(self.height):
            for x in range(self.width):
                pixel = (
                    int(round(255 * self.get_value_at(x, y)[0])),
                    int(round(255 * self.get_value_at(x, y)[1])),
                    int(round(255 * self.get_value_at(x, y)[2])),
                )

                for j in range(self.upscale_factor):
                    for i in range(self.upscale_factor):

                        coordinates = self.upscale_factor * x + i, self.upscale_factor * y + j
                        image.putpixel(coordinates, pixel)

        return image
