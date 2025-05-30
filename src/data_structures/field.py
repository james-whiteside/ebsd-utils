# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import Callable, Self, Type, Any
from PIL.Image import Image, new as new_image
from numpy import ndarray
from src.utilities.utils import format_sig_figs


class FieldType(Enum):
    BOOLEAN = "boolean"
    DISCRETE = "discrete"
    SCALAR = "scalar"
    VECTOR_2D = "2D vector"
    VECTOR_3D = "3D vector"
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
    def type(self) -> Type:
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
            case self.MATRIX: raise FieldTypeError.lacks_property(self, "serializable")
            case self.OBJECT: raise FieldTypeError.lacks_property(self, "serializable")


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
            raise FieldValueError.null_value()
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
            raise FieldTypeError.lacks_property(self.field_type, "comparable")
        else:
            return max(self.values)

    @property
    def min_value(self) -> VALUE_TYPE:
        if not self.field_type.comparable:
            raise FieldTypeError.lacks_property(self.field_type, "comparable")
        else:
            return min(self.values)

    def serialize_value_at(self, x: int, y: int, null_serialization: str = "", sig_figs: int = None) -> list[str]:
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
                    return [format(self.get_value_at(x, y))]
                except FieldNullError:
                    return [null_serialization]
            else:
                try:
                    return [format(element) for element in self.get_value_at(x, y)]
                except FieldNullError:
                    return [null_serialization for _ in range(self.field_type.size)]

    @classmethod
    def get_combined_params(cls, fields: list[Self]) -> tuple[int, int, bool]:
        if len(fields) == 0:
            raise ValueError("List must contain at least one field.")

        width = fields[0].width
        height = fields[0].height

        for field in fields:
            if field.width != width:
                raise FieldSizeMismatchError.width((field.width, width))

            if field.height != height:
                raise FieldSizeMismatchError.height((field.height, height))

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
                    raise FieldValueError.null_value()
                else:
                    field.set_value_at(x, y, value)

        field.nullable = nullable
        return field

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            value = self._values[y][x]

            if value is None:
                raise FieldNullError(x, y)
            else:
                return value

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE | None) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            self._assert_value_permitted(value)
            self._values[y][x] = value

    def _assert_value_permitted(self, value: VALUE_TYPE | None) -> None:
        if value is None:
            if self.nullable:
                return
            else:
                raise FieldValueError.null_value()

        if type(value) is not self.field_type.type:
            raise FieldValueError.wrong_type(value, self.field_type)

        if self.field_type.type is tuple and len(value) != self.field_type.size:
            raise FieldValueError.wrong_length(value, self.field_type)


class DiscreteFieldMapper[VALUE_TYPE](FieldLike):
    def __init__(self, field_type: FieldType, discrete_field: FieldLike[int], mapping: dict[int, VALUE_TYPE]):
        super().__init__(discrete_field.width, discrete_field.height, field_type, discrete_field.nullable)

        for value in discrete_field.values:
            if value not in mapping:
                FieldValueError.not_in_mapping(value)

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

        raise FieldValueError.not_in_mapping(value)


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
            raise FieldValueError.no_reverse_mapping(value)
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
            raise FieldValueError.not_valid_map_value(value)

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


class FieldLookupError(IndexError):
    def __init__(self, x: int, y: int):
        """
        Exception raised when attempting to access out-of-bounds coordinates of a field.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        """
        self.x = x
        self.y = y
        self.message = f"Coordinates are out of bound of field: ({self.x}, {self.y})"
        super().__init__(self.message)


class FieldNullError(ValueError):
    def __init__(self, x: int, y: int):
        """
        Exception raised when a null value in a nullable field is accessed.
        :param x: The x-coordinate of the null value.
        :param y: The y-coordinate of the null value.
        """
        self.x = x
        self.y = y
        self.message = f"Field has a null value at coordinate ({self.x}, {self.y})."
        super().__init__(self.message)


class FieldTypeError(TypeError):
    def __init__(self, field_type: FieldType, reason: str):
        """
        Exception raised when attempting to perform an operation on a field with an inappropriate type.
        :param field_type: The field type.
        :param reason: The reason the field type is inappropriate.
        """
        self.field_type = field_type
        self.reason = reason
        self.message = f"{self.reason}: {self.field_type.value}"
        super().__init__(self.message)

    @classmethod
    def wrong_type(cls, field_type: FieldType, correct_type: FieldType) -> Self:
        reason = f"{correct_type.value.capitalize()} field required but a field of the wrong type was provided"
        return FieldTypeError(field_type, reason)

    @classmethod
    def lacks_property(cls, field_type: FieldType, property: str) -> Self:
        reason = f"Field type is not {property}"
        return FieldTypeError(field_type, reason)


class FieldValueError(ValueError):
    def __init__(self, value: Any, reason: str):
        """
        Exception raised when attempting to write an inappropriate value to a field.
        :param value: The value.
        :param reason: The reason the value is inappropriate.
        """
        self.value = value
        self.reason = reason
        self.message = f"{self.reason}: {self.value}"
        super().__init__(self.message)

    @classmethod
    def null_value(cls) -> Self:
        reason = "Field is not nullable but a null value was provided"
        return FieldValueError(None, reason)

    @classmethod
    def wrong_type(cls, value: Any, field_type: FieldType) -> Self:
        reason = f"Field is of type {field_type.type.__name__} but a value of type {type(value).__name__} was provided"
        return FieldValueError(value, reason)

    @classmethod
    def wrong_length(cls, value: tuple, field_type: FieldType) -> Self:
        reason = f"Tuple field has size {field_type.size} but a tuple value of length {len(value)} was provided"
        return FieldValueError(value, reason)

    @classmethod
    def not_in_mapping(cls, value: int) -> Self:
        reason = f"Value is not within provided mapping for discrete field mapper"
        return FieldValueError(value, reason)

    @classmethod
    def no_reverse_mapping(cls, value: Any) -> Self:
        reason = "Cannot set value as functional field mapper does not have reverse mapping defined"
        return FieldValueError(value, reason)

    @classmethod
    def not_valid_map_value(cls, value: tuple[float, float, float]) -> Self:
        reason = "Map field may only take tuples of values between 0.0 and 1.0 but an invalid value was provided"
        return FieldValueError(value, reason)


class FieldSizeMismatchError(ValueError):
    class Dimension(Enum):
        WIDTH = "width"
        HEIGHT = "height"

    def __init__(self, dimension: Dimension, values: tuple[int, int]):
        """
        Exception raised when attempting to derive a new field from input fields with mismatched sizes.
        :param dimension: Dimension of mismatch.
        :param values: Mismatched values.
        """
        self.dimension = dimension
        self.values = values
        self.message = f"{self.dimension.name.title()}s {self.values[0]} and {self.values[1]} of supplied fields do not match."
        super().__init__(self.message)

    @classmethod
    def width(cls, values: tuple[int, int]) -> Self:
        return FieldSizeMismatchError(FieldSizeMismatchError.Dimension.WIDTH, values)

    @classmethod
    def height(cls, values: tuple[int, int]) -> Self:
        return FieldSizeMismatchError(FieldSizeMismatchError.Dimension.HEIGHT, values)
