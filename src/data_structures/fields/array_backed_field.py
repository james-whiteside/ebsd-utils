from abc import abstractmethod
from collections.abc import Callable
from typing import Self

from numpy import full, ndarray, dtype, isnan, array, array_equal

from src.data_structures.field import FieldLike, FieldType, FieldValueError, FieldLookupError, FieldNullError, \
    FieldSizeMismatchError


class ArrayBackedField[VALUE_TYPE](FieldLike[VALUE_TYPE]):
    def __init__(
            self,
            width: int,
            height: int,
            field_type: FieldType,
            array_type: dtype,
            default_value: VALUE_TYPE | None,
            nullable: bool = False,
            value_shape: tuple[int, ...] = (),
    ):
        super().__init__(width, height, field_type, nullable)
        self._value_shape = value_shape
        self._assert_value_permitted(default_value)
        self.default_value = self._cast_value(default_value)
        self._values = full((width, height) + self._value_shape, self.default_value, dtype=array_type)

    @property
    def array(self):
        return self._values

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            value = self._cast_array(self._values[x, y])

            if value is None:
                raise FieldNullError(x, y)
            else:
                return value

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE | None) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            self._assert_value_permitted(value)
            self._values[x, y] = self._cast_value(value)

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

    @property
    @abstractmethod
    def null_value(self) -> ndarray:
        ...

    @abstractmethod
    def _cast_value(self, value: VALUE_TYPE | None) -> ndarray:
        ...

    @abstractmethod
    def _cast_array(self, value: ndarray) -> VALUE_TYPE | None:
        ...

    @staticmethod
    def _assert_valid_array(
            arr: ndarray,
            width: int,
            height: int,
            field_type: FieldType,
            nullable: bool,
            null_value: ndarray,
            value_validator: Callable[[ndarray], bool],
            value_shape: tuple[int, ...] = (),
    ) -> None:
        if arr.ndim != 2 + len(value_shape):
            raise FieldSizeMismatchError.dimensions((2 + len(value_shape), arr.ndim))

        if arr.shape[0] != width:
            raise FieldSizeMismatchError.width((width, arr.shape[0]))

        if arr.shape[1] != height:
            raise FieldSizeMismatchError.height((height, arr.shape[1]))

        if not value_validator(arr):
            raise FieldValueError.wrong_array_type(arr, field_type)

        if not nullable:
            if isnan(null_value).any() and isnan(arr).any() or array_equal(arr, null_value):
                raise FieldValueError.null_value()

    @staticmethod
    def _list_to_array(
            width: int,
            height: int,
            values: list[list[VALUE_TYPE | None]],
            null_value: ndarray,
            array_type: dtype,
            value_shape: tuple[int, ...] = (),
    ) -> ndarray:
        array_values = full((width, height) + value_shape, null_value, dtype=array_type)

        for y in range(height):
            for x in range(width):
                if values[y][x] is not None:
                    array_values[x, y] = array(values[y][x], dtype=array_type)

        return array_values
