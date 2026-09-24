from abc import abstractmethod
from typing import Self

from numpy import full, ndarray, dtype, isnan, any as array_any, array

from src.data_structures.field import FieldLike, FieldType, FieldValueError, FieldLookupError, FieldNullError, \
    FieldSizeMismatchError
from src.utilities.utils import classproperty


class ArrayBackedField[VALUE_TYPE, ARRAY_TYPE](FieldLike[VALUE_TYPE]):
    def __init__(
            self,
            width: int,
            height: int,
            field_type: FieldType,
            default_value: VALUE_TYPE | None,
            nullable: bool = False,
            value_shape: tuple[int, ...] = (),
    ):
        super().__init__(width, height, field_type, nullable)
        self._value_shape = value_shape
        self._assert_value_permitted(default_value)
        self.default_value = self._cast_value(default_value)
        self._values = full((width, height) + self._value_shape, self.default_value, dtype=self._array_type)

    @property
    def array(self):
        return self._values

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            value = self._cast_array(self._values[x][y])

            if value is None:
                raise FieldNullError(x, y)
            else:
                return value

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE | None) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise FieldLookupError(x, y)
        else:
            self._assert_value_permitted(value)
            self._values[x][y] = self._cast_value(value)

    @classmethod
    def _has_null_value(cls, array: ndarray) -> bool:
        if isnan(cls._null_value):
            return isnan(array).any()
        else:
            return array_any((array == cls._null_value))

    @classmethod
    def _assert_valid_array(cls, array: ndarray, width: int, height: int, nullable: bool) -> None:
        if array.ndim != 2 + cls._value_dimensions:
            raise FieldSizeMismatchError.dimensions((2 + cls._value_dimensions, array.ndim))

        if array.shape[0] != width:
            raise FieldSizeMismatchError.width((width, array.shape[0]))

        if array.shape[1] != height:
            raise FieldSizeMismatchError.height((height, array.shape[1]))

        if not cls._is_valid_type(array):
            raise FieldValueError.wrong_array_type(array, super().field_type)

        if not nullable and cls._has_null_value(array):
            raise FieldValueError.null_value()

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

    @classproperty
    @abstractmethod
    def _array_type(cls) -> dtype:
        ...

    @classproperty
    @abstractmethod
    def _value_dimensions(cls) -> int:
        ...

    @classproperty
    @abstractmethod
    def _null_value(cls) -> ARRAY_TYPE:
        ...

    @classmethod
    @abstractmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        ...

    @abstractmethod
    def _cast_value(self, value: VALUE_TYPE | None) -> ARRAY_TYPE:
        ...

    @classmethod
    @abstractmethod
    def _cast_array(cls, value: ndarray) -> VALUE_TYPE | None:
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, width: int, height: int, values: ndarray, nullable: bool = False, **kwargs) -> Self:
        ...

    @classmethod
    @abstractmethod
    def from_list(cls, width: int, height: int, values: ndarray, nullable: bool = False, **kwargs) -> Self:
        ...

    @classmethod
    def _list_to_array(
            cls,
            width: int,
            height: int,
            values: list[list[VALUE_TYPE | None]],
            value_shape: tuple[int, ...] = (),
    ) -> ndarray:
        array_values = full((width, height) + value_shape, cls._null_value, dtype=cls._array_type)

        for y in range(height):
            for x in range(width):
                if values[y][x] is not None:
                    array_values[x][y] = array(values[y][x])

        return array_values
