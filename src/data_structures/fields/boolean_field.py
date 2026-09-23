from typing import Self

from numpy import ndarray, dtype, issubdtype, floating, all as array_all, any as array_any, int8

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField
from src.utilities.utils import classproperty


class BooleanField(ArrayBackedField[bool, int]):
    def __init__(
            self,
            width: int,
            height: int,
            default_value: bool | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, FieldType.BOOLEAN, default_value, nullable)

    @classproperty
    def _array_type(cls) -> dtype:
        return int8

    @classproperty
    def _value_dimensions(cls) -> int:
        return 0

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return not issubdtype(array.dtype, floating) and array_all((array >= -1) & (array <= 1))

    @classmethod
    def _has_null_value(cls, array: ndarray) -> bool:
        return array_any((array == -1))

    def _cast_value(self, value: bool | None) -> int:
        return 1 if value else 0 if value is not None else -1

    @classmethod
    def _cast_array(cls, value: ndarray) -> bool | None:
        return value == 1 if value != -1 else None

    @classmethod
    def from_array(
        cls,
        width: int,
        height: int,
        values: ndarray,
        nullable: bool = False,
    ) -> Self:
        cls._assert_valid_array(values, width, height, nullable)
        field = BooleanField(width, height, default_value=None, nullable=nullable)
        field._values = values
        return field
