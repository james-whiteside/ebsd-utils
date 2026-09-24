from typing import Self

from numpy import ndarray, dtype, issubdtype, floating, all as array_all, int8

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

    @classproperty
    def _null_value(cls) -> int:
        return -1

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return not issubdtype(array.dtype, floating) and array_all((array >= -1) & (array <= 1))

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
            **kwargs,
    ) -> Self:
        cls._assert_valid_array(values, width, height, nullable)
        field = BooleanField(width, height, default_value=None, nullable=True)
        field._values = values
        field.nullable = nullable
        return field

    @classmethod
    def from_list(
            cls,
            width: int,
            height: int,
            values: list[list[bool | None]],
            nullable: bool = False,
            **kwargs,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values)
        return cls.from_array(width, height, array_values, nullable=nullable, **kwargs)
