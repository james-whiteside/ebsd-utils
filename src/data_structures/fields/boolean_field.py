from typing import Self

from numpy import ndarray, issubdtype, floating, all as array_all, int8, array, array_equal

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField


class BooleanField(ArrayBackedField[bool]):
    _FIELD_TYPE = FieldType.BOOLEAN
    _ARRAY_TYPE = int8
    _NULL_VALUE = array(-1, dtype=_ARRAY_TYPE)
    _TRUE_VALUE = array(1, dtype=_ARRAY_TYPE)
    _FALSE_VALUE = array(0, dtype=_ARRAY_TYPE)

    def __init__(
            self,
            width: int,
            height: int,
            default_value: bool | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, self._FIELD_TYPE, self._ARRAY_TYPE, default_value, nullable)

    @property
    def null_value(self) -> ndarray:
        return self._NULL_VALUE

    def _cast_value(self, value: bool | None) -> ndarray:
        return self._TRUE_VALUE if value else self._FALSE_VALUE if value is not None else self._NULL_VALUE

    def _cast_array(self, value: ndarray) -> bool | None:
        return array_equal(value, self._TRUE_VALUE) if not array_equal(value, self._NULL_VALUE) else None

    @classmethod
    def from_array(
            cls,
            width: int,
            height: int,
            values: ndarray,
            nullable: bool = False,
    ) -> Self:
        def value_validator(value: ndarray) -> bool:
            return not issubdtype(value.dtype, floating) and array_all((value >= -1) & (value <= 1))

        cls._assert_valid_array(values, width, height, cls._FIELD_TYPE, nullable, cls._NULL_VALUE, value_validator)
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
    ) -> Self:
        array_values = cls._list_to_array(width, height, values, cls._NULL_VALUE, cls._ARRAY_TYPE)
        return cls.from_array(width, height, array_values, nullable=nullable)
