from enum import IntEnum
from typing import Self, override

from numpy import ndarray, issubdtype, floating, int32, iinfo, array, isin, array_equal

from src.data_structures.field import FieldType, FieldValueError
from src.data_structures.fields.array_backed_field import ArrayBackedField


class EnumField[ENUM_TYPE](ArrayBackedField[IntEnum]):
    _FIELD_TYPE = FieldType.ENUM
    _ARRAY_TYPE = int32
    _NULL_VALUE = array(-1, dtype=_ARRAY_TYPE)

    def __init__(
            self,
            width: int,
            height: int,
            enum: type[IntEnum],
            default_value: int | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, self._FIELD_TYPE, self._ARRAY_TYPE, default_value, nullable)
        self._enum = enum
        self._values = {member.value for member in enum}

        for value in self._values:
            if value < 0 or value > iinfo(int32).max:
                FieldValueError.out_of_range(value, self.field_type)

    @property
    def null_value(self) -> ndarray:
        return self._NULL_VALUE

    def _cast_value(self, value: ENUM_TYPE | None) -> ndarray:
        return array(value.value, dtype=self._ARRAY_TYPE) if value is not None else self._NULL_VALUE

    def _cast_array(self, value: ndarray) -> ENUM_TYPE | None:
        return self._enum(value) if not array_equal(value, self._NULL_VALUE) else None

    @override
    def _assert_value_permitted(self, value: int | None) -> None:
        super()._assert_value_permitted(value)

        if value is not None and value not in self._values:
            raise FieldValueError.out_of_range(value, self.field_type)

    @classmethod
    def from_array(
            cls,
            width: int,
            height: int,
            enum: type[IntEnum],
            values: ndarray,
            nullable: bool = False,
    ) -> Self:
        enum_values = [member.value for member in enum]

        for value in enum_values:
            if value < 0 or value > iinfo(int32).max:
                FieldValueError.out_of_range(value, FieldType.ENUM)

        def value_validator(value: ndarray) -> bool:
            return not issubdtype(value.dtype, floating) and isin(values, [int(cls._NULL_VALUE)] + enum_values).all()

        cls._assert_valid_array(values, width, height, cls._FIELD_TYPE, nullable, cls._NULL_VALUE, value_validator)
        field = EnumField(width, height, enum, default_value=None, nullable=True)
        field._values = values
        field.nullable = nullable
        return field

    @classmethod
    def from_list(
            cls,
            width: int,
            height: int,
            enum: type[IntEnum],
            values: list[list[ENUM_TYPE | None]],
            nullable: bool = False,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values, cls._NULL_VALUE, cls._ARRAY_TYPE)
        return cls.from_array(width, height, enum, array_values, nullable=nullable)
