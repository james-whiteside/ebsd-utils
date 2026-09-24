from typing import Self, override

from numpy import ndarray, dtype, issubdtype, floating, all as array_all, any as array_any, int32, iinfo

from src.data_structures.field import FieldType, FieldValueError
from src.data_structures.fields.array_backed_field import ArrayBackedField
from src.utilities.utils import classproperty


class DiscreteField(ArrayBackedField[int, int]):
    def __init__(
            self,
            width: int,
            height: int,
            default_value: int | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, FieldType.DISCRETE, default_value, nullable)

    @classproperty
    def _array_type(cls) -> dtype:
        return int32

    @classproperty
    def _value_dimensions(cls) -> int:
        return 0

    @classproperty
    def _null_value(cls) -> int:
        return -1

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return not issubdtype(array.dtype, floating) and array_all((array >= -1) & (array <= iinfo(int32).max))

    def _cast_value(self, value: bool | None) -> int:
        return value if value is not None else -1

    @classmethod
    def _cast_array(cls, value: ndarray) -> bool | None:
        return value if value != -1 else None

    @override
    def _assert_value_permitted(self, value: int | None) -> None:
        super()._assert_value_permitted(value)

        if value is not None and (value < 0 or value > iinfo(int32).max):
            raise FieldValueError.out_of_range(value, self.field_type)

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
        field = DiscreteField(width, height, default_value=None, nullable=True)
        field._values = values
        field.nullable = nullable
        return field

    @classmethod
    def from_list(
            cls,
            width: int,
            height: int,
            values: list[list[int | None]],
            nullable: bool = False,
            **kwargs,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values)
        return cls.from_array(width, height, array_values, nullable=nullable, **kwargs)
