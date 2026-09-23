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

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return not issubdtype(array.dtype, floating) and array_all((array >= -1) & (array <= iinfo(int32).max))

    @classmethod
    def _has_null_value(cls, array: ndarray) -> bool:
        return array_any((array == -1))

    def _cast_value(self, value: bool | None) -> int:
        return value if value is not None else -1

    @classmethod
    def _cast_array(cls, value: ndarray) -> bool | None:
        return value if value != -1 else None

    @override
    def _assert_value_permitted(self, value: int | None) -> None:
        super()._assert_value_permitted(value)

        if value < 0 or value > iinfo(int32).max:
            raise FieldValueError.out_of_range(value, self.field_type)

    @classmethod
    def from_array(
        cls,
        width: int,
        height: int,
        values: ndarray,
        nullable: bool = False,
    ) -> Self:
        cls._assert_valid_array(values, width, height, nullable)
        field = DiscreteField(width, height, default_value=None, nullable=nullable)
        field._values = values
        return field
