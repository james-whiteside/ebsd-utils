from typing import Self, Optional

from numpy import ndarray, dtype, float64, nan, isnan, full

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField
from src.utilities.utils import classproperty


class MatrixField(ArrayBackedField[ndarray, float]):
    def __init__(
            self,
            width: int,
            height: int,
            default_value: ndarray | None,
            nullable: bool = False,
            matrix_shape: tuple[int, int] = (3, 3),
    ):
        super().__init__(width, height, FieldType.MATRIX, default_value, nullable, matrix_shape)

    @classproperty
    def _array_type(cls) -> dtype:
        return dtype(float64)

    @classproperty
    def _value_dimensions(cls) -> int:
        return 2

    @classproperty
    def _null_value(cls) -> float:
        return nan

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return True

    def _cast_value(self, value: ndarray | None) -> ndarray:
        return value if value is not None else full(self._value_shape, nan, dtype=float64)

    @classmethod
    def _cast_array(cls, value: ndarray) -> ndarray | None:
        return value if not isnan(value).any() else None

    @classmethod
    def from_array(
            cls,
            width: int,
            height: int,
            values: ndarray,
            nullable: bool = False,
            matrix_shape: tuple[int, int] = (3, 3),
            **kwargs,
    ) -> Self:
        cls._assert_valid_array(values, width, height, nullable)
        field = MatrixField(width, height, default_value=None, nullable=True, matrix_shape=matrix_shape)
        field._values = values
        field.nullable = nullable
        return field

    @classmethod
    def from_list(
            cls,
            width: int,
            height: int,
            values: list[list[Optional[ndarray]]],
            nullable: bool = False,
            matrix_shape: tuple[int, int] = (3, 3),
            **kwargs,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values, value_shape=matrix_shape)
        return cls.from_array(width, height, array_values, nullable=nullable, matrix_shape=matrix_shape, **kwargs)
