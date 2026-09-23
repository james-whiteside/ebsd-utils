from typing import Self

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

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return True

    @classmethod
    def _has_null_value(cls, array: ndarray) -> bool:
        return isnan(array).any()

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
    ) -> Self:
        cls._assert_valid_array(values, width, height, nullable)
        field = MatrixField(width, height, default_value=None, nullable=nullable)
        field._values = values
        return field
