from typing import Self

from numpy import ndarray, dtype, float64, nan, isnan, array

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField
from src.utilities.utils import classproperty


class Vector2DField(ArrayBackedField[tuple[float, float], float]):
    def __init__(
            self,
            width: int,
            height: int,
            default_value: tuple[float, float] | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, FieldType.VECTOR_2D, default_value, nullable, value_shape=(2,))

    @classproperty
    def _array_type(cls) -> dtype:
        return dtype(float64)

    @classproperty
    def _value_dimensions(cls) -> int:
        return 1

    @classmethod
    def _is_valid_type(cls, array: ndarray) -> bool:
        return True

    @classmethod
    def _has_null_value(cls, array: ndarray) -> bool:
        return isnan(array).any()

    def _cast_value(self, value: tuple[float, float] | None) -> ndarray:
        return array(value) if value is not None else array((nan, nan))

    @classmethod
    def _cast_array(cls, value: ndarray) -> tuple[float, float] | None:
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
        field = Vector2DField(width, height, default_value=None, nullable=nullable)
        field._values = values
        return field
