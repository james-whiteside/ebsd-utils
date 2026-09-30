from typing import Self, Optional

from numpy import ndarray, dtype, float64, nan, isnan, full

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField


class MatrixField(ArrayBackedField[ndarray]):
    _FIELD_TYPE = FieldType.MATRIX
    _ARRAY_TYPE = dtype(float64)
    _DEFAULT_SHAPE = (3, 3)

    @staticmethod
    def _create_null_value(matrix_shape: tuple[int, int]) -> ndarray:
        return full(matrix_shape, nan, dtype=MatrixField._ARRAY_TYPE)

    def __init__(
            self,
            width: int,
            height: int,
            default_value: ndarray | None,
            nullable: bool = False,
            matrix_shape: tuple[int, int] = _DEFAULT_SHAPE,
    ):
        self._null_value = self._create_null_value(matrix_shape)
        super().__init__(width, height, self._FIELD_TYPE, self._ARRAY_TYPE, default_value, nullable, matrix_shape)

    @property
    def null_value(self) -> ndarray:
        return self._null_value

    def _cast_value(self, value: ndarray | None) -> ndarray:
        return value if value is not None else self._null_value

    def _cast_array(self, value: ndarray) -> ndarray | None:
        return value if not isnan(value).any() else None

    @classmethod
    def from_array(
            cls,
            width: int,
            height: int,
            values: ndarray,
            nullable: bool = False,
            matrix_shape: tuple[int, int] = _DEFAULT_SHAPE,
    ) -> Self:
        cls._assert_valid_array(values, width, height, cls._FIELD_TYPE, nullable, cls._create_null_value(matrix_shape), lambda _: True, matrix_shape)
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
            matrix_shape: tuple[int, int] = _DEFAULT_SHAPE,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values, cls._create_null_value(matrix_shape), cls._ARRAY_TYPE, matrix_shape)
        return cls.from_array(width, height, array_values, nullable=nullable, matrix_shape=matrix_shape)
