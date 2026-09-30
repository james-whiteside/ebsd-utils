from typing import Self

from numpy import ndarray, dtype, float64, nan, isnan, array

from src.data_structures.field import FieldType
from src.data_structures.fields.array_backed_field import ArrayBackedField


class Vector3DField(ArrayBackedField[tuple[float, float, float]]):
    _FIELD_TYPE = FieldType.VECTOR_3D
    _ARRAY_TYPE= dtype(float64)
    _NULL_VALUE = array((nan, nan, nan), dtype=_ARRAY_TYPE)
    _SHAPE = (3,)

    def __init__(
            self,
            width: int,
            height: int,
            default_value: tuple[float, float, float] | None,
            nullable: bool = False,
    ):
        super().__init__(width, height, self._FIELD_TYPE, self._ARRAY_TYPE, default_value, nullable, self._SHAPE)

    @property
    def null_value(self) -> ndarray:
        return self._NULL_VALUE

    def _cast_value(self, value: tuple[float, float, float] | None) -> ndarray:
        return array(value, dtype=self._ARRAY_TYPE) if value is not None else self._NULL_VALUE

    def _cast_array(self, value: ndarray) -> tuple[float, float, float] | None:
        return value if not isnan(value).any() else None

    @classmethod
    def from_array(
            cls,
            width: int,
            height: int,
            values: ndarray,
            nullable: bool = False,
    ) -> Self:
        cls._assert_valid_array(values, width, height, cls._FIELD_TYPE, nullable, cls._NULL_VALUE, lambda _: True, cls._SHAPE)
        field = Vector3DField(width, height, default_value=None, nullable=True)
        field._values = values
        field.nullable = nullable
        return field

    @classmethod
    def from_list(
            cls,
            width: int,
            height: int,
            values: list[list[tuple[float, float, float] | None]],
            nullable: bool = False,
    ) -> Self:
        array_values = cls._list_to_array(width, height, values, cls._NULL_VALUE, cls._ARRAY_TYPE, cls._SHAPE)
        return cls.from_array(width, height, array_values, nullable=nullable)
