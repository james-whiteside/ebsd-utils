# -*- coding: utf-8 -*-

from enum import Enum
from typing import Type, Any, Self
from src.data_structures.field import FieldType
from src.data_structures.phase import CrystalFamily, BravaisLattice


class FieldNullError(ValueError):
    def __init__(self, x: int, y: int):
        """
        Exception raised when a null value in a nullable field is accessed.
        :param x: The x-coordinate of the null value.
        :param y: The y-coordinate of the null value.
        """
        self.x = x
        self.y = y
        self.message = f"Field has a null value at coordinate ({self.x}, {self.y})."
        super().__init__(self.message)


class FieldSizeMismatchError(ValueError):
    class Dimension(Enum):
        WIDTH = "width"
        HEIGHT = "height"

    def __init__(self, dimension: Dimension, values: tuple[int, int]):
        """
        Exception raised when attempting to derive a new field from input fields with mismatched sizes.
        :param dimension: Dimension of mismatch.
        :param values: Mismatched values.
        """
        self.dimension = dimension
        self.values = values
        self.message = f"{self.dimension.name.title()}s {self.values[0]} and {self.values[1]} of supplied fields do not match."
        super().__init__(self.message)

    @classmethod
    def width(cls, values: tuple[int, int]) -> Self:
        return FieldSizeMismatchError(FieldSizeMismatchError.Dimension.WIDTH, values)

    @classmethod
    def height(cls, values: tuple[int, int]) -> Self:
        return FieldSizeMismatchError(FieldSizeMismatchError.Dimension.HEIGHT, values)


class AggregateNullError(ValueError):
    def __init__(self, id: int = None):
        """
        Exception raised when a null value in a nullable aggregate is accessed.
        :param id: ID of the group containing the null value.
        """
        self.id = id
        self.message = f"Aggregate has a null value for group {self.id}."
        super().__init__(self.message)


class CheckAggregationError(ValueError):
    def __init__(self, id: int, values: tuple[int, int]):
        """
        Exception raised when a group of points in a check-aggregate have inconsistent values.
        :param id: ID of the group.
        :param values: Inconsistent values.
        """
        self.id = id
        self.values = values
        self.message = f"Value field for check aggregate has inconsistent values {self.values[0]} and {self.values[1]} for group {self.id}."
        super().__init__(self.message)


class PhaseMissingError(LookupError):
    def __init__(self, global_id: int):
        """
        Exception raised when attempting to lookup missing phase data.
        :param global_id: ID of the phase as per the Pathfinder database.
        """
        self.global_id = global_id
        self.message = f"No data available for phase {self.global_id}."
        super().__init__(self.message)


class SymmetryNotImplementedError(NotImplementedError):
    def __init__(self, symmetry_group: CrystalFamily | BravaisLattice):
        """
        Exception raised when attempting to use properties of an unimplemented crystal family or Bravais lattice.
        :param symmetry_group: The crystal family or Bravais lattice.
        """
        self.symmetry_type: Type = type(symmetry_group)
        self.symmetry_group = symmetry_group
        self.message = f"Function or method not implemented for {self.symmetry_type.__name__}: {self.symmetry_group.value}"
        super().__init__(self.message)


class FieldTypeError(TypeError):
    def __init__(self, field_type: FieldType, reason: str):
        """
        Exception raised when attempting to perform an operation on a field with an inappropriate type.
        :param field_type: The field type.
        :param reason: The reason the field type is inappropriate.
        """
        self.field_type = field_type
        self.reason = reason
        self.message = f"{self.reason}: {self.field_type.value}"
        super().__init__(self.message)

    @classmethod
    def wrong_type(cls, field_type: FieldType, correct_type: FieldType) -> Self:
        reason = f"{correct_type.value.capitalize()} field required but a field of the wrong type was provided"
        return FieldTypeError(field_type, reason)

    @classmethod
    def lacks_property(cls, field_type: FieldType, property: str) -> Self:
        reason = f"Field type is not {property}"
        return FieldTypeError(field_type, reason)


class FieldValueError(ValueError):
    def __init__(self, value: Any, reason: str):
        """
        Exception raised when attempting to write an inappropriate value to a field.
        :param value: The value.
        :param reason: The reason the value is inappropriate.
        """
        self.value = value
        self.reason = reason
        self.message = f"{self.reason}: {self.value}"
        super().__init__(self.message)

    @classmethod
    def null_value(cls) -> Self:
        reason = "Field is not nullable but a null value was provided"
        return FieldValueError(None, reason)

    @classmethod
    def wrong_type(cls, value: Any, field_type: FieldType) -> Self:
        reason = f"Field is of type {field_type.type.__name__} but a value of type {type(value).__name__} was provided"
        return FieldValueError(value, reason)

    @classmethod
    def wrong_length(cls, value: tuple, field_type: FieldType) -> Self:
        reason = f"Tuple field has size {field_type.size} but a tuple value of length {len(value)} was provided"
        return FieldValueError(value, reason)

    @classmethod
    def not_in_mapping(cls, value: int) -> Self:
        reason = f"Value is not within provided mapping for discrete field mapper"
        return FieldValueError(value, reason)

    @classmethod
    def no_reverse_mapping(cls, value: Any) -> Self:
        reason = "Cannot set value as functional field mapper does not have reverse mapping defined"
        return FieldValueError(value, reason)

    @classmethod
    def not_valid_map_value(cls, value: tuple[float, float, float]) -> Self:
        reason = "Map field may only take tuples of values between 0.0 and 1.0 but an invalid value was provided"
        return FieldValueError(value, reason)


class InvalidEncodingError(ValueError):
    def __init__(self, value: Any, enum: Type[Enum]):
        """
        Exception raised when attempting to construct an enum from an invalid encoding.
        :param value: The encoding.
        :param enum: The enum class.
        """
        self.value = value
        self.enum = enum
        self.message = f"Value is not a valid {self.enum.__name__} code: {self.value}"
        super().__init__(self.message)


class AggregateLookupError(KeyError):
    def __init__(self, id: int):
        """
        Exception raised when attempting to look up an invalid group ID in an aggregate.
        :param id: The group ID.
        """
        self.id = id
        self.message = f"Aggregate does not contain a group with ID: {self.id}"
        super().__init__(self.message)


class FieldLookupError(IndexError):
    def __init__(self, x: int, y: int):
        """
        Exception raised when attempting to access out-of-bounds coordinates of a field.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        """
        self.x = x
        self.y = y
        self.message = f"Coordinates are out of bound of field: ({self.x}, {self.y})"
        super().__init__(self.message)
