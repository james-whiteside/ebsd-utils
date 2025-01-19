from enum import Enum
from typing import Type, Any, Self
from src.data_structures.field import FieldType
from src.data_structures.phase import CrystalFamily, BravaisLattice


class FieldNullError(ValueError):
    def __init__(self, coordinates: tuple[int, int]):
        """
        Exception raised when a null value in a nullable field is accessed.
        :param coordinates: Coordinates of the null value.
        """
        self.coordinates = coordinates
        self.message = f"Field has a null value at coordinate ({self.coordinates[0]}, {self.coordinates[1]})."
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
    def __init__(self, group_id: int = None):
        """
        Exception raised when a null value in a nullable aggregate is accessed.
        :param group_id: ID of the group containing the null value.
        """
        self.group_id = group_id
        self.message = f"Aggregate has a null value for group {self.group_id}."
        super().__init__(self.message)


class CheckAggregationError(ValueError):
    def __init__(self, group_id: int, values: tuple[int, int]):
        """
        Exception raised when a group of points in a check-aggregate have inconsistent values.
        :param group_id: ID of the group.
        :param values: Inconsistent values.
        """
        self.group_id = group_id
        self.values = values
        self.message = f"Value field for check aggregate has inconsistent values {self.values[0]} and {self.values[1]} for group {self.group_id}."
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
        Exception thrown when attempting to use properties of an unimplemented crystal family or Bravais lattice.
        :param symmetry_group: The crystal family or Bravais lattice.
        """
        self.symmetry_type: Type = type(symmetry_group)
        self.symmetry_group = symmetry_group
        self.message = f"Function or method not implemented for {self.symmetry_type.__name__}: {self.symmetry_group.value}"
        super().__init__(self.message)


class FieldTypeError(TypeError):
    def __init__(self, field_type: FieldType, reason: str):
        """
        Exception thrown when attempting to perform an operation on a field with an inappropriate type.
        :param field_type: The field type.
        :param reason: The reason the field type is inappropriate.
        """
        self.field_type = field_type
        self.reason = reason
        self.message = f"{self.reason}: {self.field_type.value}."
        super().__init__(self.message)

    @classmethod
    def wrong_type(cls, field_type: FieldType, correct_type: FieldType) -> Self:
        reason = f"{correct_type.value.capitalize()} field required but a field of the wrong type was provided"
        return FieldTypeError(field_type, reason)

    @classmethod
    def lacks_property(cls, field_type: FieldType, property: str) -> Self:
        reason = f"Field type is not {property}"
        return FieldTypeError(field_type, reason)

