from enum import Enum


class FieldNullError(ValueError):
    def __init__(self, coordinates: tuple[int, int] = None):
        self.coordinates = coordinates

        if self.coordinates is None:
            self.message = f"Field has a null value."
        else:
            self.message = f"Field has a null value at coordinate ({self.coordinates[0]}, {self.coordinates[1]})."

        super().__init__(self.message)


class FieldsInconsistentError(ValueError):
    class Dimension(Enum):
        WIDTH = "width"
        HEIGHT = "height"

    def __init__(self, dimension: Dimension, values: tuple[int, int]):
        self.values = values
        self.message = f"{dimension.name.title()}s {self.values[0]} and {self.values[1]} of supplied fields do not match."
        super().__init__(self.message)

    @classmethod
    def width(cls, values: tuple[int, int]):
        return cls(cls.Dimension.WIDTH, values)

    @classmethod
    def height(cls, values: tuple[int, int]):
        return cls(cls.Dimension.HEIGHT, values)


class AggregateNullError(ValueError):
    def __init__(self, group_id: int = None):
        self.group_id = group_id

        if self.group_id is None:
            self.message = f"Aggregate has a null value."
        else:
            self.message = f"Aggregate has a null value for group {group_id}."

        super().__init__(self.message)


class AggregateInconsistentError(ValueError):
    def __init__(self, group_id: int, values: tuple[int, int]):
        self.group_id = group_id
        self.values = values
        self.message = f"Value field for check aggregate has inconsistent values {values[0]} and {values[1]} for group {group_id}."
        super().__init__(self.message)


class PhaseMissingError(LookupError):
    def __init__(self, global_id: int):
        self.global_id = global_id
        self.message = f"No data available for phase {global_id}."
        super().__init__(self.message)
