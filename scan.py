from enum import Enum
import numpy
from fileloader import Material
from transforms import AxisSet, reduce_matrix, euler_rotation_matrix


class FieldType(Enum):
    DISCRETE = int
    SCALAR = float
    VECTOR = tuple[float, float, float]
    MATRIX = numpy.ndarray


class Field[T]:
    def __init__(
        self,
        width: int,
        height: int,
        field_type: FieldType,
        default_value: T = None,
        values: list[list[T]] = None,
    ):
        self.width = width
        self.height = height
        self.field_type = field_type

        if default_value is None and values is None:
            raise ValueError(f"Either default field value or array of field values must be provided.")

        if default_value is not None and not isinstance(default_value, self.field_type.value):
            raise ValueError(f"Type of default field value {type(default_value)} does not match field type {self.field_type.value}.")

        self.default_value = default_value
        self._values = list()

        for y in range(self.height):
            self._values.append(list())

            for x in range(self.width):
                if values is not None:
                    try:
                        value = values[y][x]
                    except IndexError:
                        raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of provided value array.")

                    if not isinstance(value, self.field_type.value):
                        raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")

                    self._values.append(values[y][x])
                else:
                    self._values.append(self.default_value)

    def get_value(self, x: int, y: int) -> T:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            return self._values[y][x]

    def set_value(self, x: int, y: int, value: T) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        elif not isinstance(value, self.field_type.value):
            raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")
        else:
            self._values[y][x] = value


class DiscreteFieldMapper[T]:
    def __init__(self, mapping: dict[int, T], discrete_field: Field[int]):
        self._mapping = mapping
        self._field = discrete_field

    def get_value(self, x: int, y: int) -> T:
        key = self._field.get_value(x, y)
        return self._mapping[key]

    def set_value(self, x: int, y: int, value: T) -> None:
        for key in self._mapping:
            if self._mapping[key] == value:
                self._field.set_value(x, y, key)
                return

        raise KeyError(f"Value is not within permitted values of discrete-valued field.")


class Scan:
    def __init__(
        self,
        file_reference: str,
        width: int,
        height: int,
        phases: dict[int, Material],
        phase_id_values: list[list[int]],
        euler_angle_values: list[list[tuple[float, float, float]]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
        axis_set: AxisSet = AxisSet.ZXZ,
    ):
        self.file_reference = file_reference
        self.width = width
        self.height = height
        self.phases = phases
        self.axis_set = axis_set
        self._phase_id = Field(self.width, self.height, FieldType.DISCRETE, values=phase_id_values)
        self.euler_angles = Field(self.width, self.height, FieldType.VECTOR, values=euler_angle_values)
        self.pattern_quality = Field(self.width, self.height, FieldType.SCALAR, values=pattern_quality_values)
        self.index_quality = Field(self.width, self.height, FieldType.SCALAR, values=index_quality_values)
        self._reduced_euler_rotation_matrices = None

    @property
    def phase(self) -> DiscreteFieldMapper[Material]:
        return DiscreteFieldMapper(self.phases, self._phase_id)

    @property
    def reduced_euler_rotation_matrices(self) -> Field[numpy.ndarray]:
        if self._reduced_euler_rotation_matrices is None:
            raise AttributeError("Reduced euler rotation matrix field not initialised.")
        else:
            return self._reduced_euler_rotation_matrices

    def init_reduced_euler_rotation_matrices(self) -> None:
        values = list()

        for y in range(self.height):
            values.append(list())

            for x in range(self.width):
                axis_set = self.axis_set
                euler_angles = self.euler_angles.get_value(x, y)
                crystal_family = self.phase.get_value(x, y).lattice_type.get_family()
                values[y].append(reduce_matrix(euler_rotation_matrix(axis_set, euler_angles), crystal_family))

        self._reduced_euler_rotation_matrices = Field(self.width, self.height, FieldType.MATRIX, values=values)
