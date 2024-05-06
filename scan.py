from enum import Enum
import numpy
from fileloader import Material
from transforms import AxisSet, reduce_matrix, euler_rotation_matrix, Axis, forward_stereographic


class FieldType(Enum):
    DISCRETE = int
    SCALAR = float
    VECTOR_2D = tuple[float, float]
    VECTOR_3D = tuple[float, float, float]
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

    def get_value_at(self, x: int, y: int) -> T:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            return self._values[y][x]

    def set_value_at(self, x: int, y: int, value: T) -> None:
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

    def get_value_at(self, x: int, y: int) -> T:
        key = self._field.get_value_at(x, y)
        return self._mapping[key]

    def set_value_at(self, x: int, y: int, value: T) -> None:
        for key in self._mapping:
            if self._mapping[key] == value:
                self._field.set_value_at(x, y, key)
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
        self.euler_angles = Field(self.width, self.height, FieldType.VECTOR_3D, values=euler_angle_values)
        self.pattern_quality = Field(self.width, self.height, FieldType.SCALAR, values=pattern_quality_values)
        self.index_quality = Field(self.width, self.height, FieldType.SCALAR, values=index_quality_values)
        self._reduced_euler_rotation_matrix = None
        self._inverse_x_pole_figure_coordinates = None
        self._inverse_y_pole_figure_coordinates = None
        self._inverse_z_pole_figure_coordinates = None

    @property
    def phase(self) -> DiscreteFieldMapper[Material]:
        return DiscreteFieldMapper(self.phases, self._phase_id)

    @property
    def reduced_euler_rotation_matrix(self) -> Field[numpy.ndarray]:
        if self._reduced_euler_rotation_matrix is None:
            raise AttributeError("Reduced euler rotation matrix field not initialised.")
        else:
            return self._reduced_euler_rotation_matrix

    def init_reduced_euler_rotation_matrices(self) -> None:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=numpy.eye(3))

        for y in range(self.height):
            for x in range(self.width):
                axis_set = self.axis_set
                euler_angles = self.euler_angles.get_value_at(x, y)
                crystal_family = self.phase.get_value_at(x, y).lattice_type.get_family()
                value = reduce_matrix(euler_rotation_matrix(axis_set, euler_angles), crystal_family)
                field.set_value_at(x, y, value)

        self._reduced_euler_rotation_matrix = field

    def inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if None in (
            self._inverse_x_pole_figure_coordinates,
            self._inverse_y_pole_figure_coordinates,
            self._inverse_z_pole_figure_coordinates,
        ):
            raise AttributeError("Inverse pole figure coordinate fields not initialised.")
        else:
            match axis:
                case Axis.X:
                    return self._inverse_x_pole_figure_coordinates
                case Axis.Y:
                    return self._inverse_y_pole_figure_coordinates
                case Axis.Z:
                    return self._inverse_z_pole_figure_coordinates

    def _gen_inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        field = Field(self.width, self.height, FieldType.VECTOR_2D, default_value=(0.0, 0.0))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == 0:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    value = forward_stereographic(*numpy.dot(rotation_matrix, axis.value).tolist())
                    field.set_value_at(x, y, value)

        return field

    def init_inverse_pole_figure_coordinates(self) -> None:
        self._inverse_x_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.X)
        self._inverse_y_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Y)
        self._inverse_z_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Z)
