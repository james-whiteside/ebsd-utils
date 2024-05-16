import math
from enum import Enum
import numpy

import channelling
from fileloader import Material
from transforms import AxisSet, reduce_matrix, euler_rotation_matrix, Axis, forward_stereographic, rotation_angle, \
    misrotation_matrix, misrotation_tensor

UNINDEXED_PHASE_ID = 0
GENERIC_BCC_PHASE_ID = 4294967294
GENERIC_FCC_PHASE_ID = 4294967295
GENERIC_PHASE_IDS = (UNINDEXED_PHASE_ID, GENERIC_BCC_PHASE_ID, GENERIC_FCC_PHASE_ID)
GND_DENSITY_CORRECTIVE_FACTOR = 3.6


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
        pixel_size: float,
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
        self.pixel_size = pixel_size
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
        self._kernel_average_misorientation = None
        self._misrotation_x_tensor = None
        self._misrotation_y_tensor = None
        self._nye_tensor = None
        self._geometrically_necessary_dislocation_density = None
        self.beam_atomic_number = None
        self.beam_energy = None
        self.beam_vector = None
        self._channelling_fraction = None

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
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
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

    @property
    def kernel_average_misorientation(self) -> Field[float]:
        if self._kernel_average_misorientation is None:
            raise AttributeError("Kernel average misorientation field not initialised.")
        else:
            return self._kernel_average_misorientation

    def init_kernel_average_misorientation(self) -> None:
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    total = 0.0
                    count = 4
                    kernel = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
                    rotation_matrix_1 = self.reduced_euler_rotation_matrix.get_value_at(x, y)

                    for dx, dy in kernel:
                        try:
                            if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                                rotation_matrix_2 = self.reduced_euler_rotation_matrix.get_value_at(x + dx, y + dy)
                                total += rotation_angle(misrotation_matrix(rotation_matrix_1, rotation_matrix_2))
                            else:
                                count -= 1
                        except IndexError:
                            count -= 1

                    if count == 0:
                        continue
                    else:
                        value = total / count
                        field.set_value_at(x, y, value)

        self._kernel_average_misorientation = field

    def misrotation_tensor(self, axis: Axis) -> Field[numpy.ndarray]:
        if None in (
            self._misrotation_x_tensor,
            self._misrotation_y_tensor,
        ):
            raise AttributeError("Misrotation tensor fields not initialised.")
        else:
            match axis:
                case Axis.X:
                    return self._misrotation_x_tensor
                case Axis.Y:
                    return self._misrotation_y_tensor
                case Axis.Z:
                    raise ValueError("Misrotation data not available for z-axis intervals.")

    def _gen_misrotation_tensor(self, axis: Axis) -> Field[numpy.ndarray]:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=numpy.zeros((3, 3)))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    total = numpy.zeros((3, 3))
                    count = 2

                    match axis:
                        case Axis.X:
                            kernel = [(-1, 0), (+1, 0)]
                        case Axis.Y:
                            kernel = [(0, -1), (0, +1)]
                        case Axis.Z:
                            raise ValueError("Misrotation data not available for z-axis intervals.")

                    rotation_matrix_1 = self.reduced_euler_rotation_matrix.get_value_at(x, y)

                    for dx, dy in kernel:
                        try:
                            if self.phase.get_value_at(x, y) == self.phase.get_value_at(x + dx, y + dy):
                                rotation_matrix_2 = self.reduced_euler_rotation_matrix.get_value_at(x + dx, y + dy)
                                total += misrotation_tensor(misrotation_matrix(rotation_matrix_1, rotation_matrix_2), self.pixel_size)
                            else:
                                count -= 1
                        except IndexError:
                            count -= 1

                    if count == 0:
                        continue
                    else:
                        value = total / count
                        field.set_value_at(x, y, value)

        return field

    def init_misrotation_tensor(self) -> None:
        self._misrotation_x_tensor = self._gen_misrotation_tensor(Axis.X)
        self._misrotation_y_tensor = self._gen_misrotation_tensor(Axis.Y)

    @property
    def nye_tensor(self) -> Field[numpy.ndarray]:
        if self._nye_tensor is None:
            raise AttributeError("Nye tensor field not initialised.")
        else:
            return self._nye_tensor

    def init_nye_tensor(self) -> None:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=numpy.zeros((3, 3)))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    value = numpy.array((
                        (0.0, self.misrotation_tensor(Axis.X).get_value_at(x, y)[2][0], -self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][0]),
                        (-self.misrotation_tensor(Axis.Y).get_value_at(x, y)[2][1], 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][1]),
                        (0.0, 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][2] - self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][2])
                    ))

                    field.set_value_at(x, y, value)

        self._nye_tensor = field

    @property
    def geometrically_necessary_dislocation_density(self) -> Field[float]:
        if self._geometrically_necessary_dislocation_density is None:
            raise AttributeError("Geometrically necessary dislocation density field not initialised.")
        else:
            return self._geometrically_necessary_dislocation_density

    def init_geometrically_necessary_dislocation_density(self) -> None:
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    nye_tensor_norm = sum(abs(element) for element in self.nye_tensor.get_value_at(x, y))
                    close_pack_distance = self.phase.get_value_at(x, y).close_pack_distance
                    value = (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm
                    field.set_value_at(x, y, value)

        self._geometrically_necessary_dislocation_density = field

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            raise AttributeError("Channelling fraction field not initialised.")
        else:
            return self._channelling_fraction

    def init_channelling_fraction(
        self,
        beam_atomic_number: int,
        beam_energy: float,
        beam_vector: tuple[float, float, float]
    ) -> None:
        self.beam_atomic_number = beam_atomic_number
        self.beam_energy = beam_energy
        self.beam_vector = beam_vector
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        channel_data = {
            local_id: channelling.load_crit_data(self.beam_atomic_number, phase.global_id, self.beam_energy)
            for local_id, phase in self.phases.items() if phase.global_id != UNINDEXED_PHASE_ID
        }

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    effective_beam_vector = numpy.dot(rotation_matrix, self.beam_vector).to_list()
                    value = channelling.fraction(effective_beam_vector, channel_data[self._phase_id.get_value_at(x, y)])
                    field.set_value_at(x, y, value)

        self._channelling_fraction = field
