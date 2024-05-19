# -*- coding: utf-8 -*-

import math
from collections.abc import Iterator, Callable
from enum import Enum
from typing import Self

from numpy import ndarray, array, zeros, eye, dot

from fileloader import get_materials
from geometry import (
    Axis,
    AxisSet,
    euler_rotation_matrix,
    reduce_matrix,
    rotation_angle,
    misrotation_matrix,
    misrotation_tensor,
    forward_stereographic,
)
from material import Material
from channelling import load_crit_data, fraction
from clustering import ClusterCategory, dbscan

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
    MATRIX = ndarray


class Field[VALUE_TYPE]:
    def __init__(
        self,
        width: int,
        height: int,
        field_type: FieldType,
        default_value: VALUE_TYPE = None,
        values: list[list[VALUE_TYPE]] = None,
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

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        else:
            return self._values[y][x]

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError(f"Coordinate ({x}, {y}) is out of bounds of field.")
        elif not isinstance(value, self.field_type.value):
            raise ValueError(f"Type of provided value {type(value)} does not match field type {self.field_type.value}.")
        else:
            self._values[y][x] = value


class DiscreteFieldMapper[VALUE_TYPE]:
    def __init__(self, mapping: dict[int, VALUE_TYPE], discrete_field: Field[int]):
        self._mapping = mapping
        self._field = discrete_field

    def get_value_at(self, x: int, y: int) -> VALUE_TYPE:
        key = self._field.get_value_at(x, y)
        return self._mapping[key]

    def set_value_at(self, x: int, y: int, value: VALUE_TYPE) -> None:
        for key in self._mapping:
            if self._mapping[key] == value:
                self._field.set_value_at(x, y, key)
                return

        raise KeyError(f"Value is not within permitted values of discrete-valued field: {value}")


class FunctionalFieldMapper[INPUT_TYPE, OUTPUT_TYPE]:
    def __init__(
        self,
        forward_mapping: Callable[INPUT_TYPE, OUTPUT_TYPE],
        field: Field[INPUT_TYPE],
        reverse_mapping: Callable[OUTPUT_TYPE, INPUT_TYPE] = None,
    ):
        self._forward_mapping = forward_mapping
        self._reverse_mapping = reverse_mapping
        self._field = field

    def get_value_at(self, x: int, y: int) -> OUTPUT_TYPE:
        return self._forward_mapping(self._field.get_value_at(x, y))

    def set_value_at(self, x: int, y: int, value: OUTPUT_TYPE) -> None:
        if self._reverse_mapping is None:
            raise AttributeError("Functional field mapper does not have a reverse mapping defined.")
        else:
            self._field.set_value_at(x, y, self._reverse_mapping(value))


class Scan:
    beam_axis = Axis.Y

    def __init__(
        self,
        file_reference: str,
        width: int,
        height: int,
        phases: dict[int, Material],
        phase_id_values: list[list[int]],
        euler_angle_degrees_values: list[list[tuple[float, float, float]]],
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
        self.euler_angles = None
        self.euler_angles_degrees = Field(self.width, self.height, FieldType.VECTOR_3D, values=euler_angle_degrees_values)
        self.pattern_quality = Field(self.width, self.height, FieldType.SCALAR, values=pattern_quality_values)
        self.index_quality = Field(self.width, self.height, FieldType.SCALAR, values=index_quality_values)
        self._reduced_euler_rotation_matrix = None
        self._inverse_x_pole_figure_coordinates = None
        self._inverse_y_pole_figure_coordinates = None
        self._inverse_z_pole_figure_coordinates = None
        self._kernel_average_misorientation = None
        self.pixel_size = None
        self._misrotation_x_tensor = None
        self._misrotation_y_tensor = None
        self._nye_tensor = None
        self._geometrically_necessary_dislocation_density = None
        self.beam_atomic_number = None
        self.beam_energy = None
        self.beam_tilt = None
        self._channelling_fraction = None
        self.core_point_neighbour_threshold = None
        self.neighbourhood_radius = None
        self.cluster_count = None
        self._orientation_clustering_category_id = None
        self._orientation_cluster_id = None

    @classmethod
    def from_pathfinder_file(cls, data_path: str, materials_path: str, file_reference: str = None) -> Self:
        if file_reference is None:
            file_reference = data_path.split("/")[-1].split(".")[0].lstrip("p")

        with open(data_path, "r") as file:
            materials = dict()
            file_materials = get_materials(materials_path)
            file.readline()

            while True:
                line = file.readline().rstrip("\n").split(",")

                if line == ["Map Size:"]:
                    break

                local_id = int(line[0])
                global_id = int(line[2])

                try:
                    material = file_materials[global_id]
                except KeyError:
                    raise KeyError(f"No material in materials file with ID: {global_id}")

                materials[local_id] = material

            width = int(file.readline().rstrip("\n").split(",")[1])
            height = int(file.readline().rstrip("\n").split(",")[1])
            phase_id_values = list()
            euler_angle_degrees_values = list()
            index_quality_values = list()
            pattern_quality_values = list()
            file.readline()
            file.readline()

            for y in range(height):
                phase_id_values.append(list())
                euler_angle_degrees_values.append(list())
                index_quality_values.append(list())
                pattern_quality_values.append(list())

                for x in range(width):
                    line = file.readline().rstrip("\n").split(",")
                    phase_id_values[y].append(int(line[2]))
                    euler_angle_degrees_values[y].append((float(line[3]), float(line[4]), float(line[5])))
                    index_quality_values[y].append(float(line[6]))
                    pattern_quality_values[y].append(float(line[7]))

        return Scan(
            file_reference=file_reference,
            width=width,
            height=height,
            phases=materials,
            phase_id_values=phase_id_values,
            euler_angle_degrees_values=euler_angle_degrees_values,
            pattern_quality_values=pattern_quality_values,
            index_quality_values=index_quality_values,
        )

    def to_pathfinder_file(
        self,
        path: str,
        show_phases: bool = True,
        show_map_size: bool = True,
        show_map_scale: bool = False,
        show_channelling_params: bool = False,
        show_clustering_params: bool = False,
        show_scan_coordinates: bool = True,
        show_phase: bool = True,
        show_euler_angles: bool = True,
        show_index_quality: bool = True,
        show_pattern_quality: bool = True,
        show_inverse_x_pole_figure_coordinates: bool = False,
        show_inverse_y_pole_figure_coordinates: bool = False,
        show_inverse_z_pole_figure_coordinates: bool = False,
        show_kernel_average_misorientation: bool = False,
        show_geometrically_necessary_dislocation_density: bool = False,
        show_channelling_fraction: bool = False,
        show_orientation_cluster: bool = False,
    ):
        rows = self._rows(
            show_phases=show_phases,
            show_map_size=show_map_size,
            show_map_scale=show_map_scale,
            show_channelling_params=show_channelling_params,
            show_clustering_params=show_clustering_params,
            show_scan_coordinates=show_scan_coordinates,
            show_phase=show_phase,
            show_euler_angles=show_euler_angles,
            show_index_quality=show_index_quality,
            show_pattern_quality=show_pattern_quality,
            show_inverse_x_pole_figure_coordinates=show_inverse_x_pole_figure_coordinates,
            show_inverse_y_pole_figure_coordinates=show_inverse_y_pole_figure_coordinates,
            show_inverse_z_pole_figure_coordinates=show_inverse_z_pole_figure_coordinates,
            show_kernel_average_misorientation=show_kernel_average_misorientation,
            show_geometrically_necessary_dislocation_density=show_geometrically_necessary_dislocation_density,
            show_channelling_fraction=show_channelling_fraction,
            show_orientation_cluster=show_orientation_cluster,
        )

        with open(path, "w") as file:
            for row in rows:
                file.write(f"{row}\n")

    def _rows(
        self,
        show_phases: bool = True,
        show_map_size: bool = True,
        show_map_scale: bool = False,
        show_channelling_params: bool = False,
        show_clustering_params: bool = False,
        show_scan_coordinates: bool = True,
        show_phase: bool = True,
        show_euler_angles: bool = True,
        show_index_quality: bool = True,
        show_pattern_quality: bool = True,
        show_inverse_x_pole_figure_coordinates: bool = False,
        show_inverse_y_pole_figure_coordinates: bool = False,
        show_inverse_z_pole_figure_coordinates: bool = False,
        show_kernel_average_misorientation: bool = False,
        show_geometrically_necessary_dislocation_density: bool = False,
        show_channelling_fraction: bool = False,
        show_orientation_cluster: bool = False,
    ) -> Iterator[str]:
        for row in self._metadata_rows(
            show_phases=show_phases,
            show_map_size=show_map_size,
            show_map_scale=show_map_scale,
            show_channelling_params=show_channelling_params,
            show_clustering_params=show_clustering_params,
        ):
            yield row

        for row in self._header_rows(
            show_scan_coordinates=show_scan_coordinates,
            show_phase=show_phase,
            show_euler_angles=show_euler_angles,
            show_index_quality=show_index_quality,
            show_pattern_quality=show_pattern_quality,
            show_inverse_x_pole_figure_coordinates=show_inverse_x_pole_figure_coordinates,
            show_inverse_y_pole_figure_coordinates=show_inverse_y_pole_figure_coordinates,
            show_inverse_z_pole_figure_coordinates=show_inverse_z_pole_figure_coordinates,
            show_kernel_average_misorientation=show_kernel_average_misorientation,
            show_geometrically_necessary_dislocation_density=show_geometrically_necessary_dislocation_density,
            show_channelling_fraction=show_channelling_fraction,
            show_orientation_cluster=show_orientation_cluster,
        ):
            yield row

        for row in self._data_rows(
            show_scan_coordinates=show_scan_coordinates,
            show_phase=show_phase,
            show_euler_angles=show_euler_angles,
            show_index_quality=show_index_quality,
            show_pattern_quality=show_pattern_quality,
            show_inverse_x_pole_figure_coordinates=show_inverse_x_pole_figure_coordinates,
            show_inverse_y_pole_figure_coordinates=show_inverse_y_pole_figure_coordinates,
            show_inverse_z_pole_figure_coordinates=show_inverse_z_pole_figure_coordinates,
            show_kernel_average_misorientation=show_kernel_average_misorientation,
            show_geometrically_necessary_dislocation_density=show_geometrically_necessary_dislocation_density,
            show_channelling_fraction=show_channelling_fraction,
            show_orientation_cluster=show_orientation_cluster,
        ):
            yield row

    def _metadata_rows(
        self,
        show_phases: bool = True,
        show_map_size: bool = True,
        show_map_scale: bool = False,
        show_channelling_params: bool = False,
        show_clustering_params: bool = False,
    ) -> str:
        if show_phases:
            yield "Phases"

            for local_id, phase in self.phases.items():
                yield f"{local_id},{phase.name},{phase.global_id}"

        if show_map_size:
            yield f"Map Size:"
            yield f"X,{self.width}"
            yield f"Y,{self.height}"

        if show_map_scale:
            yield f"Map scale:"
            yield f"Pixel size (μm),{self.pixel_size_micrometres}"

        if show_channelling_params:
            yield f"Channelling:"
            yield f"Atomic number,{self.beam_atomic_number}"
            yield f"Energy (eV),{self.beam_energy}"
            yield f"Tilt (deg),{self.beam_tilt_degrees}"

        if show_clustering_params:
            yield f"Clustering:"
            yield f"Core point threshold,{self.core_point_neighbour_threshold}"
            yield f"Point neighbourhood radius (deg),{self.neighbourhood_radius_degrees}"
            yield f"Cluster count,{self.cluster_count}"

    @staticmethod
    def _header_rows(
        show_scan_coordinates: bool = True,
        show_phase: bool = True,
        show_euler_angles: bool = True,
        show_index_quality: bool = True,
        show_pattern_quality: bool = True,
        show_inverse_x_pole_figure_coordinates: bool = False,
        show_inverse_y_pole_figure_coordinates: bool = False,
        show_inverse_z_pole_figure_coordinates: bool = False,
        show_kernel_average_misorientation: bool = False,
        show_geometrically_necessary_dislocation_density: bool = False,
        show_channelling_fraction: bool = False,
        show_orientation_cluster: bool = False
    ) -> Iterator[str]:
        yield "Data:"
        columns: list[str] = list()

        if show_scan_coordinates:
            columns += ["X", "Y"]

        if show_phase:
            columns += ["Phase"]

        if show_euler_angles:
            columns += ["Euler1", "Euler2", "Euler3"]

        if show_index_quality:
            columns += ["Index Quality"]

        if show_pattern_quality:
            columns += ["Pattern Quality"]

        if show_inverse_x_pole_figure_coordinates:
            columns += ["X-IPF x-coordinate", "X-IPF y-coordinate"]

        if show_inverse_y_pole_figure_coordinates:
            columns += ["Y-IPF x-coordinate", "Y-IPF y-coordinate"]

        if show_inverse_z_pole_figure_coordinates:
            columns += ["Z-IPF x-coordinate", "Z-IPF y-coordinate"]

        if show_kernel_average_misorientation:
            columns += ["Kernel Average Misorientation"]

        if show_geometrically_necessary_dislocation_density:
            columns += ["GND Density"]

        if show_channelling_fraction:
            columns += ["Channelling Fraction"]

        if show_orientation_cluster:
            columns += ["Point Category", "Point Cluster"]

        yield ",".join(columns)

    def _data_rows(
        self,
        show_scan_coordinates: bool = True,
        show_phase: bool = True,
        show_euler_angles: bool = True,
        show_index_quality: bool = True,
        show_pattern_quality: bool = True,
        show_inverse_x_pole_figure_coordinates: bool = False,
        show_inverse_y_pole_figure_coordinates: bool = False,
        show_inverse_z_pole_figure_coordinates: bool = False,
        show_kernel_average_misorientation: bool = False,
        show_geometrically_necessary_dislocation_density: bool = False,
        show_channelling_fraction: bool = False,
        show_orientation_cluster: bool = False
    ) -> Iterator[str]:
        for y in range(self.height):
            for x in range(self.width):
                columns: list[str] = list()

                if show_scan_coordinates:
                    columns += [x, y]

                if show_phase:
                    columns += [self._phase_id.get_value_at(x, y)]

                if show_euler_angles:
                    columns += [str(angle) for angle in self.euler_angles_degrees.get_value_at(x, y)]

                if show_index_quality:
                    columns += [str(self.index_quality.get_value_at(x, y))]

                if show_pattern_quality:
                    columns += [str(self.pattern_quality.get_value_at(x, y))]

                if show_inverse_x_pole_figure_coordinates:
                    self.euler_angles.get_value_at(x, y)
                    columns += [str(coordinate) for coordinate in (self.inverse_pole_figure_coordinates(Axis.X).get_value_at(x, y))]

                if show_inverse_y_pole_figure_coordinates:
                    columns += [str(coordinate) for coordinate in (self.inverse_pole_figure_coordinates(Axis.Y).get_value_at(x, y))]

                if show_inverse_z_pole_figure_coordinates:
                    columns += [str(coordinate) for coordinate in (self.inverse_pole_figure_coordinates(Axis.Z).get_value_at(x, y))]

                if show_kernel_average_misorientation:
                    columns += [str(self.kernel_average_misorientation_degrees.get_value_at(x, y))]

                if show_geometrically_necessary_dislocation_density:
                    columns += [str(self.geometrically_necessary_dislocation_density_logarithmic().get_value_at(x, y))]

                if show_channelling_fraction:
                    columns += [str(self.channelling_fraction().get_value_at(x, y))]

                if show_orientation_cluster:
                    columns += [
                        str(self.orientation_clustering_category().get_value_at(x, y).code),
                        str(self.orientation_cluster_id().get_value_at(x, y)),
                    ]

                yield ",".join(columns)

    @property
    def phase(self) -> DiscreteFieldMapper[Material]:
        return DiscreteFieldMapper(self.phases, self._phase_id)

    @property
    def euler_angles_degrees(self) -> FunctionalFieldMapper[tuple[float, float, float], tuple[float, float, float]]:
        def tuple_degrees(angles: tuple[float, ...]) -> tuple[float, ...]: return tuple(math.degrees(angle) for angle in angles)
        def tuple_radians(angles: tuple[float, ...]) -> tuple[float, ...]: return tuple(math.radians(angle) for angle in angles)
        return FunctionalFieldMapper(tuple_degrees, self.euler_angles, tuple_radians)

    @euler_angles_degrees.setter
    def euler_angles_degrees(self, value: Field[tuple[float, float, float]]) -> None:
        def tuple_radians(angles: tuple[float, ...]) -> tuple[float, ...]: return tuple(math.radians(angle) for angle in angles)
        euler_angle_values = list()

        for y in range(self.height):
            euler_angle_values.append(list())

            for x in range(self.width):
                euler_angle_values[y].append(tuple_radians(value.get_value_at(x, y)))

        self.euler_angles = Field(self.width, self.height, FieldType.VECTOR_3D, values=euler_angle_values)

    @property
    def reduced_euler_rotation_matrix(self) -> Field[ndarray]:
        if self._reduced_euler_rotation_matrix is None:
            self._init_reduced_euler_rotation_matrices()

        return self._reduced_euler_rotation_matrix

    def inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        if None in (
            self._inverse_x_pole_figure_coordinates,
            self._inverse_y_pole_figure_coordinates,
            self._inverse_z_pole_figure_coordinates,
        ):
            self._init_inverse_pole_figure_coordinates()

        match axis:
            case Axis.X:
                return self._inverse_x_pole_figure_coordinates
            case Axis.Y:
                return self._inverse_y_pole_figure_coordinates
            case Axis.Z:
                return self._inverse_z_pole_figure_coordinates

    @property
    def kernel_average_misorientation(self) -> Field[float]:
        if self._kernel_average_misorientation is None:
            self._init_kernel_average_misorientation()

        return self._kernel_average_misorientation

    @property
    def kernel_average_misorientation_degrees(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(math.degrees, self.kernel_average_misorientation)

    @property
    def pixel_size_micrometres(self) -> float | None:
        if self.pixel_size is None:
            return None
        else:
            return self.pixel_size * 10 ** 6

    @pixel_size_micrometres.setter
    def pixel_size_micrometres(self, value: float) -> None:
        self.pixel_size = value * 10 ** -6

    def misrotation_tensor(self, axis: Axis, pixel_size_micrometres: float = None) -> Field[ndarray]:
        if None in (self._misrotation_x_tensor, self._misrotation_y_tensor) or (pixel_size_micrometres is not None and self.pixel_size_micrometres != pixel_size_micrometres):
            self.pixel_size_micrometres = pixel_size_micrometres if pixel_size_micrometres is not None else self.pixel_size_micrometres

            if self.pixel_size is None:
                raise AttributeError("Misrotation tensor fields not initialised and initialisation arguments were not provided.")

            self._init_misrotation_tensors()

        match axis:
            case Axis.X:
                return self._misrotation_x_tensor
            case Axis.Y:
                return self._misrotation_y_tensor
            case Axis.Z:
                raise ValueError("Misrotation data not available for z-axis intervals.")

    def nye_tensor(self, pixel_size_micrometres: float = None) -> Field[ndarray]:
        if self._nye_tensor is None or (pixel_size_micrometres is not None and self.pixel_size_micrometres != pixel_size_micrometres):
            self.pixel_size_micrometres = pixel_size_micrometres if pixel_size_micrometres is not None else self.pixel_size_micrometres

            if self.pixel_size is None:
                raise AttributeError("Nye tensor field not initialised and initialisation arguments were not provided.")

            self._init_nye_tensor()

        return self._nye_tensor

    def geometrically_necessary_dislocation_density(self, pixel_size_micrometres: float = None) -> Field[float]:
        if self._geometrically_necessary_dislocation_density is None or (pixel_size_micrometres is not None and self.pixel_size_micrometres != pixel_size_micrometres):
            self.pixel_size_micrometres = pixel_size_micrometres if pixel_size_micrometres is not None else self.pixel_size_micrometres

            if self.pixel_size is None:
                raise AttributeError("Geometrically necessary dislocation density field not initialised and initialisation arguments were not provided.")

            self._init_geometrically_necessary_dislocation_density()

        return self._geometrically_necessary_dislocation_density

    def geometrically_necessary_dislocation_density_logarithmic(self, pixel_size: float = None) -> FunctionalFieldMapper[float, float]:
        def log_or_zero(value): 0.0 if value == 0.0 else math.log10(value)
        return FunctionalFieldMapper(log_or_zero, self.geometrically_necessary_dislocation_density(pixel_size))

    @property
    def beam_vector(self) -> ndarray:
        match self.beam_axis:
            case Axis.X:
                raise NotImplementedError()
            case Axis.Y:
                return array((0, -math.sin(self.beam_tilt), math.cos(self.beam_tilt)))
            case Axis.Z:
                raise NotImplementedError()

    @property
    def beam_tilt_degrees(self) -> float | None:
        if self.beam_tilt is None:
            return None
        else:
            return math.degrees(self.beam_tilt)

    @beam_tilt_degrees.setter
    def beam_tilt_degrees(self, value: float) -> None:
        self.beam_tilt = math.radians(value)

    def channelling_fraction(
        self,
        beam_atomic_number: int = None,
        beam_energy: float = None,
        beam_tilt_degrees: float = None,
    ) -> Field[float]:
        if self._channelling_fraction is None or any((
            beam_atomic_number is not None and self.beam_atomic_number != beam_atomic_number,
            beam_energy is not None and self.beam_energy != beam_energy,
            beam_tilt_degrees is not None and self.beam_tilt_degrees != beam_tilt_degrees,
        )):
            self.beam_atomic_number = beam_atomic_number if beam_atomic_number is not None else self.beam_atomic_number
            self.beam_energy = beam_energy if beam_energy is not None else self.beam_energy
            self.beam_tilt_degrees = beam_tilt_degrees if beam_tilt_degrees is not None else self.beam_tilt_degrees

            if None in (self.beam_atomic_number, self.beam_energy, self.beam_tilt):
                raise AttributeError("Channelling fraction field not initialised and initialisation arguments were not provided.")

            self._init_channelling_fraction()

        return self._channelling_fraction

    @property
    def neighbourhood_radius_degrees(self) -> float | None:
        if self.neighbourhood_radius is None:
            return None
        else:
            return math.degrees(self.neighbourhood_radius)

    @neighbourhood_radius_degrees.setter
    def neighbourhood_radius_degrees(self, value: float) -> None:
        self.neighbourhood_radius = math.radians(value)

    def orientation_clustering_category(
        self,
        core_point_neighbourhood_threshold: int = None,
        neighbourhood_radius_degrees: float = None,
    ) -> DiscreteFieldMapper[ClusterCategory]:
        if self._orientation_clustering_category_id is None or any((
            core_point_neighbourhood_threshold is not None and self.core_point_neighbour_threshold != core_point_neighbourhood_threshold,
            neighbourhood_radius_degrees is not None and self.neighbourhood_radius_degrees != neighbourhood_radius_degrees,
        )):
            self.core_point_neighbour_threshold = core_point_neighbourhood_threshold if core_point_neighbourhood_threshold is not None else self.core_point_neighbour_threshold
            self.neighbourhood_radius_degrees = neighbourhood_radius_degrees if neighbourhood_radius_degrees is not None else self.neighbourhood_radius_degrees

            if None in (self.core_point_neighbour_threshold, self.neighbourhood_radius):
                raise AttributeError("Orientation cluster fields not initialised and initialisation arguments were not provided.")

            self._init_orientation_cluster()

        mapping = {category.value: category for category in ClusterCategory}
        return DiscreteFieldMapper(mapping, self._orientation_clustering_category_id)

    def orientation_cluster_id(
        self,
        core_point_neighbourhood_threshold: int = None,
        neighbourhood_radius: float = None,
    ) -> Field[int]:
        if self._orientation_cluster_id is None or any((
            core_point_neighbourhood_threshold is not None and self.core_point_neighbour_threshold != core_point_neighbourhood_threshold,
            neighbourhood_radius is not None and self.neighbourhood_radius != neighbourhood_radius,
        )):
            self.core_point_neighbour_threshold = core_point_neighbourhood_threshold if core_point_neighbourhood_threshold is not None else self.core_point_neighbour_threshold
            self.neighbourhood_radius = neighbourhood_radius if neighbourhood_radius is not None else self.neighbourhood_radius

            if None in (self.core_point_neighbour_threshold, self.neighbourhood_radius):
                raise AttributeError(
                    "Orientation cluster fields not initialised and initialisation arguments were not provided.")

            self._init_orientation_cluster()

        return self._orientation_cluster_id

    def _init_reduced_euler_rotation_matrices(self) -> None:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=eye(3))

        for y in range(self.height):
            for x in range(self.width):
                axis_set = self.axis_set
                euler_angles = self.euler_angles.get_value_at(x, y)
                crystal_family = self.phase.get_value_at(x, y).lattice_type.get_family()
                value = reduce_matrix(euler_rotation_matrix(axis_set, euler_angles), crystal_family)
                field.set_value_at(x, y, value)

        self._reduced_euler_rotation_matrix = field

    def _gen_inverse_pole_figure_coordinates(self, axis: Axis) -> Field[tuple[float, float]]:
        field = Field(self.width, self.height, FieldType.VECTOR_2D, default_value=(0.0, 0.0))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    value = forward_stereographic(*dot(rotation_matrix, axis.value).tolist())
                    field.set_value_at(x, y, value)

        return field

    def _init_inverse_pole_figure_coordinates(self) -> None:
        self._inverse_x_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.X)
        self._inverse_y_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Y)
        self._inverse_z_pole_figure_coordinates = self._gen_inverse_pole_figure_coordinates(Axis.Z)

    def _init_kernel_average_misorientation(self) -> None:
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

    def _gen_misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=zeros((3, 3)))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    total = zeros((3, 3))
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

    def _init_misrotation_tensors(self) -> None:
        self._misrotation_x_tensor = self._gen_misrotation_tensor(Axis.X)
        self._misrotation_y_tensor = self._gen_misrotation_tensor(Axis.Y)

    def _init_nye_tensor(self) -> None:
        field = Field(self.width, self.height, FieldType.MATRIX, default_value=zeros((3, 3)))

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    value = array((
                        (0.0, self.misrotation_tensor(Axis.X).get_value_at(x, y)[2][0], -self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][0]),
                        (-self.misrotation_tensor(Axis.Y).get_value_at(x, y)[2][1], 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][1]),
                        (0.0, 0.0, self.misrotation_tensor(Axis.Y).get_value_at(x, y)[0][2] - self.misrotation_tensor(Axis.X).get_value_at(x, y)[1][2])
                    ))

                    field.set_value_at(x, y, value)

        self._nye_tensor = field

    def _init_geometrically_necessary_dislocation_density(self) -> None:
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    nye_tensor_norm = sum(abs(element) for element in self.nye_tensor().get_value_at(x, y))
                    close_pack_distance = self.phase.get_value_at(x, y).close_pack_distance
                    value = (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm
                    field.set_value_at(x, y, value)

        self._geometrically_necessary_dislocation_density = field

    def _init_channelling_fraction(self) -> None:
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        channel_data = {
            local_id: load_crit_data(self.beam_atomic_number, phase.global_id, self.beam_energy)
            for local_id, phase in self.phases.items() if phase.global_id != UNINDEXED_PHASE_ID
        }

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    effective_beam_vector = dot(rotation_matrix, self.beam_vector).to_list()
                    value = fraction(effective_beam_vector, channel_data[self._phase_id.get_value_at(x, y)])
                    field.set_value_at(x, y, value)

        self._channelling_fraction = field

    def _init_orientation_cluster(self) -> None:
        phase = zeros((self.height, self.width))
        reduced_euler_rotation_matrix = zeros((self.height, self.width, 3, 3))

        for y in range(self.height):
            for x in range(self.width):
                phase[y][x] = self.phase.get_value_at(x, y).global_id
                reduced_euler_rotation_matrix[y][x] = self.reduced_euler_rotation_matrix.get_value_at(x, y)

        cluster_count, category_id_array, cluster_id_array = dbscan(
            self.width,
            self.height,
            phase,
            reduced_euler_rotation_matrix,
            self.core_point_neighbour_threshold,
            self.neighbourhood_radius
        )

        self.cluster_count = cluster_count
        self._orientation_clustering_category_id = Field(self.width, self.height, FieldType.DISCRETE, values=category_id_array.tolist())
        self._orientation_cluster_id = Field(self.width, self.height, FieldType.DISCRETE, values=cluster_id_array.tolist())
