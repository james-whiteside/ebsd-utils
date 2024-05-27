# -*- coding: utf-8 -*-

from collections.abc import Iterator
from typing import Self
from numpy import ndarray, array, zeros, eye, dot
from field import FieldType, Field, DiscreteFieldMapper, FunctionalFieldMapper
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
from material import Material, UNINDEXED_PHASE_ID
from channelling import load_crit_data, fraction
from clustering import ClusterCategory, dbscan
from parameter_groups import ScaleParameters, ChannellingParameters, ClusteringParameters
from utilities import tuple_degrees, tuple_radians, float_degrees, float_radians, log_or_zero


GND_DENSITY_CORRECTIVE_FACTOR = 3.6


class Scan:
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
        self.scale_parameters = ScaleParameters()
        self._misrotation_x_tensor = None
        self._misrotation_y_tensor = None
        self._nye_tensor = None
        self._geometrically_necessary_dislocation_density = None
        self.channelling_parameters = ChannellingParameters()
        self._channelling_fraction = None
        self.clustering_parameters = ClusteringParameters()
        self._cluster_count = None
        self._orientation_clustering_category_id = None
        self._orientation_cluster_id = None

    @property
    def phase(self) -> DiscreteFieldMapper[Material]:
        return DiscreteFieldMapper(self.phases, self._phase_id)

    @property
    def euler_angles_degrees(self) -> FunctionalFieldMapper[tuple[float, float, float], tuple[float, float, float]]:
        return FunctionalFieldMapper(tuple_degrees, self.euler_angles, tuple_radians)

    @euler_angles_degrees.setter
    def euler_angles_degrees(self, value: Field[tuple[float, float, float]]) -> None:
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
        return FunctionalFieldMapper(float_degrees, self.kernel_average_misorientation, float_radians)

    def misrotation_tensor(self, axis: Axis) -> Field[ndarray]:
        if None in (self._misrotation_x_tensor, self._misrotation_y_tensor):
            self._init_misrotation_tensors()

        match axis:
            case Axis.X:
                return self._misrotation_x_tensor
            case Axis.Y:
                return self._misrotation_y_tensor
            case Axis.Z:
                raise ValueError("Misrotation data not available for z-axis intervals.")

    @property
    def nye_tensor(self) -> Field[ndarray]:
        if self._nye_tensor is None:
            self._init_nye_tensor()

        return self._nye_tensor

    @property
    def geometrically_necessary_dislocation_density(self) -> Field[float]:
        if self._geometrically_necessary_dislocation_density is None:
            self._init_geometrically_necessary_dislocation_density()

        return self._geometrically_necessary_dislocation_density

    @property
    def geometrically_necessary_dislocation_density_logarithmic(self) -> FunctionalFieldMapper[float, float]:
        return FunctionalFieldMapper(log_or_zero, self.geometrically_necessary_dislocation_density)

    @property
    def channelling_fraction(self) -> Field[float]:
        if self._channelling_fraction is None:
            self._init_channelling_fraction()

        return self._channelling_fraction

    @property
    def cluster_count(self) -> int:
        if self._cluster_count is None:
            self._init_orientation_cluster()

        return self._cluster_count

    @property
    def orientation_clustering_category(self) -> DiscreteFieldMapper[ClusterCategory]:
        if self._orientation_clustering_category_id is None:
            self._init_orientation_cluster()

        mapping = {category.value: category for category in ClusterCategory}
        return DiscreteFieldMapper(mapping, self._orientation_clustering_category_id)

    @property
    def orientation_cluster_id(self) -> Field[int]:
        if self._orientation_cluster_id is None:
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
                    value = forward_stereographic(*dot(rotation_matrix, array(axis.value)).tolist())
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
                                total += misrotation_tensor(misrotation_matrix(rotation_matrix_1, rotation_matrix_2), self.scale_parameters.pixel_size)
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
                    nye_tensor_norm = sum(abs(element) for row in self.nye_tensor.get_value_at(x, y).tolist() for element in row)
                    close_pack_distance = self.phase.get_value_at(x, y).close_pack_distance
                    value = (GND_DENSITY_CORRECTIVE_FACTOR / close_pack_distance) * nye_tensor_norm
                    field.set_value_at(x, y, value)

        self._geometrically_necessary_dislocation_density = field

    def _init_channelling_fraction(self) -> None:
        field = Field(self.width, self.height, FieldType.SCALAR, default_value=0.0)

        channel_data = {
            local_id: load_crit_data(self.channelling_parameters.beam_atomic_number, phase.global_id, self.channelling_parameters.beam_energy)
            for local_id, phase in self.phases.items() if phase.global_id != UNINDEXED_PHASE_ID
        }

        for y in range(self.height):
            for x in range(self.width):
                if self.phase.get_value_at(x, y).global_id == UNINDEXED_PHASE_ID:
                    continue
                else:
                    rotation_matrix = self.reduced_euler_rotation_matrix.get_value_at(x, y)
                    effective_beam_vector = dot(rotation_matrix, self.channelling_parameters.beam_vector).tolist()
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
            self.clustering_parameters.core_point_neighbour_threshold,
            self.clustering_parameters.neighbourhood_radius
        )

        self._cluster_count = cluster_count
        self._orientation_clustering_category_id = Field(self.width, self.height, FieldType.DISCRETE, values=category_id_array.astype(int).tolist())
        self._orientation_cluster_id = Field(self.width, self.height, FieldType.DISCRETE, values=cluster_id_array.astype(int).tolist())

    @classmethod
    def from_pathfinder_file(cls, data_path: str, materials_path: str, file_reference: str = None) -> Self:
        if file_reference is None:
            file_reference = data_path.split("/")[-1].split(".")[0].lstrip("p")

        with open(data_path, "r", encoding="utf-8") as file:
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

        with open(path, "w", encoding="utf-8") as file:
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
            yield f"Pixel size (μm),{self.scale_parameters.pixel_size_micrometres}"

        if show_channelling_params:
            yield f"Channelling:"
            yield f"Atomic number,{self.channelling_parameters.beam_atomic_number}"
            yield f"Energy (eV),{self.channelling_parameters.beam_energy}"
            yield f"Tilt (deg),{self.channelling_parameters.beam_tilt_degrees}"

        if show_clustering_params:
            yield f"Clustering:"
            yield f"Core point threshold,{self.clustering_parameters.core_point_neighbour_threshold}"
            yield f"Point neighbourhood radius (deg),{self.clustering_parameters.neighbourhood_radius_degrees}"
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
                    columns += [str(x), str(y)]

                if show_phase:
                    columns += [str(self._phase_id.get_value_at(x, y))]

                if show_euler_angles:
                    columns += [str(angle) for angle in self.euler_angles_degrees.get_value_at(x, y)]

                if show_index_quality:
                    columns += [str(self.index_quality.get_value_at(x, y))]

                if show_pattern_quality:
                    columns += [str(self.pattern_quality.get_value_at(x, y))]

                if show_inverse_x_pole_figure_coordinates:
                    self.euler_angles.get_value_at(x, y)
                    columns += [str(coordinate) for coordinate in
                                (self.inverse_pole_figure_coordinates(Axis.X).get_value_at(x, y))]

                if show_inverse_y_pole_figure_coordinates:
                    columns += [str(coordinate) for coordinate in
                                (self.inverse_pole_figure_coordinates(Axis.Y).get_value_at(x, y))]

                if show_inverse_z_pole_figure_coordinates:
                    columns += [str(coordinate) for coordinate in
                                (self.inverse_pole_figure_coordinates(Axis.Z).get_value_at(x, y))]

                if show_kernel_average_misorientation:
                    columns += [str(self.kernel_average_misorientation_degrees.get_value_at(x, y))]

                if show_geometrically_necessary_dislocation_density:
                    columns += [str(self.geometrically_necessary_dislocation_density_logarithmic.get_value_at(x, y))]

                if show_channelling_fraction:
                    columns += [str(self.channelling_fraction.get_value_at(x, y))]

                if show_orientation_cluster:
                    columns += [
                        str(self.orientation_clustering_category.get_value_at(x, y).code),
                        str(self.orientation_cluster_id.get_value_at(x, y)),
                    ]

                yield ",".join(columns)
