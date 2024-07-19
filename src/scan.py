# -*- coding: utf-8 -*-

from collections.abc import Iterator
from typing import Self
from src.field import FieldNullError
from src.field_manager import FieldManager
from src.geometry import Axis, AxisSet
from src.map_manager import MapManager
from src.parameter_groups import ScanParameters, ScaleParameters, ChannellingParameters, ClusteringParameters
from src.phase import Phase, UNINDEXED_PHASE_ID


class Scan:
    def __init__(
        self,
        file_reference: str,
        width: int,
        height: int,
        phases: dict[int, Phase],
        phase_id_values: list[list[int]],
        euler_angle_degrees_values: list[list[tuple[float, float, float] | None]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
        axis_set: AxisSet = AxisSet.ZXZ,
    ):
        self._scan_parameters = ScanParameters()
        self.scale_parameters = ScaleParameters()
        self.channelling_parameters = ChannellingParameters()
        self.clustering_parameters = ClusteringParameters()
        self._scan_parameters.set(file_reference, width, height, phases, axis_set)

        self.field = FieldManager(
            self._scan_parameters,
            self.scale_parameters,
            self.channelling_parameters,
            self.clustering_parameters,
            phase_id_values,
            euler_angle_degrees_values,
            pattern_quality_values,
            index_quality_values,
        )

        self.map = MapManager(self.field)

    @property
    def file_reference(self) -> str:
        return self._scan_parameters.file_reference

    @property
    def width(self) -> int:
        return self._scan_parameters.width

    @property
    def height(self) -> int:
        return self._scan_parameters.height

    @property
    def phases(self) -> dict[int, Phase]:
        return self._scan_parameters.phases

    @property
    def axis_set(self) -> AxisSet:
        return self._scan_parameters.axis_set

    @property
    def cluster_count(self) -> int:
        return self.field._cluster_count

    @classmethod
    def from_pathfinder_file(cls, data_path: str, file_reference: str = None) -> Self:
        if file_reference is None:
            file_reference = data_path.split("/")[-1].split(".")[0].lstrip("p")

        with open(data_path, "r", encoding="utf-8") as file:
            materials = dict()
            file_materials = Phase.load_from_materials_file()
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
            phase_id_values: list[list[int | None]] = list()
            euler_angle_degrees_values: list[list[tuple[float, float, float] | None]] = list()
            index_quality_values: list[list[float]] = list()
            pattern_quality_values: list[list[float]] = list()
            file.readline()
            file.readline()

            for y in range(height):
                phase_id_values.append(list())
                euler_angle_degrees_values.append(list())
                index_quality_values.append(list())
                pattern_quality_values.append(list())

                for x in range(width):
                    line = file.readline().rstrip("\n").split(",")
                    local_phase_id = int(line[2])

                    if materials[local_phase_id].global_id == UNINDEXED_PHASE_ID:
                        phase_id_values[y].append(None)
                        euler_angle_degrees_values[y].append(None)
                    else:
                        phase_id_values[y].append(local_phase_id)
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
            yield "Phases:"

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
                    columns += self.field._phase_id.serialize_value_at(x, y, null_serialization=str(UNINDEXED_PHASE_ID))

                if show_euler_angles:
                    columns += self.field.euler_angles_degrees.serialize_value_at(x, y)

                if show_index_quality:
                    columns += self.field.index_quality.serialize_value_at(x, y)

                if show_pattern_quality:
                    columns += self.field.pattern_quality.serialize_value_at(x, y)

                if show_inverse_x_pole_figure_coordinates:
                    columns += self.field.inverse_pole_figure_coordinates(Axis.X).serialize_value_at(x, y)

                if show_inverse_y_pole_figure_coordinates:
                    columns += self.field.inverse_pole_figure_coordinates(Axis.Y).serialize_value_at(x, y)

                if show_inverse_z_pole_figure_coordinates:
                    columns += self.field.inverse_pole_figure_coordinates(Axis.Z).serialize_value_at(x, y)

                if show_kernel_average_misorientation:
                    columns += self.field.kernel_average_misorientation_degrees.serialize_value_at(x, y)

                if show_geometrically_necessary_dislocation_density:
                    columns += self.field.geometrically_necessary_dislocation_density_logarithmic.serialize_value_at(x, y)

                if show_channelling_fraction:
                    columns += self.field.channelling_fraction.serialize_value_at(x, y)

                if show_orientation_cluster:
                    try:
                        columns += [self.field.orientation_clustering_category.get_value_at(x, y).code]
                    except FieldNullError:
                        columns += [""]

                    columns += self.field.orientation_cluster_id.serialize_value_at(x, y)

                yield ",".join(columns)
