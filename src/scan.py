# -*- coding: utf-8 -*-

from collections.abc import Iterator
from os import makedirs
from typing import Self
from numpy import zeros
from src.data_structures.aggregate_manager import AggregateManager
from src.data_structures.field import FieldNullError
from src.data_structures.field_manager import FieldManager
from src.data_structures.map import Map
from src.utilities.config import Config
from src.utilities.geometry import orthogonalise_matrix, euler_angles, Axis
from src.data_structures.map_manager import MapManager
from src.data_structures.parameter_groups import ScanParams
from src.data_structures.phase import Phase
from src.utilities.utilities import tuple_degrees


class Scan:
    def __init__(
        self,
        data_ref: str,
        width: int,
        height: int,
        phases: dict[int, Phase],
        phase_id_values: list[list[int]],
        euler_angle_values: list[list[tuple[float, float, float] | None]],
        pattern_quality_values: list[list[float]],
        index_quality_values: list[list[float]],
        config: Config,
        reduction_factor: int = 0,
        pixel_size: float = None,
    ):
        if pixel_size is None:
            pixel_size = config.pixel_size

        self.params = ScanParams(data_ref, width, height, phases, pixel_size, reduction_factor)
        self.config = config

        self.field = FieldManager(
            self.params,
            phase_id_values,
            euler_angle_values,
            pattern_quality_values,
            index_quality_values,
            self.config,
        )

        self._map = None
        self._cluster_aggregate = None

    @property
    def map(self) -> MapManager:
        if self._map is None:
            self._map = MapManager(self.field)

        return self._map

    @property
    def cluster_aggregate(self) -> AggregateManager:
        if self._cluster_aggregate is None:
            self._cluster_aggregate = AggregateManager(self.field, self.field.orientation_cluster_id)

        return self._cluster_aggregate

    @property
    def cluster_count(self) -> int:
        return self.field._cluster_count

    def _reduce_resolution(self) -> Self:
        if self.params.width % 2 != 0 or self.params.height % 2 != 0:
            raise ArithmeticError("Can only reduce resolution of scan with even width and height.")

        data_ref = self.params.data_ref
        width = self.params.width // 2
        height = self.params.height // 2
        phases = self.params.phases
        reduction_factor = self.params.reduction_factor + 1
        pixel_size = self.params.pixel_size * 2
        config = self.config

        phase_id_values: list[list[int | None]] = list()
        euler_angle_values: list[list[tuple[float, float, float] | None]] = list()
        index_quality_values: list[list[float]] = list()
        pattern_quality_values: list[list[float]] = list()

        for y in range(height):
            phase_id_values.append(list())
            euler_angle_values.append(list())
            index_quality_values.append(list())
            pattern_quality_values.append(list())

            for x in range(width):
                kernel = [(0, 0), (1, 0), (0, 1), (1, 1)]

                kernel_phases = set()

                for dx, dy in kernel:
                    try:
                        phase = self.field._phase_id.get_value_at(2 * x + dx, 2 * y + dy)
                    except FieldNullError:
                        continue

                    kernel_phases.add(phase)

                if len(kernel_phases) != 1:
                    phase_id = None
                    euler_angle_aggregate = None
                    index_quality_aggregate = 0.0
                    pattern_quality_aggregate = 0.0
                else:
                    phase_id = kernel_phases.pop()
                    count = 0
                    orientation_matrix_total = zeros((3, 3))
                    index_quality_total = 0.0
                    pattern_quality_total = 0.0

                    for dx, dy in kernel:
                        try:
                            self.field._phase_id.get_value_at(2 * x + dx, 2 * y + dy)
                            orientation_matrix = self.field.orientation_matrix.get_value_at(2 * x + dx, 2 * y + dy)
                            index_quality = self.field.index_quality.get_value_at(2 * x + dx, 2 * y + dy)
                            pattern_quality = self.field.pattern_quality.get_value_at(2 * x + dx, 2 * y + dy)
                        except FieldNullError:
                            continue

                        orientation_matrix_total += orientation_matrix
                        index_quality_total += index_quality
                        pattern_quality_total += pattern_quality
                        count += 1

                    try:
                        orientation_matrix_aggregate = orthogonalise_matrix(orientation_matrix_total / count, self.config.scaling_tolerance)
                        euler_angle_aggregate = tuple_degrees(euler_angles(orientation_matrix_aggregate, self.config.axis_set))
                        index_quality_aggregate = index_quality_total / count
                        pattern_quality_aggregate = pattern_quality_total / len(kernel)
                    except ArithmeticError:
                        phase_id = None
                        euler_angle_aggregate = None
                        index_quality_aggregate = 0.0
                        pattern_quality_aggregate = 0.0

                phase_id_values[y].append(phase_id)
                euler_angle_values[y].append(euler_angle_aggregate)
                index_quality_values[y].append(index_quality_aggregate)
                pattern_quality_values[y].append(pattern_quality_aggregate)

        scan = Scan(
            data_ref=data_ref,
            width=width,
            height=height,
            phases=phases,
            phase_id_values=phase_id_values,
            euler_angle_values=euler_angle_values,
            pattern_quality_values=pattern_quality_values,
            index_quality_values=index_quality_values,
            config=config,
            reduction_factor=reduction_factor,
            pixel_size=pixel_size,
        )

        return scan

    def reduce_resolution(self, reduction_factor: int) -> Self:
        if reduction_factor <= 0:
            return self
        else:
            return self._reduce_resolution().reduce_resolution(reduction_factor - 1)

    @classmethod
    def from_pathfinder_file(cls, data_path: str, config: Config, data_ref: str = None) -> Self:
        if data_ref is None:
            data_ref = data_path.split("/")[-1].split(".")[0].lstrip("p")

        with open(data_path, "r", encoding="utf-8") as file:
            materials = dict()
            file_materials = Phase.from_materials_file(config.materials_file)
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
            euler_angle_values: list[list[tuple[float, float, float] | None]] = list()
            index_quality_values: list[list[float]] = list()
            pattern_quality_values: list[list[float]] = list()
            file.readline()
            file.readline()

            for y in range(height):
                phase_id_values.append(list())
                euler_angle_values.append(list())
                index_quality_values.append(list())
                pattern_quality_values.append(list())

                for x in range(width):
                    line = file.readline().rstrip("\n").split(",")
                    local_phase_id = int(line[2])

                    if materials[local_phase_id].global_id == Phase.UNINDEXED_ID:
                        phase_id_values[y].append(None)
                        euler_angle_values[y].append(None)
                    else:
                        phase_id_values[y].append(local_phase_id)
                        euler_angle_values[y].append((float(line[3]), float(line[4]), float(line[5])))

                    index_quality_values[y].append(float(line[6]))
                    pattern_quality_values[y].append(float(line[7]))

        return Scan(
            data_ref=data_ref,
            width=width,
            height=height,
            phases=materials,
            phase_id_values=phase_id_values,
            euler_angle_values=euler_angle_values,
            pattern_quality_values=pattern_quality_values,
            index_quality_values=index_quality_values,
            config=config,
        )

    def to_pathfinder_file(self, path: str):
        with open(path, "w", encoding="utf-8") as file:
            for row in self._rows():
                file.write(f"{row}\n")

    def to_maps(self, dir: str):
        makedirs(dir, exist_ok=True)

        for map in self._maps():
            path = f"{dir}/{map.map_type.name}.png"
            map.image.save(path)

    def _rows(self) -> Iterator[str]:
        for row in self._metadata_rows():
            yield row

        if self.config.compute_clustering:
            for row in self._cluster_aggregate_rows():
                yield row

        for row in self._data_rows():
            yield row

    def _metadata_rows(self) -> str:
        yield "Phases:"

        for local_id, phase in self.params.phases.items():
            yield f"{local_id},{phase.name},{phase.global_id}"

        yield f"Map size:"
        yield f"X,{self.params.width}"
        yield f"Y,{self.params.height}"
        yield f"Map scale:"
        yield f"Pixel size (μm),{self.params.pixel_size_microns}"

        if self.config.compute_channelling:
            yield f"Channelling:"
            yield f"Atomic number,{self.config.beam_atomic_number}"
            yield f"Energy (eV),{self.config.beam_energy}"
            yield f"Tilt (deg),{self.config.beam_tilt_deg}"

        if self.config.compute_clustering:
            yield f"Clustering:"
            yield f"Core point threshold,{self.config.core_point_threshold}"
            yield f"Point neighbourhood radius (deg),{self.config.neighbourhood_radius_deg}"
            yield f"Cluster count,{self.cluster_count}"

    def _cluster_aggregate_rows(self) -> Iterator[str]:
        yield "Cluster aggregates:"
        columns: list[str] = list()
        columns += ["Cluster ID"]
        columns += ["Cluster Size"]
        columns += ["Phase"]
        columns += ["Euler1", "Euler2", "Euler3"]
        columns += ["Index Quality"]
        columns += ["Pattern Quality"]

        if self.config.compute_channelling:
            columns += ["Beam-IPF x-coordinate", "Beam-IPF y-coordinate"]

        columns += ["Kernel Average Misorientation"]

        if self.config.compute_dislocation:
            columns += ["GND Density"]

        if self.config.compute_channelling:
            columns += ["Channelling Fraction"]

        yield ",".join(columns)

        for id in self.cluster_aggregate.group_ids:
            columns: list[str] = list()
            columns += [str(id)]
            columns += self.cluster_aggregate.count.serialize_value_for(id)
            columns += self.cluster_aggregate._phase_id.serialize_value_for(id)
            columns += self.cluster_aggregate.euler_angles_deg.serialize_value_for(id)
            columns += self.cluster_aggregate.index_quality.serialize_value_for(id)
            columns += self.cluster_aggregate.pattern_quality.serialize_value_for(id)

            if self.config.compute_channelling:
                beam_axis = self.config.beam_axis
                columns += self.cluster_aggregate.ipf_coordinates(beam_axis).serialize_value_for(id)

            columns += self.cluster_aggregate.average_misorientation_deg.serialize_value_for(id)

            if self.config.compute_dislocation:
                columns += self.cluster_aggregate.gnd_density_log.serialize_value_for(id)

            if self.config.compute_channelling:
                columns += self.cluster_aggregate.channelling_fraction.serialize_value_for(id)

            yield ",".join(columns)

    def _data_rows(self) -> Iterator[str]:
        yield "Data:"
        columns: list[str] = list()
        columns += ["X", "Y"]
        columns += ["Phase"]
        columns += ["Euler1", "Euler2", "Euler3"]
        columns += ["Index Quality"]
        columns += ["Pattern Quality"]

        if self.config.compute_channelling:
            columns += ["Beam-IPF x-coordinate", "Beam-IPF y-coordinate"]

        columns += ["Kernel Average Misorientation"]

        if self.config.compute_dislocation:
            columns += ["GND Density"]

        if self.config.compute_channelling:
            columns += ["Channelling Fraction"]

        if self.config.compute_clustering:
            columns += ["Point Category", "Point Cluster"]

        yield ",".join(columns)

        for y in range(self.params.height):
            for x in range(self.params.width):
                columns = list()
                columns += [str(x), str(y)]
                columns += self.field._phase_id.serialize_value_at(x, y, null_serialization=str(Phase.UNINDEXED_ID))
                columns += self.field.euler_angles_deg.serialize_value_at(x, y)
                columns += self.field.index_quality.serialize_value_at(x, y)
                columns += self.field.pattern_quality.serialize_value_at(x, y)

                if self.config.compute_channelling:
                    beam_axis = self.config.beam_axis
                    columns += self.field.ipf_coordinates(beam_axis).serialize_value_at(x, y)

                columns += self.field.average_misorientation_deg.serialize_value_at(x, y)

                if self.config.compute_dislocation:
                    columns += self.field.gnd_density_log.serialize_value_at(x, y)

                if self.config.compute_channelling:
                    columns += self.field.channelling_fraction.serialize_value_at(x, y)

                if self.config.compute_clustering:
                    try:
                        columns += [self.field.clustering_category.get_value_at(x, y).code]
                    except FieldNullError:
                        columns += [""]

                    columns += self.field.orientation_cluster_id.serialize_value_at(x, y)

                yield ",".join(columns)

    def _maps(self) -> Iterator[Map]:
        yield self.map.phase
        yield self.map.euler_angle
        yield self.map.pattern_quality
        yield self.map.index_quality
        yield self.map.orientation(Axis.X)
        yield self.map.orientation(Axis.Y)
        yield self.map.orientation(Axis.Z)
        yield self.map.average_misorientation

        if self.config.compute_dislocation:
            yield self.map.gnd_density

        if self.config.compute_channelling:
            yield self.map.orientation(self.config.beam_axis)
            yield self.map.channelling_fraction

        if self.config.compute_clustering:
            yield self.map.orientation_cluster
