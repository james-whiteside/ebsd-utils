# -*- coding: utf-8 -*-

from random import Random
from typing import Self
from numpy import zeros
from src.data_structures.aggregate_manager import AggregateManager
from src.data_structures.field import FieldNullError
from src.data_structures.field_manager import FieldManager
from src.utilities.config import Config
from src.utilities.filestore import load_data, dump_analysis, dump_maps
from src.utilities.geometry import orthogonalise_matrix, euler_angles
from src.data_structures.map_manager import MapManager
from src.data_structures.parameter_groups import ScanParams
from src.data_structures.phase import Phase
from src.utilities.utils import tuple_degrees


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
            pixel_size = config.data.pixel_size

        self.params = ScanParams(data_ref, width, height, phases, pixel_size, reduction_factor)
        self.config = config
        self._random_source = Random(config.analysis.random_seed)

        self.field = FieldManager(
            self.params,
            phase_id_values,
            euler_angle_values,
            pattern_quality_values,
            index_quality_values,
            self.config,
            self._random_source,
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
                        orientation_matrix_aggregate = orthogonalise_matrix(orientation_matrix_total / count, self.config.resolution.scaling_tolerance)
                        euler_angle_aggregate = tuple_degrees(euler_angles(orientation_matrix_aggregate, self.config.data.euler_axis_set))
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
    def from_csv(cls, data_path: str, config: Config, data_ref: str = None) -> Self:
        return load_data(data_path, config, data_ref)

    def to_csv(self, dir: str):
        dump_analysis(self, dir)

    def to_maps(self, dir: str):
        dump_maps(self, dir)
