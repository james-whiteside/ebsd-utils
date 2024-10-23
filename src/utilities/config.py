# -*- coding: utf-8 -*-

from os import getcwd
from configparser import ConfigParser
from src.utilities.geometry import AxisSet
from src.data_structures.parameter_groups import (
    ProjectParams,
    DataParams,
    AnalysisParams,
    MapParams,
    ResolutionParams,
    DislocationParams,
    ChannellingParams,
    ClusteringParams,
)


class Config:
    def __init__(self, path: str = "config.ini"):
        config_path = f"{getcwd()}/{path}"
        parser = ConfigParser()
        parser.read(config_path)

        self.project = ProjectParams(
            data_dir=self._str(parser["project"]["ebsd_data_dir"]),
            materials_file=self._str(parser["project"]["materials_file"]),
            channelling_cache_dir=self._str(parser["project"]["channelling_cache_dir"]),
            analysis_dir=self._str(parser["project"]["analysis_output_dir"]),
            map_dir=self._str(parser["project"]["map_output_dir"]),
        )

        self.data = DataParams(
            euler_axis_set=self._axis_set(parser["data"]["euler_axis_set"]),
            pixel_size_microns=self._float(parser["data"]["pixel_size"]),
        )

        self.analysis = AnalysisParams(
            reduce_resolution=self._bool(parser["analysis"]["reduce_resolution"]),
            compute_dislocation=self._bool(parser["analysis"]["compute_dislocation_densities"]),
            compute_channelling=self._bool(parser["analysis"]["compute_channelling_fractions"]),
            compute_clustering=self._bool(parser["analysis"]["compute_orientation_clusters"]),
            use_cuda=self._bool(parser["analysis"]["use_cuda"]),
        )

        self.maps = MapParams(
            upscale_factor=self._int(parser["maps"]["upscale_factor"]),
        )

        self.resolution = ResolutionParams(
            reduction_factor=self._int(parser["resolution_reduction"]["reduction_factor"]),
            scaling_tolerance=self._float(parser["resolution_reduction"]["scaling_tolerance"]),
        )

        self.dislocation = DislocationParams(
            corrective_factor=self._float(parser["dislocation_density"]["corrective_factor"]),
        )

        self.channelling = ChannellingParams(
            beam_atomic_number=self._int(parser["channelling_fraction"]["beam_atomic_number"]),
            beam_energy=self._float(parser["channelling_fraction"]["beam_energy"]),
            beam_tilt_deg=self._float(parser["channelling_fraction"]["beam_tilt"]),
        )

        self.clustering = ClusteringParams(
            core_point_threshold=self._int(parser["orientation_clustering"]["neighbour_threshold"]),
            neighbourhood_radius_deg=self._float(parser["orientation_clustering"]["neighbourhood_radius"]),
        )

    @staticmethod
    def _str(value: str) -> str:
        return value.strip()

    @staticmethod
    def _int(value: str) -> int:
        return int(Config._str(value))

    @staticmethod
    def _float(value: str) -> float:
        return float(Config._str(value))

    @staticmethod
    def _bool(value: str) -> bool:
        if Config._str(value) not in ("true", "false"):
            raise ValueError(f"Boolean config value must be 'true' or 'false', not: '{Config._str(value)}'")

        return Config._str(value) == "true"

    @staticmethod
    def _str_list(value: str) -> list[str]:
        return [Config._str(item) for item in value.strip().lstrip("[").rstrip("]").split(",")]

    @staticmethod
    def _int_list(value: str) -> list[int]:
        return [Config._int(item) for item in Config._str_list(value)]

    @staticmethod
    def _axis_set(value: str) -> AxisSet:
        return AxisSet[Config._str(value).upper()]
