# -*- coding: utf-8 -*-

from os import getcwd
from configparser import ConfigParser
from math import degrees, sin, cos
from src.utilities.geometry import Axis, AxisSet


class Config:
    def __init__(self, path: str = "config.ini"):
        config_path = f"{getcwd()}/{path}"
        parser = ConfigParser()
        parser.read(config_path)
        self.data_dir = self._str(parser["project"]["ebsd_data_dir"])
        self.materials_file = self._str(parser["project"]["materials_file"])
        self.channelling_cache_dir = self._str(parser["project"]["channelling_cache_dir"])
        self.analysis_dir = self._str(parser["project"]["analysis_output_dir"])
        self.map_dir = self._str(parser["project"]["map_output_dir"])
        self.axis_set = self._axis_set(parser["data"]["euler_axis_set"])
        self.pixel_size_microns = self._float(parser["data"]["pixel_size"])
        self.reduce_resolution = self._bool(parser["analysis"]["reduce_resolution"])
        self.compute_dislocation = self._bool(parser["analysis"]["compute_dislocation_densities"])
        self.compute_channelling = self._bool(parser["analysis"]["compute_channelling_fractions"])
        self.compute_clustering = self._bool(parser["analysis"]["compute_orientation_clusters"])
        self.use_cuda = self._bool(parser["analysis"]["use_cuda"])
        self.upscale_factor = self._int(parser["maps"]["upscale_factor"])
        self.reduction_factor = self._int(parser["resolution_reduction"]["reduction_factor"])
        self.scaling_tolerance = self._float(parser["resolution_reduction"]["scaling_tolerance"])
        self.gnd_corrective_factor = self._float(parser["dislocation_density"]["corrective_factor"])
        self.beam_atomic_number = self._int(parser["channelling_fraction"]["beam_atomic_number"])
        self.beam_energy = self._float(parser["channelling_fraction"]["beam_energy"])
        self.beam_tilt_rad = self._float(parser["channelling_fraction"]["beam_tilt"])
        self.core_point_threshold = self._int(parser["orientation_clustering"]["neighbour_threshold"])
        self.neighbourhood_radius_rad = self._float(parser["orientation_clustering"]["neighbourhood_radius"])

    @property
    def pixel_size(self) -> float:
        return self.pixel_size_microns * 10.0 ** -6.0

    @property
    def beam_tilt_deg(self) -> float:
        return degrees(self.beam_tilt_rad)

    @property
    def beam_axis(self) -> Axis:
        beam_vector = 0.0, -sin(self.beam_tilt_rad), cos(self.beam_tilt_rad)
        return Axis.beam(beam_vector)

    @property
    def beam_vector(self) -> tuple[float, float, float]:
        return self.beam_axis.vector

    @property
    def neighbourhood_radius_deg(self) -> float:
        return degrees(self.neighbourhood_radius_rad)

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
