# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser


class Config:
    def __init__(self, path: str = "config.ini"):
        config_path = f"{os.getcwd()}/{path}"
        parser = ConfigParser()
        parser.read(config_path)
        self.data_dir = self._str(parser["project"]["ebsd_data_dir"])
        self.materials_file = self._str(parser["project"]["materials_file"])
        self.channelling_cache_dir = self._str(parser["project"]["channelling_cache_dir"])
        self.analysis_dir = self._str(parser["project"]["analysis_output_dir"])
        self.map_dir = self._str(parser["project"]["map_output_dir"])
        self.reduction_factor = self._int(parser["resolution_reduction"]["reduction_factor"])
        self.scaling_tolerance = self._float(parser["resolution_reduction"]["scaling_tolerance"])
        self.pixel_size = self._float(parser["dislocation_density"]["pixel_size"])
        self.gnd_corrective_factor = self._float(parser["dislocation_density"]["corrective_factor"])
        self.beam_atomic_number = self._int(parser["channelling_fraction"]["beam_atomic_number"])
        self.beam_energy = self._float(parser["channelling_fraction"]["beam_energy"])
        self.beam_tilt = self._float(parser["channelling_fraction"]["beam_tilt"])
        self.neighbour_threshold = self._int(parser["orientation_clustering"]["neighbour_threshold"])
        self.neighbourhood_radius = self._float(parser["orientation_clustering"]["neighbourhood_radius"])
        self.use_cuda = self._bool(parser["orientation_clustering"]["use_cuda"])

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
