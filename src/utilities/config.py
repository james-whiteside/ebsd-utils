# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser


class Config:
    def __init__(self, path: str = "config.ini"):
        config_path = f"{os.getcwd()}/{path}"
        parser = ConfigParser()
        parser.read(config_path)
        self.ebsd_data_dir = self._str(parser["project"]["ebsd_data_dir"])
        self.materials_file = self._str(parser["project"]["materials_file"])
        self.channelling_cache_dir = self._str(parser["project"]["channelling_cache_dir"])
        self.analysis_output_dir = self._str(parser["project"]["analysis_output_dir"])
        self.map_output_dir = self._str(parser["project"]["map_output_dir"])
        self.unindexed_phase_id = self._int(parser["phases"]["unindexed_phase_id"])
        self.generic_bcc_phase_id = self._int(parser["phases"]["generic_bcc_phase_id"])
        self.generic_fcc_phase_id = self._int(parser["phases"]["generic_fcc_phase_id"])
        self.resolution_reduction_scaling_tolerance = self._float(parser["resolution_reduction"]["scaling_tolerance"])
        self.gnd_density_corrective_factor = self._float(parser["dislocation_density"]["corrective_factor"])
        self.orientation_clustering_use_cuda = self._bool(parser["orientation_clustering"]["use_cuda"])

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
