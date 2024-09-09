# -*- coding: utf-8 -*-

from math import radians, degrees, sin, cos
from src.utilities.geometry import Axis
from src.data_structures.phase import Phase


class ScanParams:
    def __init__(
        self,
        data_ref: str,
        width: int,
        height: int,
        phases: dict[int, Phase],
        reduction_factor: int,
    ):
        self.are_set = True
        self.data_ref = data_ref
        self.width = width
        self.height = height
        self.phases = phases
        self.reduction_factor = reduction_factor

    @property
    def analysis_ref(self) -> str:
        if self.reduction_factor == 0:
            return f"{self.data_ref}"
        else:
            return f"{self.data_ref}-{self.reduction_factor}"


class ScaleParams:
    def __init__(self, pixel_size_microns: float):
        self.pixel_size = pixel_size_microns * 10 ** -6

    @property
    def pixel_size_microns(self) -> float:
        return self.pixel_size * 10 ** 6


class ChannellingParams:
    def __init__(self, beam_atomic_number: int, beam_energy: float, beam_tilt_deg: float):
        self.beam_atomic_number = beam_atomic_number
        self.beam_energy = beam_energy
        self.beam_tilt_rad = radians(beam_tilt_deg)

    @property
    def beam_tilt_deg(self) -> float:
        return degrees(self.beam_tilt_rad)

    @property
    def beam_axis(self) -> Axis:
        beam_vector = 0, -sin(self.beam_tilt_rad), cos(self.beam_tilt_rad)
        return Axis.beam(beam_vector)

    @property
    def beam_vector(self) -> tuple[float, float, float]:
        return self.beam_axis.vector


class ClusteringParams:
    def __init__(self, core_point_threshold: int, neighbourhood_radius_deg: float):
        self.core_point_threshold = core_point_threshold
        self.neighbourhood_radius_rad = radians(neighbourhood_radius_deg)

    @property
    def neighbourhood_radius_deg(self) -> float:
        return degrees(self.neighbourhood_radius_rad)
