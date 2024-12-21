# -*- coding: utf-8 -*-

from math import radians, sin, cos
from src.utilities.geometry import Axis, AxisSet
from src.data_structures.phase import Phase


class ScanParams:
    def __init__(
        self,
        data_ref: str,
        width: int,
        height: int,
        phases: dict[int, Phase],
        pixel_size: float,
        reduction_factor: int,
    ):
        self.are_set = True
        self.data_ref = data_ref
        self.width = width
        self.height = height
        self.phases = phases
        self.pixel_size = pixel_size
        self.reduction_factor = reduction_factor

    @property
    def analysis_ref(self) -> str:
        if self.reduction_factor == 0:
            return f"{self.data_ref}"
        else:
            return f"{self.data_ref}-{self.reduction_factor}"

    @property
    def pixel_size_microns(self) -> float:
        return self.pixel_size * 10.0 ** 6.0


class ProjectParams:
    def __init__(
        self,
        data_dir: str,
        materials_file: str,
        cache_dir: str,
        analysis_dir: str,
        map_dir: str,
    ):
        self.data_dir = data_dir
        self.materials_file = materials_file
        self.cache_dir = cache_dir
        self.analysis_dir = analysis_dir
        self.map_dir = map_dir

    @property
    def phase_cache_dir(self) -> str:
        return f"{self.cache_dir}/phase"

    @property
    def channelling_cache_dir(self) -> str:
        return f"{self.cache_dir}/channelling"


class DataParams:
    def __init__(
        self,
        euler_axis_set: AxisSet,
        pixel_size_microns: float,
    ):
        self.euler_axis_set = euler_axis_set
        self.pixel_size_microns = pixel_size_microns

    @property
    def pixel_size(self) -> float:
        return self.pixel_size_microns * 10.0 ** -6.0


class AnalysisParams:
    def __init__(
        self,
        reduce_resolution: bool,
        compute_dislocation: bool,
        compute_channelling: bool,
        compute_clustering: bool,
        use_cuda: bool,
    ):
        self.reduce_resolution = reduce_resolution
        self.compute_dislocation = compute_dislocation
        self.compute_channelling = compute_channelling
        self.compute_clustering = compute_clustering
        self.use_cuda = use_cuda


class MapParams:
    def __init__(
        self,
        upscale_factor: int,
    ):
        self.upscale_factor = upscale_factor


class ResolutionParams:
    def __init__(
        self,
        reduction_factor: int,
        scaling_tolerance: float,
    ):
        self.reduction_factor = reduction_factor
        self.scaling_tolerance = scaling_tolerance


class DislocationParams:
    def __init__(
        self,
        corrective_factor: float,
    ):
        self.corrective_factor = corrective_factor


class ChannellingParams:
    def __init__(
        self,
        beam_atomic_number: int,
        beam_energy: float,
        beam_tilt_deg: float,
    ):
        self.beam_atomic_number = beam_atomic_number
        self.beam_energy = beam_energy
        self.beam_tilt_deg = beam_tilt_deg

    @property
    def beam_tilt_rad(self) -> float:
        return radians(self.beam_tilt_deg)

    @property
    def beam_axis(self) -> Axis:
        beam_vector = 0.0, -sin(self.beam_tilt_rad), cos(self.beam_tilt_rad)
        return Axis.beam(beam_vector)

    @property
    def beam_vector(self) -> tuple[float, float, float]:
        return self.beam_axis.vector


class ClusteringParams:
    def __init__(
        self,
        core_point_threshold: int,
        neighbourhood_radius_deg: float,
    ):
        self.core_point_threshold = core_point_threshold
        self.neighbourhood_radius_deg = neighbourhood_radius_deg

    @property
    def neighbourhood_radius_rad(self) -> float:
        return radians(self.neighbourhood_radius_deg)
