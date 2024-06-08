# -*- coding: utf-8 -*-

import math
from numpy import ndarray, array
from geometry import Axis, AxisSet
from phase import Phase


class ScanParameters:
    def __init__(self):
        self.are_set = False
        self._file_reference = None
        self._width = None
        self._height = None
        self._phases = None
        self._axis_set = None

    def set(
        self,
        file_reference: str,
        width: int,
        height: int,
        phases: dict[int, Phase],
        axis_set: AxisSet,
    ) -> None:
        if self.are_set:
            raise RuntimeError("Scale parameters cannot be set more than once.")

        self.are_set = True
        self._file_reference = file_reference
        self._width = width
        self._height = height
        self._phases = phases
        self._axis_set = axis_set

    @property
    def file_reference(self) -> str:
        if not self.are_set:
            raise AttributeError("Scan parameters have not been set.")
        else:
            return self._file_reference

    @property
    def width(self) -> int:
        if not self.are_set:
            raise AttributeError("Scan parameters have not been set.")
        else:
            return self._width

    @property
    def height(self) -> int:
        if not self.are_set:
            raise AttributeError("Scan parameters have not been set.")
        else:
            return self._height

    @property
    def phases(self) -> dict[int, Phase]:
        if not self.are_set:
            raise AttributeError("Scan parameters have not been set.")
        else:
            return self._phases

    @property
    def axis_set(self) -> AxisSet:
        if not self.are_set:
            raise AttributeError("Scan parameters have not been set.")
        else:
            return self._axis_set


class ScaleParameters:
    def __init__(self):
        self.are_set = False
        self._pixel_size = None

    def set(self, pixel_size_micrometres: float) -> None:
        if self.are_set:
            raise RuntimeError("Scale parameters cannot be set more than once.")

        pixel_size = pixel_size_micrometres * 10 ** -6
        self.are_set = True
        self._pixel_size = pixel_size

    @property
    def pixel_size(self) -> float:
        if not self.are_set:
            raise AttributeError("Scale parameters have not been set.")
        else:
            return self._pixel_size

    @property
    def pixel_size_micrometres(self) -> float:
        return self.pixel_size * 10 ** 6


class ChannellingParameters:
    def __init__(self):
        self.are_set = False
        self._beam_atomic_number = None
        self._beam_energy = None
        self._beam_axis = Axis.Y
        self._beam_tilt = None

    def set(self, beam_atomic_number: int, beam_energy: float, beam_tilt_degrees: float) -> None:
        if self.are_set:
            raise RuntimeError("Channelling parameters cannot be set more than once.")

        beam_tilt = math.radians(beam_tilt_degrees)
        self.are_set = True
        self._beam_atomic_number = beam_atomic_number
        self._beam_energy = beam_energy
        self._beam_tilt = beam_tilt

    @property
    def beam_atomic_number(self) -> int:
        if not self.are_set:
            raise AttributeError("Channelling parameters have not been set.")
        else:
            return self._beam_atomic_number

    @property
    def beam_energy(self) -> float:
        if not self.are_set:
            raise AttributeError("Channelling parameters have not been set.")
        else:
            return self._beam_energy

    @property
    def beam_tilt(self) -> float:
        if not self.are_set:
            raise AttributeError("Channelling parameters have not been set.")
        else:
            return self._beam_tilt

    @property
    def beam_tilt_degrees(self) -> float:
        return math.degrees(self.beam_tilt)

    @property
    def beam_vector(self) -> ndarray:
        match self._beam_axis:
            case Axis.X:
                raise NotImplementedError()
            case Axis.Y:
                return array((0, -math.sin(self.beam_tilt), math.cos(self.beam_tilt)))
            case Axis.Z:
                raise NotImplementedError()


class ClusteringParameters:
    def __init__(self):
        self.are_set = False
        self._core_point_neighbour_threshold = None
        self._neighbourhood_radius = None

    def set(self, core_point_neighbour_threshold: int, neighbourhood_radius_degrees: float) -> None:
        if self.are_set:
            raise RuntimeError("Clustering parameters cannot be set more than once.")

        neighbourhood_radius = math.radians(neighbourhood_radius_degrees)
        self.are_set = True
        self._core_point_neighbour_threshold = core_point_neighbour_threshold
        self._neighbourhood_radius = neighbourhood_radius

    @property
    def core_point_neighbour_threshold(self) -> int:
        if not self.are_set:
            raise AttributeError("Clustering parameters have not been set.")
        else:
            return self._core_point_neighbour_threshold

    @property
    def neighbourhood_radius(self) -> float:
        if not self.are_set:
            raise AttributeError("Clustering parameters have not been set.")
        else:
            return self._neighbourhood_radius

    @property
    def neighbourhood_radius_degrees(self) -> float:
        return math.degrees(self.neighbourhood_radius)
