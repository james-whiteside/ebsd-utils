# -*- coding: utf-8 -*-

from math import pi, sqrt, sin, cos, tan, acos, atan2
from copy import deepcopy
from enum import Enum
from os import listdir, makedirs
from json import dump as json_dump, load as json_load
from typing import Self, Any
from numpy import ndarray, dot, array
from src.utilities.geometry import reduce_vector


class CrystalFamily(Enum):
    """
    Enumeration of the crystal lattice types, as listed in Tab. A2.1.
    """
    C = "c"
    T = "t"
    O = "o"
    H = "h"
    M = "m"
    A = "a"

    @property
    def max_euler_angles(self) -> tuple[float, float, float]:
        match self:
            case CrystalFamily.C:
                return 2 * pi, acos(sqrt(3) / 3), 0.5 * pi
            case _:
                raise NotImplementedError()

    def reduce_matrix(self, R: ndarray) -> ndarray:
        """
        Reduces a lattice orientation matrix ``R`` into the fundamental region of its Bravais lattice by reflection.
        :param R: The lattice orientation matrix ``R``.
        :return: The reduced matrix.
        """
        reduced_R = deepcopy(R)

        match self:
            case CrystalFamily.C:
                if reduced_R[2][2] < 0:
                    reduced_R = dot(array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]), reduced_R)

                if reduced_R[1][2] > 0:
                    reduced_R = dot(array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)

                if reduced_R[0][2] < 0:
                    reduced_R = dot(array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)

                if reduced_R[1][2] > reduced_R[0][2]:
                    reduced_R = dot(array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), reduced_R)
            case _:
                raise NotImplementedError()

        return reduced_R

    def ipf_coordinates(self, vector: tuple[float, float, float]) -> tuple[float, float]:
        """
        Computes the inverse pole figure coordinates ``(X, Y)`` of a lattice vector ``(u, v, w)`` for the given lattice symmetry.
        :param vector: The lattice vector ``(u, v, w)``.
        :return: The inverse pole figure coordinates ``(X, Y)``.
        """
        match self:
            case CrystalFamily.C:
                u, v, w = reduce_vector(vector)
                r = sqrt(u ** 2 + v ** 2 + w ** 2)
                theta = acos(w / r)
                phi = atan2(v, u)
                rho = tan(theta / 2)
                X = rho * cos(phi)
                Y = rho * sin(phi)
            case _:
                raise NotImplementedError()

        return X, Y


class BravaisLattice(Enum):
    """
    Enumeration of the Bravais lattice types, as listed in Tab. A2.1.
    """

    CP = "cP"
    CI = "cI"
    CF = "cF"
    TP = "tP"
    TI = "tI"
    OP = "oP"
    OI = "oI"
    OF = "oF"
    OS = "oS"
    HP = "hP"
    HR = "hR"
    MP = "mP"
    MS = "mS"
    AP = "aP"

    @property
    def family(self) -> CrystalFamily:
        return CrystalFamily(self.value[0])


class PhaseMissingError(FileNotFoundError):
    pass


class Phase:
    UNINDEXED_ID = 0
    GENERIC_BCC_ID = 4294967294
    GENERIC_FCC_ID = 4294967295
    GENERIC_IDS = [UNINDEXED_ID, GENERIC_BCC_ID, GENERIC_FCC_ID]

    def __init__(
        self,
        global_id: int,
        name: str,
        atomic_number: float,
        atomic_weight: float,
        density: float,
        vibration_amplitude: float,
        lattice_type: BravaisLattice,
        lattice_constants: tuple[float, float, float],
        lattice_angles: tuple[float, float, float],
        diamond_structure: bool,
    ):
        self.global_id = global_id
        self.name = name
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight
        self.density = density
        self.vibration_amplitude = vibration_amplitude
        self.lattice_type = lattice_type
        self.lattice_constants = lattice_constants
        self.lattice_angles = lattice_angles
        self.diamond_structure = diamond_structure

    def __eq__(self, other):
        if not isinstance(other, Phase):
            return False

        return self.global_id == other.global_id

    @property
    def close_pack_distance(self) -> float:
        match self.lattice_type:
            case BravaisLattice.CP:
                return self.lattice_constants[0]
            case BravaisLattice.CI:
                return sqrt(3) * self.lattice_constants[0] / 2
            case BravaisLattice.CF:
                return sqrt(2) * self.lattice_constants[0] / 2
            case _:
                raise NotImplementedError()

    def cache(self, cache_path: str) -> None:
        makedirs(cache_path, exist_ok=True)

        json_rep = {
            "global_id": self.global_id,
            "name": self.name,
            "atomic_number": self.atomic_number,
            "atomic_weight": self.atomic_weight,
            "density": self.density,
            "vibration_amplitude": self.vibration_amplitude,
            "lattice_type": self.lattice_type.value,
            "lattice_constants": list(self.lattice_constants),
            "lattice_angles": list(self.lattice_angles),
            "diamond_structure": self.diamond_structure,
        }

        with open(f"{cache_path}/{self.global_id}.json", "w") as file:
            json_dump(json_rep, file)

    @classmethod
    def cache_all(cls, cache_path: str, phases: list[Self]) -> None:
        for phase in phases:
            phase.cache(cache_path)

    @classmethod
    def load(cls, cache_path: str, global_id: int) -> Self:
        file_path = f"{cache_path}/{global_id}.json"

        try:
            with open(file_path, "r") as file:
                json_rep: dict[str, Any] = json_load(file)
        except FileNotFoundError:
            raise PhaseMissingError(f"No phase found in cache with global_id {global_id}")

        kwargs = {
            "global_id": json_rep["global_id"],
            "name": json_rep["name"],
            "atomic_number": json_rep["atomic_number"],
            "atomic_weight": json_rep["atomic_weight"],
            "density": json_rep["density"],
            "vibration_amplitude": json_rep["vibration_amplitude"],
            "lattice_type": BravaisLattice(json_rep["lattice_type"]),
            "lattice_constants": tuple(json_rep["lattice_constants"]),
            "lattice_angles": tuple(json_rep["lattice_angles"]),
            "diamond_structure": json_rep["diamond_structure"],
        }

        return Phase(**kwargs)

    @classmethod
    def load_all(cls, cache_path: str) -> dict[int, Self]:
        phases: dict[int, Phase] = dict()
        global_ids = [int(path.split(".")[0]) for path in listdir(cache_path)]

        for global_id in global_ids:
            phase = cls.load(cache_path, global_id)
            phases[phase.global_id] = phase

        return phases
