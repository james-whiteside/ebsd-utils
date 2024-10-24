# -*- coding: utf-8 -*-

import math
from copy import deepcopy
from enum import Enum
from typing import Self
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
                return 2 * math.pi, math.acos(math.sqrt(3) / 3), 0.5 * math.pi
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
                r = math.sqrt(u ** 2 + v ** 2 + w ** 2)
                theta = math.acos(w / r)
                phi = math.atan2(v, u)
                rho = math.tan(theta / 2)
                X = rho * math.cos(phi)
                Y = rho * math.sin(phi)
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


class Phase:
    UNINDEXED_ID = 0
    GENERIC_BCC_ID = 4294967294
    GENERIC_FCC_ID = 4294967295
    GENERIC_PHASES = [UNINDEXED_ID, GENERIC_BCC_ID, GENERIC_FCC_ID]

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
                return math.sqrt(3) * self.lattice_constants[0] / 2
            case BravaisLattice.CF:
                return math.sqrt(2) * self.lattice_constants[0] / 2
            case _:
                raise NotImplementedError()

    @classmethod
    def from_materials_file(cls, materials_file: str) -> dict[int, Self]:
        path = materials_file
        phases = dict()

        with open(path, "r") as file:
            file.readline()

            for line in file:
                args = line.split(',')
                global_id = int(args[0])

                phase = Phase(
                    global_id=global_id,
                    name=args[1],
                    atomic_number=float(args[2]),
                    atomic_weight=float(args[3]),
                    density=float(args[4]),
                    vibration_amplitude=float(args[5]),
                    lattice_type=BravaisLattice(args[6]),
                    lattice_constants=(float(args[7]), float(args[8]), float(args[9])),
                    lattice_angles=(math.radians(float(args[10])), math.radians(float(args[11])), math.radians(float(args[12]))),
                    diamond_structure=args == "Y",
                )

                phases[global_id] = phase

        return phases
