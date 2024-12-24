# -*- coding: utf-8 -*-

from math import pi, sqrt, sin, cos, tan, acos, atan2
from copy import deepcopy
from enum import Enum
from os import listdir, makedirs
from json import dump as json_dump, load as json_load
from typing import Self, Any
from xml.etree import ElementTree
from numpy import ndarray, dot, array
from src.utilities.geometry import reduce_vector
from src.utilities.utils import tuple_radians


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

    @property
    def code(self) -> int:
        match self:
            case BravaisLattice.CP: return 1
            case BravaisLattice.CI: return 2
            case BravaisLattice.CF: return 3
            case BravaisLattice.TP: return 4
            case BravaisLattice.TI: return 5
            case BravaisLattice.OP: return 6
            case BravaisLattice.OI: return 7
            case BravaisLattice.OS: return 8
            case BravaisLattice.OF: return 9
            case BravaisLattice.HR: return 10
            case BravaisLattice.HP: return 11
            case BravaisLattice.MP: return 12
            case BravaisLattice.MS: return 13
            case BravaisLattice.AP: return 14

    @classmethod
    def from_code(cls, code: int) -> Self:
        for lattice in BravaisLattice:
            if lattice.code == code:
                return lattice

        raise ValueError(f"Value is not a valid Bravais lattice code: {code}")


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
        density_cgs: float,
        vibration_amplitude_nm: float,
        lattice_type: BravaisLattice,
        lattice_constants_nm: tuple[float, float, float],
        lattice_angles_deg: tuple[float, float, float],
        diamond_structure: bool,
    ):
        self.global_id = global_id
        self.name = name
        self.atomic_number = atomic_number
        self.atomic_weight = atomic_weight
        self.density_cgs = density_cgs
        self.vibration_amplitude_nm = vibration_amplitude_nm
        self.lattice_type = lattice_type
        self.lattice_constants_nm = lattice_constants_nm
        self.lattice_angles_deg = lattice_angles_deg
        self.diamond_structure = diamond_structure

    def __eq__(self, other):
        if not isinstance(other, Phase):
            return False

        return self.global_id == other.global_id

    @property
    def density(self) -> float:
        return self.density_cgs * 10.0 ** 3.0

    @property
    def vibration_amplitude(self) -> float:
        return self.vibration_amplitude_nm * 10 ** -9.0

    @property
    def lattice_constants(self) -> tuple[float, float, float]:
        return (
            self.lattice_constants_nm[0] * 10.0 ** -9.0,
            self.lattice_constants_nm[1] * 10.0 ** -9.0,
            self.lattice_constants_nm[2] * 10.0 ** -9.0,
        )

    @property
    def lattice_angles_rad(self) -> tuple[float, float, float]:
        return tuple_radians(self.lattice_angles_deg)

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

    @property
    def close_pack_distance_nm(self) -> float:
        return self.close_pack_distance * 10.0 ** 9.0

    def cache(self, cache_path: str) -> None:
        makedirs(cache_path, exist_ok=True)

        json_rep = {
            "global_id": self.global_id,
            "name": self.name,
            "atomic_number": self.atomic_number,
            "atomic_weight": self.atomic_weight,
            "density_cgs": self.density_cgs,
            "vibration_amplitude_nm": self.vibration_amplitude_nm,
            "lattice_type": self.lattice_type.value,
            "lattice_constants_nm": list(self.lattice_constants_nm),
            "lattice_angles_deg": list(self.lattice_angles_deg),
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
            raise PhaseMissingError(f"No phase found in cache with ID {global_id}.")

        kwargs = {
            "global_id": json_rep["global_id"],
            "name": json_rep["name"],
            "atomic_number": json_rep["atomic_number"],
            "atomic_weight": json_rep["atomic_weight"],
            "density_cgs": json_rep["density_cgs"],
            "vibration_amplitude_nm": json_rep["vibration_amplitude_nm"],
            "lattice_type": BravaisLattice(json_rep["lattice_type"]),
            "lattice_constants_nm": tuple(json_rep["lattice_constants_nm"]),
            "lattice_angles_deg": tuple(json_rep["lattice_angles_deg"]),
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

    @classmethod
    def build(cls, global_id: int, database_path: str | None) -> Self:
        info_found = False

        if database_path is not None:
            db = ElementTree.parse(database_path).getroot()

            for phase_info in db.iter("CrystalPhaseInfo"):
                if int(phase_info.find("CrystalID").text) == global_id:
                    info_found = True
                    global_id = int(phase_info.find("CrystalID").text)
                    name = phase_info.find("ElementName").text
                    lattice_type = BravaisLattice.from_code(int(phase_info.find("BravaisLatticeID").text))
                    a = float(phase_info.find("Cell_A").text)
                    b = float(phase_info.find("Cell_B").text)
                    c = float(phase_info.find("Cell_C").text)
                    alpha = float(phase_info.find("Cell_Alpha").text)
                    beta = float(phase_info.find("Cell_Beta").text)
                    gamma = float(phase_info.find("Cell_Gamma").text)
                    print(f"Name: {name}")
                    print(f"Lattice type: {lattice_type.value}")
                    print(f"Lattice constants: {a} nm, {b} nm, {c} nm")
                    print(f"Lattice angles: {alpha} deg, {beta} deg, {gamma} deg")
                    break

            if not info_found:
                print(f"Warning: No phase found in database with ID {global_id}. Manual entry required.")

        if not info_found:
            name = input("Enter phase name: ")
            lattice_type = BravaisLattice[input("Enter Bravais lattice Pearson symbol: ").upper()]
            a = float(input("Enter first lattice constant (nm): "))
            b = float(input("Enter second lattice constant (nm): "))
            c = float(input("Enter third lattice constant (nm): "))
            alpha = float(input("Enter first lattice angle (deg): "))
            beta = float(input("Enter second lattice angle (deg): "))
            gamma = float(input("Enter third lattice angle (deg): "))

        atomic_number = float(input("Enter average atomic number: "))
        atomic_weight = float(input("Enter average atomic weight: "))
        density_cgs = float(input("Enter density (g/cm³): "))
        vibration_amplitude_nm = float(input("Enter thermal vibration amplitude (nm): "))
        diamond_structure = (lattice_type.value == "cF") and (input("Does crystal have diamond structure? (Y/N): ").lower() == "y")

        phase = Phase(
            global_id=global_id,
            name=name,
            atomic_number=atomic_number,
            atomic_weight=atomic_weight,
            density_cgs=density_cgs,
            vibration_amplitude_nm=vibration_amplitude_nm,
            lattice_type=lattice_type,
            lattice_constants_nm=(a, b, c),
            lattice_angles_deg=(alpha, beta, gamma),
            diamond_structure=diamond_structure,
        )

        return phase
