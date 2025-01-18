# -*- coding: utf-8 -*-

from dataclasses import dataclass
from math import pi, sqrt, sin, cos, tan, acos, atan2
from copy import deepcopy
from enum import Enum
from typing import Self
from numpy import ndarray, dot, array
from src.utilities.filestore import dump_phase, load_phase, load_phase_database_entry
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


class PhaseMissingError(LookupError):
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

    def save(self, phase_dir: str) -> None:
        dump_phase(self, phase_dir)

    @classmethod
    def load(cls, global_id: int, phase_dir: str, database_path: str = None) -> Self:
        try:
            return load_phase(global_id, phase_dir)
        except FileNotFoundError:
            print(f"Warning: No data found for phase with ID {global_id}.")

            if input("Enter phase information now? (Y/N): ").lower() == "y":
                phase = cls.build(global_id, database_path)
                phase.save(phase_dir)
                return phase
            else:
                raise PhaseMissingError(f"No data available for phase with ID {global_id}.")

    @dataclass
    class DatabaseEntry:
        global_id: int
        name: str
        lattice_type: BravaisLattice
        lattice_constants_nm: tuple[float, float, float]
        lattice_angles_deg: tuple[float, float, float]

    @classmethod
    def build(cls, global_id: int, database_path: str = None) -> Self:
        if database_path is not None:
            try:
                database_entry = load_phase_database_entry(global_id, database_path)
            except FileNotFoundError:
                print("Warning: Phase database missing. Manual entry required.")
                database_entry = None
            except PhaseMissingError:
                print(f"Warning: No phase found in database with ID {global_id}. Manual entry required.")
                database_entry = None
        else:
            database_entry = None

        if database_entry is None:
            name = input("Enter phase name: ")
            lattice_type = BravaisLattice[input("Enter Bravais lattice Pearson symbol: ").upper()]
            a = float(input("Enter first lattice constant (nm): "))
            b = float(input("Enter second lattice constant (nm): "))
            c = float(input("Enter third lattice constant (nm): "))
            alpha = float(input("Enter first lattice angle (deg): "))
            beta = float(input("Enter second lattice angle (deg): "))
            gamma = float(input("Enter third lattice angle (deg): "))

            database_entry = cls.DatabaseEntry(
                global_id=global_id,
                name=name,
                lattice_type=lattice_type,
                lattice_constants_nm=(a, b, c),
                lattice_angles_deg=(alpha, beta, gamma),
            )
        else:
            print(f"Name: {database_entry.name}")
            print(f"Lattice type: {database_entry.lattice_type.value}")
            print(f"Lattice constants: {", ".join(f"{constant} nm" for constant in database_entry.lattice_constants_nm)}")
            print(f"Lattice angles: {", ".join(f"{angle} deg" for angle in database_entry.lattice_angles_deg)}")

        atomic_number = float(input("Enter average atomic number: "))
        atomic_weight = float(input("Enter average atomic weight: "))
        density_cgs = float(input("Enter density (g/cm³): "))
        vibration_amplitude_nm = float(input("Enter thermal vibration amplitude (nm): "))
        diamond_structure = (database_entry.lattice_type == BravaisLattice.CF) and (input("Does crystal have diamond structure? (Y/N): ").lower() == "y")

        phase = Phase(
            global_id=global_id,
            name=database_entry.name,
            atomic_number=atomic_number,
            atomic_weight=atomic_weight,
            density_cgs=density_cgs,
            vibration_amplitude_nm=vibration_amplitude_nm,
            lattice_type=database_entry.lattice_type,
            lattice_constants_nm=database_entry.lattice_constants_nm,
            lattice_angles_deg=database_entry.lattice_angles_deg,
            diamond_structure=diamond_structure,
        )

        return phase
