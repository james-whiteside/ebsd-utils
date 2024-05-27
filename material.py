import math
from enum import Enum


UNINDEXED_PHASE_ID = 0
GENERIC_BCC_PHASE_ID = 4294967294
GENERIC_FCC_PHASE_ID = 4294967295
GENERIC_PHASE_IDS = (UNINDEXED_PHASE_ID, GENERIC_BCC_PHASE_ID, GENERIC_FCC_PHASE_ID)


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
    NONE = "None"


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
    NONE = "None"

    def get_family(self) -> CrystalFamily:
        if self is BravaisLattice.NONE:
            return CrystalFamily.NONE
        else:
            return CrystalFamily(self.value[0])


class Material:
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
        has_diamond_structure: bool,
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
        self.has_diamond_structure = has_diamond_structure

    def __eq__(self, other):
        if not isinstance(other, Material):
            return False

        if self.global_id == other.global_id:
            return True
        else:
            return False

    @property
    def close_pack_distance(self) -> float:
        if self.lattice_type is BravaisLattice.CP:
            return self.lattice_constants[0]
        elif self.lattice_type is BravaisLattice.CI:
            return math.sqrt(3) * self.lattice_constants[0] / 2
        elif self.lattice_type is BravaisLattice.CF:
            return math.sqrt(2) * self.lattice_constants[0] / 2
        else:
            raise NotImplementedError

    def as_dict(self):
        return {
            "global_id": self.global_id,
            "name": self.name,
            "Z": self.atomic_number,
            "A": self.atomic_weight,
            "density": self.density,
            "vibration": self.vibration_amplitude,
            "type": self.lattice_type,
            "constants": self.lattice_constants,
            "angles": self.lattice_angles,
            "diamond": self.has_diamond_structure,
        }
