# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from enum import Enum
from itertools import permutations
from math import cos
from typing import Self

from numpy import ndarray

from src.data_structures.phase import BravaisLattice, CrystalFamily, SymmetryNotImplementedError
from src.utilities.orientation import get_plane_family, get_cubic_twin_relationship_matrix, get_heterophase_relationship_matrix
from src.utilities.utils import format_sig_figs


class OrientationRelationshipCategory(Enum):
    TWIN = "twin"
    HETEROPHASE = "heterophase"


class OrientationRelationship(ABC):
    def __init__(self, id: str, category: OrientationRelationshipCategory):
        self.id = id
        self.category = category

    @property
    @abstractmethod
    def family(self) -> list[Self]:
        ...

    @abstractmethod
    def uses_lattice_type(self, lattice_type: BravaisLattice) -> bool:
        ...



class TwinOrientationRelationship(OrientationRelationship):
    def __init__(
            self,
            id: str,
            lattice_type: BravaisLattice,
            reflection_plane: tuple[int, int, int],
    ):
        super().__init__(id, OrientationRelationshipCategory.TWIN)
        self.lattice_type = lattice_type
        self.reflection_plane = reflection_plane

    @property
    def family(self) -> list[Self]:
        plane_family = get_plane_family(self.reflection_plane)
        return [TwinOrientationRelationship(self.id, self.lattice_type, plane) for plane in plane_family]

    def get_matrix(self) -> ndarray:
        match self.lattice_type.family:
            case CrystalFamily.C: return get_cubic_twin_relationship_matrix(self.reflection_plane)
            case _: raise SymmetryNotImplementedError(self.lattice_type.family)

    def uses_lattice_type(self, lattice_type: BravaisLattice) -> bool:
        return lattice_type is self.lattice_type



class HeterophaseOrientationRelationship(OrientationRelationship):
    def __init__(
            self,
            id: str,
            lattice_type_1: BravaisLattice,
            lattice_type_2: BravaisLattice,
            vector_pair_1: tuple[tuple[int, int, int], tuple[int, int, int]],
            vector_pair_2: tuple[tuple[int, int, int], tuple[int, int, int]],
    ):
        super().__init__(id, OrientationRelationshipCategory.HETEROPHASE)
        self.lattice_type_1 = lattice_type_1
        self.lattice_type_2 = lattice_type_2
        self.vector_pair_1 = vector_pair_1
        self.vector_pair_2 = vector_pair_2

    @property
    def family(self) -> list[Self]:
        family: list[HeterophaseOrientationRelationship] = list()

        parity_sets = sorted(set(permutations((1, 1, 1, 1, -1, -1, -1, -1), 4)), reverse=True)
        assert (len(parity_sets) == 16)

        for parity_set in parity_sets:
            reflected_vector_pair_1 = (
                (
                    parity_set[0] * self.vector_pair_1[0][0],
                    parity_set[0] * self.vector_pair_1[0][1],
                    parity_set[0] * self.vector_pair_1[0][2],
                ),
                (
                    parity_set[1] * self.vector_pair_1[1][0],
                    parity_set[1] * self.vector_pair_1[1][1],
                    parity_set[1] * self.vector_pair_1[1][2],
                ),
            )

            reflected_vector_pair_2 = (
                (
                    parity_set[2] * self.vector_pair_2[0][0],
                    parity_set[2] * self.vector_pair_2[0][1],
                    parity_set[2] * self.vector_pair_2[0][2],
                ),
                (
                    parity_set[3] * self.vector_pair_2[1][0],
                    parity_set[3] * self.vector_pair_2[1][1],
                    parity_set[3] * self.vector_pair_2[1][2],
                ),
            )

            reflected_relationship = HeterophaseOrientationRelationship(
                self.id,
                self.lattice_type_1,
                self.lattice_type_2,
                reflected_vector_pair_1,
                reflected_vector_pair_2,
            )

            family.append(reflected_relationship)

        return family

    def get_matrix(self, lattice_constants_1: tuple[float, float, float], lattice_constants_2: tuple[float, float, float]) -> ndarray:
        return get_heterophase_relationship_matrix(
            self.vector_pair_1[0],
            self.vector_pair_1[1],
            self.vector_pair_2[0],
            self.vector_pair_2[1],
            lattice_constants_1,
            lattice_constants_2,
        )

    def uses_lattice_type(self, lattice_type: BravaisLattice) -> bool:
        return lattice_type is self.lattice_type_1 or lattice_type is self.lattice_type_2


class OrientationRelationshipMatch:
    def __init__(
            self,
            relationship_id: str,
            category: OrientationRelationshipCategory,
            cluster_1_id: int,
            cluster_2_id: int,
            misrotation: float
    ):
        self.relationship_id = relationship_id
        self.category = category
        self.cluster_1_id = cluster_1_id
        self.cluster_2_id = cluster_2_id
        self.misrotation = misrotation

    @property
    def alignment(self) -> float:
        return cos(self.misrotation)

    def serialize_value(self, sig_figs: int = None) -> list[str]:
        def format(value: float) -> str:
            if sig_figs is not None:
                return format_sig_figs(value, sig_figs)
            else:
                return str(value)

        return [self.relationship_id, str(self.cluster_1_id), str(self.cluster_2_id), format(self.misrotation), format(self.alignment)]


class OrientationRelationshipSummary:
    def __init__(self, matches: list[OrientationRelationshipMatch]):
        self.matches = sorted(matches, key=lambda match: match.misrotation)
        cluster_ids = {match.cluster_1_id for match in self.matches} | {match.cluster_2_id for match in self.matches}

        self._matches_by_cluster = {
            cluster_id: [match for match in matches if cluster_id == match.cluster_1_id or cluster_id == match.cluster_2_id]
            for cluster_id in cluster_ids
        }

    def closest_match_for_cluster(self, cluster_id: int) -> OrientationRelationshipMatch | None:
        try: return self._matches_by_cluster[cluster_id][0]
        except KeyError: return None

    def serialize_closest_match_for(self, cluster_id: int, null_serialization: str = "", sig_figs: int = None) -> list[str]:
        match = self.closest_match_for_cluster(cluster_id)
        if match is None: return [null_serialization for _ in range(4)]
        other_id = match.cluster_2_id if cluster_id == match.cluster_1_id else match.cluster_2_id

        def format(value: float) -> str:
            if sig_figs is not None:
                return format_sig_figs(value, sig_figs)
            else:
                return str(value)

        return [match.relationship_id, str(other_id), format(match.misrotation), format(match.alignment)]

