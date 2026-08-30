# -*- coding: utf-8 -*-
from abc import ABC
from enum import Enum
from math import cos

from src.data_structures.phase import BravaisLattice


class OrientationRelationshipCategory(Enum):
    TWIN = "twin"
    HETEROPHASE = "heterophase"


class OrientationRelationship(ABC):
    def __init__(self, id: str, category: OrientationRelationshipCategory):
        self.id = id
        self.category = category


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
