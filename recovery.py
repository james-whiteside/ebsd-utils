# -*- coding: utf-8 -*-
from src.data_structures.orientation_relationship import TwinOrientationRelationship, OrientationRelationshipCategory, \
    OrientationRelationship
from src.data_structures.phase import BravaisLattice
from src.utilities.orientation import HeterophaseOrientationRelationship


def get_orientation_relationships(path: str = "orientation-relationship/vars.csv") -> list[OrientationRelationship]:
    with open(path, "r") as file:
        file.readline()
        category = OrientationRelationshipCategory(file.readline())
        file.readline()
        relationships: list[OrientationRelationship] = list()

        match category:
            case OrientationRelationshipCategory.TWIN:
                for line in file:
                    data = line.split(",")
                    id = data[0]
                    lattice_type = BravaisLattice(data[1])
                    plane = int(data[2]), int(data[3]), int(data[4])
                    relationship = TwinOrientationRelationship(id, lattice_type, plane)
                    relationships.append(relationship)
            case OrientationRelationshipCategory.HETEROPHASE:
                for line in file:
                    data = line.split(",")
                    id = data[0]
                    lattice_type_1 = BravaisLattice(data[1])
                    lattice_type_2 = BravaisLattice(data[2])
                    vector_pair_1 = (int(data[3]), int(data[4]), int(data[5])), (int(data[6]), int(data[7]), int(data[8]))
                    vector_pair_2 = (int(data[9]), int(data[10]), int(data[11])), (int(data[12]), int(data[13]), int(data[14]))
                    relationship = HeterophaseOrientationRelationship(id, lattice_type_1, lattice_type_2, vector_pair_1, vector_pair_2)
                    relationships.append(relationship)

        return relationships
