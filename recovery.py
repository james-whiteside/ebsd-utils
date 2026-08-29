# -*- coding: utf-8 -*-
import math

import numpy
from numpy import ndarray

from src.data_structures.analysis import Analysis
from src.data_structures.phase import BravaisLattice
from src.utilities.geometry import rotation_angle, misrotation_matrix
from src.utilities.orientation import get_relationship_matrix, get_plane_family, get_twin_matrix, \
    OrientationRelationship, get_relationship_family


def get_orientation_relationship_variants(path: str = "orientation/vars.csv") -> dict[str, OrientationRelationship]:
    variants: dict[str, OrientationRelationship] = dict()

    with open(path, "r") as file:
        file.readline()

        for line in file:
            data = line.split(",")
            name = data[0]
            lattice_type_1 = BravaisLattice(data[1])
            lattice_type_2 = BravaisLattice(data[2])
            vector_pair_1 = (int(data[3]), int(data[4]), int(data[5])), (int(data[6]), int(data[7]), int(data[8])) # u1A, u1B
            vector_pair_2 = (int(data[9]), int(data[10]), int(data[11])), (int(data[12]), int(data[13]), int(data[14])) # u2A, u2B
            variant = OrientationRelationship(lattice_type_1, lattice_type_2, vector_pair_1, vector_pair_2)
            variants[name] = variant

    return variants


def get_twin_variants(path: str = "orientation/twin.csv") -> dict[str, tuple[int, int, int]]:
    variants: dict[str, tuple[int, int, int]] = dict()

    with open(path, "r") as file:
        file.readline()

        for line in file:
            data = line.split(",")
            name = "twin-" + data[0] + data[1] + data[2]
            vector = int(data[0]), int(data[1]), int(data[2])
            variants[name] = vector

    return variants


def print_matches(analysis: Analysis) -> None:
    variants = get_orientation_relationship_variants()
    twins = get_twin_variants()
    matches = list()

    for cluster_1_id in analysis.cluster_aggregate.group_ids:
        for cluster_2_id in analysis.cluster_aggregate.group_ids:
            if cluster_1_id == cluster_2_id:
                continue

            cluster_1_phase = analysis.cluster_aggregate.phase.get_value_for(cluster_1_id)
            cluster_2_phase = analysis.cluster_aggregate.phase.get_value_for(cluster_2_id)
            cluster_1_lattice_type = cluster_1_phase.lattice_type
            cluster_2_lattice_type = cluster_2_phase.lattice_type
            cluster_1_lattice_constants = cluster_1_phase.lattice_constants
            cluster_2_lattice_constants = cluster_2_phase.lattice_constants
            cluster_1_orientation: ndarray = analysis.cluster_aggregate.reduced_matrix.get_value_for(cluster_1_id)
            cluster_2_orientation: ndarray = analysis.cluster_aggregate.reduced_matrix.get_value_for(cluster_2_id)

            for name, variant in variants.items():
                if variant.lattice_type_1 == cluster_1_lattice_type and variant.lattice_type_2 == cluster_2_lattice_type:
                    family = get_relationship_family(variant)
                    theta = 2 * math.pi

                    for relationship in family:
                        J = get_relationship_matrix(
                            relationship.vector_pair_1[0],
                            relationship.vector_pair_1[1],
                            relationship.vector_pair_2[0],
                            relationship.vector_pair_2[1],
                            cluster_1_lattice_constants,
                            cluster_2_lattice_constants
                        )

                        dR = misrotation_matrix(numpy.dot(J, cluster_1_orientation), cluster_2_orientation)
                        theta = min(rotation_angle(dR), theta)

                    match = dict()
                    match["variant"] = name
                    match["k1"] = cluster_1_id
                    match["k2"] = cluster_2_id
                    match["dTheta"] = theta
                    match["cosine"] = math.cos(theta)
                    matches.append(match)

            for name, vector in twins.items():
                if cluster_1_lattice_type == cluster_2_lattice_type and cluster_1_id < cluster_2_id:
                    family = get_plane_family(vector)
                    theta = 2 * math.pi

                    for plane in family:
                        J = get_twin_matrix(plane)
                        dR = misrotation_matrix(numpy.dot(J, cluster_1_orientation), cluster_2_orientation)
                        theta = min(rotation_angle(dR), theta)

                    match = dict()
                    match["variant"] = name
                    match["k1"] = cluster_1_id
                    match["k2"] = cluster_2_id
                    match["dTheta"] = theta
                    match["cosine"] = math.cos(theta)
                    matches.append(match)

    for match in sorted(matches, key=lambda item: math.degrees(item["dTheta"])):
        output = ""
        output += match["variant"] + ","
        output += str(match["k1"]) + ","
        output += str(match["k2"]) + ","
        output += str(math.degrees(match["dTheta"])) + ","
        output += str(match["cosine"]) + "\n"
        print(output)
