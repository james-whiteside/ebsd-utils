# -*- coding: utf-8 -*-
from math import pi, degrees

import numpy
from numpy import ndarray

from src.data_structures.analysis import Analysis
from src.data_structures.orientation_relationship import OrientationRelationshipMatch, OrientationRelationshipCategory, \
    HeterophaseOrientationRelationship, TwinOrientationRelationship, OrientationRelationship
from src.utilities.geometry import rotation_angle, misrotation_matrix
from src.utilities.orientation import get_relationship_matrix, get_plane_family, get_twin_matrix, get_relationship_family


def get_matches(analysis: Analysis, orientation_relationships: list[OrientationRelationship]) -> list[OrientationRelationshipMatch]:
    matches: list[OrientationRelationshipMatch] = list()

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

            for variant in orientation_relationships:
                match variant:
                    case TwinOrientationRelationship():
                        if cluster_1_lattice_type == variant.lattice_type and cluster_2_lattice_type == variant.lattice_type and cluster_1_id < cluster_2_id:
                            family = get_plane_family(variant.reflection_plane)
                            theta = 2 * pi

                            for plane in family:
                                orientation_relationship = get_twin_matrix(plane)
                                misrotation = misrotation_matrix(numpy.dot(orientation_relationship, cluster_1_orientation), cluster_2_orientation)
                                theta = min(rotation_angle(misrotation), theta)

                            match = OrientationRelationshipMatch(variant.id, OrientationRelationshipCategory.TWIN, cluster_1_id, cluster_2_id, theta)
                            matches.append(match)
                    case HeterophaseOrientationRelationship():
                        if cluster_1_lattice_type == variant.lattice_type_1 and cluster_2_lattice_type == variant.lattice_type_2:
                            family = get_relationship_family(variant)
                            theta = 2 * pi

                            for relationship in family:
                                orientation_relationship = get_relationship_matrix(
                                    relationship.vector_pair_1[0],
                                    relationship.vector_pair_1[1],
                                    relationship.vector_pair_2[0],
                                    relationship.vector_pair_2[1],
                                    cluster_1_lattice_constants,
                                    cluster_2_lattice_constants
                                )

                                misrotation = misrotation_matrix(numpy.dot(orientation_relationship, cluster_1_orientation), cluster_2_orientation)
                                theta = min(rotation_angle(misrotation), theta)

                            match = OrientationRelationshipMatch(variant.id, OrientationRelationshipCategory.HETEROPHASE, cluster_1_id, cluster_2_id, theta)
                            matches.append(match)

    return sorted(matches, key=lambda match: degrees(match.misrotation))
