# -*- coding: utf-8 -*-
from math import pi, degrees

import numpy
from numpy import ndarray

from src.data_structures.aggregate_manager import AggregateManager
from src.data_structures.orientation_relationship import (
    OrientationRelationshipCategory,
    OrientationRelationship,
    TwinOrientationRelationship,
    HeterophaseOrientationRelationship,
    OrientationRelationshipMatch,
    OrientationRelationshipSummary,
)
from src.utilities.geometry import rotation_angle, misrotation_matrix


def orientation_relationship_matches(cluster_aggregate: AggregateManager, orientation_relationships: list[OrientationRelationship]) -> OrientationRelationshipSummary:
    matches: list[OrientationRelationshipMatch] = list()

    for cluster_1_id in cluster_aggregate.group_ids:
        for cluster_2_id in cluster_aggregate.group_ids:
            if cluster_1_id == cluster_2_id:
                continue

            cluster_1_phase = cluster_aggregate.phase.get_value_for(cluster_1_id)
            cluster_2_phase = cluster_aggregate.phase.get_value_for(cluster_2_id)
            cluster_1_lattice_type = cluster_1_phase.lattice_type
            cluster_2_lattice_type = cluster_2_phase.lattice_type
            cluster_1_lattice_constants = cluster_1_phase.lattice_constants
            cluster_2_lattice_constants = cluster_2_phase.lattice_constants
            cluster_1_orientation: ndarray = cluster_aggregate.reduced_matrix.get_value_for(cluster_1_id)
            cluster_2_orientation: ndarray = cluster_aggregate.reduced_matrix.get_value_for(cluster_2_id)

            for variant in orientation_relationships:
                match variant:
                    case TwinOrientationRelationship():
                        if cluster_1_lattice_type == variant.lattice_type and cluster_2_lattice_type == variant.lattice_type and cluster_1_id < cluster_2_id:
                            theta = 2 * pi

                            for relationship in variant.family:
                                relationship_matrix = relationship.get_matrix()
                                misrotation = misrotation_matrix(numpy.dot(relationship_matrix, cluster_1_orientation), cluster_2_orientation)
                                theta = min(rotation_angle(misrotation), theta)

                            match = OrientationRelationshipMatch(variant.id, OrientationRelationshipCategory.TWIN, cluster_1_id, cluster_2_id, theta)
                            matches.append(match)
                    case HeterophaseOrientationRelationship():
                        if cluster_1_lattice_type == variant.lattice_type_1 and cluster_2_lattice_type == variant.lattice_type_2:
                            theta = 2 * pi

                            for relationship in variant.family:
                                relationship_matrix = relationship.get_matrix(cluster_1_lattice_constants, cluster_2_lattice_constants)
                                misrotation = misrotation_matrix(numpy.dot(relationship_matrix, cluster_1_orientation), cluster_2_orientation)
                                theta = min(rotation_angle(misrotation), theta)

                            match = OrientationRelationshipMatch(variant.id, OrientationRelationshipCategory.HETEROPHASE, cluster_1_id, cluster_2_id, theta)
                            matches.append(match)

    return OrientationRelationshipSummary(matches)
