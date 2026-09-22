# -*- coding: utf-8 -*-

from math import pi

from src.algorithms.orientation_relationship import orientation_relationship_matches
from src.data_structures.aggregate_manager import AggregateManager
from src.data_structures.orientation_relationship import OrientationRelationship, OrientationRelationshipMatch
from src.utilities.logging import Logger
from src.utilities.utils import format_sig_figs


class OrientationRelationshipSummary:
    def __init__(
            self,
            cluster_aggregate: AggregateManager,
            orientation_relationships: list[OrientationRelationship],
            logger: Logger,
            maximum_misrotation_rad: float = pi,
    ):
        self._cluster_aggregate = cluster_aggregate
        self._orientation_relationships = orientation_relationships
        self._logger = logger
        self.maximum_misrotation_rad = maximum_misrotation_rad
        self._matches = None
        self._matches_by_cluster = None

    @property
    def matches(self) -> list[OrientationRelationshipMatch]:
        if self._matches is None:
            self._logger.debug("Generating orientation relationship matches...")
            matches = orientation_relationship_matches(self._cluster_aggregate, self._orientation_relationships)
            filtered_matches = [match for match in matches if match.misrotation <= self.maximum_misrotation_rad]
            self._matches = sorted(filtered_matches, key=lambda match: match.misrotation)

        return self._matches

    @property
    def matches_by_cluster(self) -> dict[int, list[OrientationRelationshipMatch]]:
        if self._matches_by_cluster is None:
            cluster_ids = {match.cluster_1_id for match in self.matches} | {match.cluster_2_id for match in self.matches}

            self._matches_by_cluster = {
                cluster_id: [match for match in self.matches if cluster_id == match.cluster_1_id or cluster_id == match.cluster_2_id]
                for cluster_id in cluster_ids
            }

        return self._matches_by_cluster

    def closest_match_for_cluster(self, cluster_id: int) -> OrientationRelationshipMatch | None:
        try: return self.matches_by_cluster[cluster_id][0]
        except KeyError: return None

    def serialize_closest_match_for(self, cluster_id: int, null_serialization: str = "", sig_figs: int = None) -> list[str]:
        match = self.closest_match_for_cluster(cluster_id)
        if match is None: return [null_serialization for _ in range(4)]
        other_id = match.cluster_2_id if cluster_id == match.cluster_1_id else match.cluster_1_id

        def format(value: float) -> str:
            if sig_figs is not None:
                return format_sig_figs(value, sig_figs)
            else:
                return str(value)

        return [match.relationship_id, str(other_id), format(match.misrotation_deg), format(match.alignment)]