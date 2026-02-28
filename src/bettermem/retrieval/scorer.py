from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from bettermem.core.nodes import NodeId


@dataclass
class PathVisit:
    path: Sequence[NodeId]
    log_prob: float


@dataclass
class QueryScorer:
    """Accumulate scores over nodes from traversal results."""

    node_scores: Dict[NodeId, float] = field(default_factory=dict)
    path_visits: List[PathVisit] = field(default_factory=list)

    def add_paths(self, paths: Iterable[Sequence[NodeId]], log_probs: Iterable[float]) -> None:
        for path, lp in zip(paths, log_probs):
            self.path_visits.append(PathVisit(path=list(path), log_prob=float(lp)))
            weight = _log_prob_to_weight(lp)
            for node in path:
                self.node_scores[node] = self.node_scores.get(node, 0.0) + weight

    def add_visit_counts(self, visit_counts: Mapping[NodeId, int]) -> None:
        for node, count in visit_counts.items():
            self.node_scores[node] = self.node_scores.get(node, 0.0) + float(count)

    def scores(self) -> Mapping[NodeId, float]:
        return self.node_scores


def _log_prob_to_weight(log_prob: float) -> float:
    """Map a log-probability to a positive weight."""
    import math

    # exp(log_prob) can underflow; cap at a small positive value
    return max(math.exp(min(0.0, log_prob)), 1e-12)

