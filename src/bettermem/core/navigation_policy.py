"""Intent-conditioned navigation policy: P(v_{t+1} | v_t, I_t, S_t)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

from bettermem.core.nodes import NodeKind, TopicNode, get_topic_centroid
from bettermem.retrieval.intent import TraversalIntent
from bettermem.retrieval.relation import r_intent

if TYPE_CHECKING:
    from bettermem.core.graph import Graph


@dataclass
class SemanticState:
    """State S_t: query embedding, path history, and prior over nodes."""

    query_embedding: Optional[Sequence[float]] = None
    path_history: List[str] = field(default_factory=list)
    prior: Mapping[str, float] = field(default_factory=dict)


def _cos_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _get_topic_candidates(graph: "Graph", current_id: str) -> List[str]:
    """Return topic-node candidates: outgoing topic neighbors, parent, and siblings."""
    node = graph.get_node(current_id)
    if node is None:
        return []
    candidates_set: set = set()
    for neigh_id in graph.get_neighbors(current_id):
        neigh = graph.get_node(neigh_id)
        if neigh is not None and neigh.kind == NodeKind.TOPIC:
            candidates_set.add(neigh_id)
    if isinstance(node, TopicNode) and node.parent_id is not None:
        parent = graph.get_node(node.parent_id)
        if parent is not None and parent.kind == NodeKind.TOPIC:
            candidates_set.add(node.parent_id)
        for other in graph.iter_nodes():
            if other.kind != NodeKind.TOPIC or other.id == current_id:
                continue
            if isinstance(other, TopicNode) and other.parent_id == node.parent_id:
                candidates_set.add(other.id)
    return list(candidates_set)


def _repetition_penalty(node_id: str, path_history: Sequence[str], delta: float) -> float:
    """Return penalty for having visited node_id (delta per visit)."""
    count = sum(1 for x in path_history if x == node_id)
    return delta * count


class IntentConditionedPolicy:
    """Policy that scores and selects next topic node given v_t, I_t, S_t.

    Score(k) = α·cos(μ_k, q) + β·cos(μ_k, μ_i) + γ·R_intent(i,k) − δ·repetition_penalty
    """

    def __init__(
        self,
        graph: "Graph",
        *,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.5,
        delta: float = 0.3,
    ) -> None:
        self._graph = graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def score_candidate(
        self,
        current_id: str,
        candidate_id: str,
        intent: TraversalIntent,
        state: SemanticState,
    ) -> float:
        """Compute Score(k) for candidate k from current node i."""
        node_i = self._graph.get_node(current_id)
        node_k = self._graph.get_node(candidate_id)
        if node_i is None or node_k is None:
            return 0.0

        mu_i = get_topic_centroid(node_i) if isinstance(node_i, TopicNode) else None
        mu_k = get_topic_centroid(node_k) if isinstance(node_k, TopicNode) else None
        q = state.query_embedding

        term_relevance = 0.0
        if self.alpha != 0 and q is not None and mu_k is not None:
            term_relevance = self.alpha * _cos_sim(mu_k, q)

        term_continuity = 0.0
        if self.beta != 0 and mu_i is not None and mu_k is not None:
            term_continuity = self.beta * _cos_sim(mu_k, mu_i)

        term_intent = self.gamma * r_intent(current_id, candidate_id, intent, self._graph)

        rep = _repetition_penalty(candidate_id, state.path_history, self.delta)

        return term_relevance + term_continuity + term_intent - rep

    def next_distribution(
        self,
        current_id: str,
        intent: TraversalIntent,
        state: SemanticState,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Return P(k) = softmax(Score(k)/T) over topic candidates."""
        candidates = _get_topic_candidates(self._graph, current_id)
        if not candidates:
            return {}

        scores = [
            self.score_candidate(current_id, k, intent, state)
            for k in candidates
        ]
        if temperature <= 0:
            temperature = 1e-9
        exp_scores = [math.exp(s / temperature) for s in scores]
        total = sum(exp_scores)
        if total <= 0:
            return {k: 1.0 / len(candidates) for k in candidates}
        return {k: exp_scores[i] / total for i, k in enumerate(candidates)}

    def step(
        self,
        current_id: str,
        intent: TraversalIntent,
        state: SemanticState,
        *,
        temperature: float = 1.0,
        greedy: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """Choose next node: greedy argmax or sample from distribution."""
        dist = self.next_distribution(current_id, intent, state, temperature=temperature)
        if not dist:
            return None
        if greedy:
            return max(dist.items(), key=lambda x: x[1])[0]
        r = (rng or random).random()
        cum = 0.0
        for k, p in dist.items():
            cum += p
            if r <= cum:
                return k
        return list(dist.keys())[-1]
