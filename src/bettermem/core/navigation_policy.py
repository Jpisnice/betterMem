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
    """Return topic-node candidates: outgoing topic neighbors, parents, children, and siblings."""
    node = graph.get_node(current_id)
    if node is None:
        return []
    candidates_set: set = set()
    for neigh_id in graph.get_neighbors(current_id):
        neigh = graph.get_node(neigh_id)
        if neigh is not None and neigh.kind == NodeKind.TOPIC:
            candidates_set.add(neigh_id)
    for parent_id in graph.get_parents(current_id):
        n = graph.get_node(parent_id)
        if n is not None and n.kind == NodeKind.TOPIC:
            candidates_set.add(parent_id)
    for child_id in graph.get_children(current_id):
        n = graph.get_node(child_id)
        if n is not None and n.kind == NodeKind.TOPIC:
            candidates_set.add(child_id)
    for sib_id in graph.get_siblings(current_id):
        n = graph.get_node(sib_id)
        if n is not None and n.kind == NodeKind.TOPIC:
            candidates_set.add(sib_id)
    if isinstance(node, TopicNode) and node.parent_ids:
        for pid in node.parent_ids:
            parent = graph.get_node(pid)
            if parent is not None and parent.kind == NodeKind.TOPIC:
                candidates_set.add(pid)
    return list(candidates_set)


def _repetition_penalty(node_id: str, path_history: Sequence[str], delta: float) -> float:
    """Return penalty for having visited node_id; quadratic in count to strongly discourage revisits."""
    count = sum(1 for x in path_history if x == node_id)
    return delta * (count * count)


def _backtrack_penalty(candidate_id: str, path_history: Sequence[str], penalty: float) -> float:
    """Return penalty for going back to the node we just came from (prevents oscillation).

    path_history[-1] is the current node; path_history[-2] is the node we
    arrived from. Penalize if the candidate equals [-2].
    """
    if penalty <= 0 or len(path_history) < 2:
        return 0.0
    if path_history[-2] == candidate_id:
        return penalty
    return 0.0


class IntentConditionedPolicy:
    """Policy that scores and selects next topic node given v_t, I_t, S_t.

    Score(k) = α·cos(μ_k, q) + β·cos(μ_k, μ_i) + γ·R_intent(i,k)
               + novelty_bonus·[k not in path] + prior_weight·prior(k)
               − δ·rep_penalty − backtrack_penalty
    """

    def __init__(
        self,
        graph: "Graph",
        *,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.5,
        delta: float = 0.5,
        backtrack_penalty: float = 5.0,
        novelty_bonus: float = 0.3,
        prior_weight: float = 0.2,
        clarify_similarity_threshold: Optional[float] = None,
    ) -> None:
        self._graph = graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.backtrack_penalty = backtrack_penalty
        self.novelty_bonus = novelty_bonus
        self.prior_weight = prior_weight
        self.clarify_similarity_threshold = clarify_similarity_threshold

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

        term_intent = self.gamma * r_intent(
            current_id,
            candidate_id,
            intent,
            self._graph,
            clarify_similarity_threshold=self.clarify_similarity_threshold,
        )

        rep = _repetition_penalty(candidate_id, state.path_history, self.delta)
        backtrack = _backtrack_penalty(candidate_id, state.path_history, self.backtrack_penalty)

        term_novelty = 0.0
        if self.novelty_bonus > 0 and candidate_id not in state.path_history:
            term_novelty = self.novelty_bonus

        term_prior = 0.0
        if self.prior_weight > 0 and state.prior:
            term_prior = self.prior_weight * state.prior.get(candidate_id, 0.0)

        return term_relevance + term_continuity + term_intent + term_novelty + term_prior - rep - backtrack

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
