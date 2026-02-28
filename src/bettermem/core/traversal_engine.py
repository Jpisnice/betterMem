from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .graph import Graph
from .navigation_policy import IntentConditionedPolicy, SemanticState
from .nodes import NodeId
from .transition_model import TransitionModel
from bettermem.retrieval.intent import TraversalIntent


Path = List[NodeId]


@dataclass
class TraversalResult:
    paths: List[Path]
    visit_counts: Dict[NodeId, int]


class TraversalEngine:
    """Implements intent-conditioned traversal over the topic graph.

    Primary entry point is intent_conditioned_navigate. Beam search, random walk,
    and personalized_pagerank are deprecated and kept for backward compatibility.
    """

    def __init__(
        self,
        graph: Graph,
        transition_model: TransitionModel,
    ) -> None:
        self._graph = graph
        self._transition_model = transition_model

    # ------------------------------------------------------------------
    # Intent-conditioned navigation (primary)
    # ------------------------------------------------------------------
    def intent_conditioned_navigate(
        self,
        start_nodes: Sequence[NodeId],
        steps: int,
        intent: TraversalIntent,
        semantic_state: SemanticState,
        *,
        policy: Optional[IntentConditionedPolicy] = None,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> TraversalResult:
        """Single policy traversal: at each step choose next topic by Score(k)."""
        if steps <= 0 or not start_nodes:
            return TraversalResult(paths=[], visit_counts={})

        if policy is None:
            policy = IntentConditionedPolicy(self._graph)

        current = start_nodes[0]
        path: Path = [current]
        visit_counts: Dict[NodeId, int] = {current: 1}

        for _ in range(steps - 1):
            state = SemanticState(
                query_embedding=semantic_state.query_embedding,
                path_history=list(path),
                prior=semantic_state.prior,
            )
            next_id = policy.step(
                current,
                intent,
                state,
                temperature=temperature,
                greedy=greedy,
            )
            if next_id is None:
                break
            path.append(next_id)
            visit_counts[next_id] = visit_counts.get(next_id, 0) + 1
            current = next_id

        return TraversalResult(paths=[path], visit_counts=visit_counts)

    # ------------------------------------------------------------------
    # Beam search (deprecated)
    # ------------------------------------------------------------------
    def beam_search(
        self,
        start_nodes: Sequence[NodeId],
        *,
        steps: int,
        beam_width: int,
    ) -> TraversalResult:
        """Beam search over paths using the second-order transition model.

        Deprecated: use intent_conditioned_navigate instead.
        """
        warnings.warn(
            "beam_search is deprecated; use intent_conditioned_navigate.",
            DeprecationWarning,
            stacklevel=2,
        )
        if steps <= 0:
            return TraversalResult(paths=[], visit_counts={})

        # Initialize with trivial paths of length 1
        beams: List[Tuple[Path, float]] = [([n], 0.0) for n in start_nodes]
        visit_counts: Dict[NodeId, int] = {}

        for _ in range(steps):
            new_beams: List[Tuple[Path, float]] = []
            for path, logp in beams:
                if len(path) < 2:
                    continue
                i, j = path[-2], path[-1]
                dist = self._transition_model.transition_prob(i, j).probs
                for k, p in dist.items():
                    if p <= 0.0:
                        continue
                    new_path = path + [k]
                    new_logp = logp + _safe_log(p)
                    new_beams.append((new_path, new_logp))

            if not new_beams:
                break

            # Keep top-k by log-probability
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        for path, _ in beams:
            for node in path:
                visit_counts[node] = visit_counts.get(node, 0) + 1

        return TraversalResult(paths=[p for p, _ in beams], visit_counts=visit_counts)

    # ------------------------------------------------------------------
    # Random walk
    # ------------------------------------------------------------------
    def random_walk(
        self,
        start_nodes: Sequence[NodeId],
        *,
        steps: int,
        exploration_factor: float = 0.0,
    ) -> TraversalResult:
        """Random walk driven by P(v_t | v_{t-1}, v_{t-2}).

        Deprecated: use intent_conditioned_navigate instead.
        """
        warnings.warn(
            "random_walk is deprecated; use intent_conditioned_navigate.",
            DeprecationWarning,
            stacklevel=2,
        )
        import random

        if len(start_nodes) < 2 or steps <= 0:
            return TraversalResult(paths=[], visit_counts={})

        path: Path = list(start_nodes[:2])
        visit_counts: Dict[NodeId, int] = {}
        for n in path:
            visit_counts[n] = visit_counts.get(n, 0) + 1

        for _ in range(steps):
            i, j = path[-2], path[-1]
            k = self._transition_model.sample_next(
                i,
                j,
                exploration_factor=exploration_factor,
                rng=random,
            )
            if k is None:
                break
            path.append(k)
            visit_counts[k] = visit_counts.get(k, 0) + 1

        return TraversalResult(paths=[path], visit_counts=visit_counts)

    # ------------------------------------------------------------------
    # Personalized PageRank (first-order over graph adjacency)
    # ------------------------------------------------------------------
    def personalized_pagerank(
        self,
        prior: Mapping[NodeId, float],
        *,
        alpha: float = 0.85,
        max_iters: int = 50,
        tol: float = 1e-6,
    ) -> Mapping[NodeId, float]:
        """Compute a personalized PageRank vector R.

        R = alpha * T * R + (1 - alpha) * P0

        where T is the transition matrix induced by the graph adjacency
        and P0 is the given prior over nodes.

        Deprecated: use intent_conditioned_navigate for goal-directed navigation.
        """
        warnings.warn(
            "personalized_pagerank is deprecated; use intent_conditioned_navigate.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not prior:
            return {}
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

        nodes = list(self._graph.nodes.keys())
        r: Dict[NodeId, float] = {n: prior.get(n, 0.0) for n in nodes}

        # Normalize prior
        total_prior = sum(prior.values())
        if total_prior <= 0.0:
            return {}
        p0 = {n: prior.get(n, 0.0) / total_prior for n in nodes}

        for _ in range(max_iters):
            new_r: Dict[NodeId, float] = {n: 0.0 for n in nodes}

            # Propagate mass according to outgoing adjacency
            for i in nodes:
                neighbors = self._graph.get_neighbors(i)
                if not neighbors:
                    continue
                out_sum = sum(neighbors.values())
                if out_sum <= 0.0:
                    continue
                for j, w in neighbors.items():
                    new_r[j] = new_r.get(j, 0.0) + alpha * r[i] * (w / out_sum)

            # Teleportation
            for n in nodes:
                new_r[n] = new_r.get(n, 0.0) + (1.0 - alpha) * p0.get(n, 0.0)

            # Check convergence
            delta = sum(abs(new_r[n] - r.get(n, 0.0)) for n in nodes)
            r = new_r
            if delta < tol:
                break

        return r


def _safe_log(x: float) -> float:
    import math

    if x <= 0.0:
        return -1e9
    return math.log(x)

