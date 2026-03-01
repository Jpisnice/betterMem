from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

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


def _blend_and_step(
    policy_dist: Mapping[NodeId, float],
    markov_dist: Mapping[NodeId, float],
    eta: float,
    temperature: float,
    greedy: bool,
    rng: random.Random,
) -> Optional[NodeId]:
    """Blend P_policy and P_markov, then sample or argmax. Returns None if both empty."""
    if not policy_dist and not markov_dist:
        return None
    keys = set(policy_dist.keys()) | set(markov_dist.keys())
    if not keys:
        return None
    blended: Dict[NodeId, float] = {}
    for k in keys:
        p_p = policy_dist.get(k, 0.0)
        p_m = markov_dist.get(k, 0.0)
        blended[k] = eta * p_p + (1.0 - eta) * p_m
    total = sum(blended.values())
    if total <= 0:
        return None
    inv = 1.0 / total
    blended = {k: v * inv for k, v in blended.items()}
    if greedy:
        return max(blended.items(), key=lambda x: x[1])[0]
    r = rng.random()
    cum = 0.0
    for k, p in blended.items():
        cum += p
        if r <= cum:
            return k
    return list(blended.keys())[-1]


class TraversalEngine:
    """Intent-conditioned semantic hierarchical traversal over the topic graph.

    Single entry point: intent_conditioned_navigate. When transition_policy_mix_eta < 1,
    blends policy distribution with Markov transition distribution.
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
        transition_policy_mix_eta: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> TraversalResult:
        """Traversal: blend policy and Markov when eta < 1; step 1 uses policy only."""
        if steps <= 0 or not start_nodes:
            return TraversalResult(paths=[], visit_counts={})

        if policy is None:
            policy = IntentConditionedPolicy(self._graph)
        rng = rng or random.Random()

        current = start_nodes[0]
        path: Path = [current]
        visit_counts: Dict[NodeId, int] = {current: 1}
        prev: Optional[NodeId] = None

        for _ in range(steps - 1):
            state = SemanticState(
                query_embedding=semantic_state.query_embedding,
                path_history=list(path),
                prior=semantic_state.prior,
            )
            policy_dist = policy.next_distribution(current, intent, state, temperature=temperature)
            if not policy_dist:
                break

            if transition_policy_mix_eta >= 1.0 or prev is None:
                next_id = policy.step(
                    current,
                    intent,
                    state,
                    temperature=temperature,
                    greedy=greedy,
                    rng=rng,
                )
            else:
                markov_d = self._transition_model.transition_prob(prev, current).probs
                next_id = _blend_and_step(
                    policy_dist,
                    markov_d,
                    eta=transition_policy_mix_eta,
                    temperature=temperature,
                    greedy=greedy,
                    rng=rng,
                )
            if next_id is None:
                break
            path.append(next_id)
            visit_counts[next_id] = visit_counts.get(next_id, 0) + 1
            prev = current
            current = next_id

        return TraversalResult(paths=[path], visit_counts=visit_counts)

