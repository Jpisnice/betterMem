from __future__ import annotations

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


class TraversalEngine:
    """Intent-conditioned semantic hierarchical traversal over the topic graph.

    Single entry point: intent_conditioned_navigate.
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

