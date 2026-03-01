from __future__ import annotations

import math

import pytest

from bettermem.core.graph import Graph
from bettermem.core.nodes import Node, NodeKind
from bettermem.core.transition_model import TransitionModel
from bettermem.core.traversal_engine import TraversalEngine


def test_transition_model_fit_and_distribution() -> None:
    model = TransitionModel(smoothing_lambda=0.5)

    # Two simple sequences over three nodes
    seq1 = ["a", "b", "c", "c"]
    seq2 = ["a", "b", "d"]
    model.fit([seq1, seq2])

    dist = model.transition_prob("a", "b").probs
    # There should be probability mass on successors "c" and "d"
    assert set(dist.keys()) >= {"c", "d"}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_transition_model_sampling_respects_support() -> None:
    model = TransitionModel(smoothing_lambda=0.5)
    model.fit([["x", "y", "z"]])

    # With zero exploration, samples should always be from observed support
    samples = {model.sample_next("x", "y", exploration_factor=0.0) for _ in range(10)}
    assert samples == {"z"}


def test_transition_model_to_from_dict_roundtrip() -> None:
    model = TransitionModel(smoothing_lambda=0.3)
    model.fit([["n1", "n2", "n3"], ["n2", "n3", "n4"]])

    payload = model.to_dict()
    restored = TransitionModel.from_dict(payload)

    original = model.transition_prob("n1", "n2").probs
    cloned = restored.transition_prob("n1", "n2").probs
    assert original == cloned


def test_traversal_engine_intent_conditioned() -> None:
    from bettermem.core.nodes import TopicNode
    from bettermem.core.navigation_policy import SemanticState
    from bettermem.retrieval.intent import TraversalIntent

    graph = Graph()
    for nid in ("t:0:0", "t:0:1", "t:1:0"):
        graph.add_node(
            TopicNode(
                id=nid,
                label=nid,
                level=1,
                parent_id="t:0" if nid.startswith("t:0") else "t:1",
                metadata={"centroid": [1.0, 0.0] if nid == "t:0:0" else [0.9, 0.1]},
            )
        )
    graph.add_edge("t:0:0", "t:0:1", weight=1.0)
    graph.add_edge("t:0:0", "t:1:0", weight=0.5)
    graph.add_edge("t:0:1", "t:1:0", weight=0.5)

    tm = TransitionModel()
    tm.fit([["t:0:0", "t:0:1", "t:1:0"]])
    engine = TraversalEngine(graph=graph, transition_model=tm)

    state = SemanticState(
        query_embedding=[1.0, 0.0],
        path_history=[],
        prior={"t:0:0": 0.7, "t:0:1": 0.2, "t:1:0": 0.1},
    )
    result = engine.intent_conditioned_navigate(
        start_nodes=["t:0:0"],
        steps=3,
        intent=TraversalIntent.NEUTRAL,
        semantic_state=state,
        greedy=True,
    )
    assert len(result.paths) == 1
    assert len(result.paths[0]) >= 1
    assert result.paths[0][0] == "t:0:0"
    assert set(result.visit_counts.keys()) >= {"t:0:0"}

