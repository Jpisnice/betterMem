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


def test_traversal_engine_random_walk_and_pagerank() -> None:
    # Build a tiny chain graph n1 -> n2 -> n3
    graph = Graph()
    for nid in ("n1", "n2", "n3"):
        graph.add_node(Node(id=nid, kind=NodeKind.TOPIC))
    graph.add_edge("n1", "n2", weight=1.0)
    graph.add_edge("n2", "n3", weight=1.0)

    tm = TransitionModel()
    tm.fit([["n1", "n2", "n3"]])
    engine = TraversalEngine(graph=graph, transition_model=tm)

    # Random walk should at least visit all nodes in the simple chain
    result = engine.random_walk(["n1", "n2"], steps=5, exploration_factor=0.0)
    assert set(result.visit_counts.keys()) >= {"n1", "n2", "n3"}

    # Personalized PageRank should produce scores for all nodes with non-zero prior
    prior = {"n1": 0.5, "n2": 0.5}
    scores = engine.personalized_pagerank(prior, max_iters=20)
    assert set(scores.keys()) == {"n1", "n2", "n3"}
    assert all(v >= 0.0 for v in scores.values())
    assert not math.isclose(sum(scores.values()), 0.0)


def test_traversal_engine_invalid_alpha_raises() -> None:
    graph = Graph()
    graph.add_node(Node(id="n1", kind=NodeKind.TOPIC))
    tm = TransitionModel()
    engine = TraversalEngine(graph=graph, transition_model=tm)

    with pytest.raises(ValueError):
        engine.personalized_pagerank({"n1": 1.0}, alpha=0.0)

