"""Tests for intent classification, relation scoring, and intent-conditioned navigation."""

from __future__ import annotations

import pytest

from bettermem.core.graph import Graph
from bettermem.core.navigation_policy import IntentConditionedPolicy, SemanticState
from bettermem.core.nodes import NodeKind, TopicNode
from bettermem.core.traversal_engine import TraversalEngine
from bettermem.core.transition_model import TransitionModel
from bettermem.retrieval.intent import TraversalIntent, classify_intent_heuristic
from bettermem.retrieval.relation import RelationType, get_relation_type, r_intent


def test_classify_intent_heuristic_deepen() -> None:
    assert classify_intent_heuristic("explain more about this") == TraversalIntent.DEEPEN
    assert classify_intent_heuristic("tell me more") == TraversalIntent.DEEPEN


def test_classify_intent_heuristic_broaden() -> None:
    assert classify_intent_heuristic("what is the big picture") == TraversalIntent.BROADEN
    assert classify_intent_heuristic("give me an overview") == TraversalIntent.BROADEN


def test_classify_intent_heuristic_compare() -> None:
    assert classify_intent_heuristic("compare these approaches") == TraversalIntent.COMPARE
    assert classify_intent_heuristic("how is this different") == TraversalIntent.COMPARE


def test_classify_intent_heuristic_neutral() -> None:
    assert classify_intent_heuristic("transformer attention") == TraversalIntent.NEUTRAL
    assert classify_intent_heuristic("") == TraversalIntent.NEUTRAL


def test_relation_type_parent_child() -> None:
    from bettermem.core.edges import EdgeKind

    graph = Graph()
    graph.add_node(TopicNode(id="t:0", label="Coarse", level=0))
    graph.add_node(TopicNode(id="t:0:0", label="Sub", level=1, parent_id="t:0"))
    graph.add_edge("t:0", "t:0:0", weight=1.0, kind=EdgeKind.TOPIC_SUBTOPIC)

    assert get_relation_type(graph, "t:0", "t:0:0") == RelationType.CHILD
    assert get_relation_type(graph, "t:0:0", "t:0") == RelationType.PARENT


def test_r_intent_deepen_child() -> None:
    from bettermem.core.edges import EdgeKind

    graph = Graph()
    graph.add_node(TopicNode(id="t:0", label="Coarse", level=0))
    graph.add_node(TopicNode(id="t:0:0", label="Sub", level=1, parent_id="t:0"))
    graph.add_edge("t:0", "t:0:0", weight=1.0, kind=EdgeKind.TOPIC_SUBTOPIC)

    assert r_intent("t:0", "t:0:0", TraversalIntent.DEEPEN, graph) == 1.0
    assert r_intent("t:0:0", "t:0", TraversalIntent.BROADEN, graph) == 1.0


def test_policy_next_distribution_empty_candidates() -> None:
    graph = Graph()
    graph.add_node(TopicNode(id="alone", label="Alone", level=0))
    policy = IntentConditionedPolicy(graph)
    state = SemanticState(path_history=[], prior={})
    dist = policy.next_distribution("alone", TraversalIntent.NEUTRAL, state)
    assert dist == {}


def test_policy_step_returns_none_when_no_candidates() -> None:
    graph = Graph()
    graph.add_node(TopicNode(id="alone", label="Alone", level=0))
    policy = IntentConditionedPolicy(graph)
    state = SemanticState(path_history=[], prior={})
    next_id = policy.step("alone", TraversalIntent.NEUTRAL, state, greedy=True)
    assert next_id is None


def test_intent_conditioned_navigate_single_step() -> None:
    from bettermem.core.edges import EdgeKind

    graph = Graph()
    graph.add_node(TopicNode(id="t:0", label="A", level=0))
    graph.add_node(TopicNode(id="t:1", label="B", level=0))
    graph.add_edge("t:0", "t:1", weight=1.0, kind=EdgeKind.TOPIC_TOPIC)

    tm = TransitionModel()
    tm.fit([["t:0", "t:1"]])
    engine = TraversalEngine(graph=graph, transition_model=tm)
    state = SemanticState(path_history=[], prior={"t:0": 0.5, "t:1": 0.5})

    result = engine.intent_conditioned_navigate(
        start_nodes=["t:0"],
        steps=2,
        intent=TraversalIntent.NEUTRAL,
        semantic_state=state,
    )
    assert len(result.paths) == 1
    assert len(result.paths[0]) >= 1
    assert result.visit_counts
