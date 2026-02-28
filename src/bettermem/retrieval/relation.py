"""Structural relation types and intent-based relation scoring R_intent(i, k)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from bettermem.core.edges import EdgeKind
from bettermem.core.nodes import NodeKind, TopicNode
from bettermem.retrieval.intent import TraversalIntent

if TYPE_CHECKING:
    from bettermem.core.graph import Graph


class RelationType(str, Enum):
    """Structural relation from current node i to candidate node k."""

    PARENT = "parent"  # k is parent of i (hierarchy)
    CHILD = "child"  # k is child of i (hierarchy)
    SIBLING = "sibling"  # same parent
    SEMANTIC_NEIGHBOR = "semantic_neighbor"  # TOPIC_TOPIC edge
    DISTANT_RELATED = "distant_related"  # other


def get_relation_type(graph: "Graph", source: str, target: str) -> RelationType:
    """Classify the structural relation from source node i to target node k.

    k may be an outgoing neighbor of i, or the parent of i (reverse hierarchy),
    or a sibling (same parent). Used for intent-conditioned scoring.
    """
    node_i = graph.get_node(source)
    node_k = graph.get_node(target)
    if node_i is None or node_k is None:
        return RelationType.DISTANT_RELATED

    # Parent: i has parent_id and it is k
    if isinstance(node_i, TopicNode) and node_i.parent_id is not None:
        if node_i.parent_id == target:
            return RelationType.PARENT

    # Child: edge (i, k) with TOPIC_SUBTOPIC
    kind_ik = graph.get_edge_kind(source, target)
    if kind_ik == EdgeKind.TOPIC_SUBTOPIC:
        return RelationType.CHILD

    # Semantic neighbor: topic-topic edge
    if kind_ik == EdgeKind.TOPIC_TOPIC:
        return RelationType.SEMANTIC_NEIGHBOR

    # Sibling: both topic nodes, same parent_id
    if isinstance(node_i, TopicNode) and isinstance(node_k, TopicNode):
        if (
            node_i.parent_id is not None
            and node_k.parent_id is not None
            and node_i.parent_id == node_k.parent_id
            and source != target
        ):
            return RelationType.SIBLING

    return RelationType.DISTANT_RELATED


def r_intent(source: str, target: str, intent: TraversalIntent, graph: "Graph") -> float:
    """Intent-relation score: +1 if (source, target) relation matches intent, else 0.

    R_intent(i, k) in the policy score. Used to bias navigation toward
    structure that matches user intent (e.g. deepen -> child, broaden -> parent).
    """
    if intent == TraversalIntent.NEUTRAL:
        return 0.0

    rel = get_relation_type(graph, source, target)

    if intent == TraversalIntent.DEEPEN and rel == RelationType.CHILD:
        return 1.0
    if intent == TraversalIntent.BROADEN and rel == RelationType.PARENT:
        return 1.0
    if intent == TraversalIntent.COMPARE and rel == RelationType.SIBLING:
        return 1.0
    if intent == TraversalIntent.APPLY and rel == RelationType.SEMANTIC_NEIGHBOR:
        return 1.0
    if intent == TraversalIntent.CLARIFY and rel == RelationType.SEMANTIC_NEIGHBOR:
        return 1.0
    # APPLY could also allow distant_related for "related domain"; optional
    if intent == TraversalIntent.APPLY and rel == RelationType.DISTANT_RELATED:
        return 0.5  # weaker bonus for distant

    return 0.0
