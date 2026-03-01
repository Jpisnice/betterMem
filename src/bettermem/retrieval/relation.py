"""Structural relation types and intent-based relation scoring R_intent(i, k)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

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

    Hierarchy relations (parent, child, sibling) take priority over edge-kind
    classification.  Two siblings that also share a TOPIC_TOPIC edge must be
    classified as SIBLING so that COMPARE intent can find them.
    """
    node_i = graph.get_node(source)
    node_k = graph.get_node(target)
    if node_i is None or node_k is None:
        return RelationType.DISTANT_RELATED

    # --- structural hierarchy (checked first) ---

    parents = graph.get_parents(source)
    if target in parents:
        return RelationType.PARENT
    if isinstance(node_i, TopicNode) and target in node_i.parent_ids:
        return RelationType.PARENT

    children = graph.get_children(source)
    if target in children:
        return RelationType.CHILD

    siblings = graph.get_siblings(source)
    if target in siblings:
        return RelationType.SIBLING
    if isinstance(node_i, TopicNode) and isinstance(node_k, TopicNode) and source != target:
        if set(node_i.parent_ids) & set(node_k.parent_ids):
            return RelationType.SIBLING

    # --- edge-kind fallback ---

    kind_ik = graph.get_edge_kind(source, target)
    if kind_ik == EdgeKind.TOPIC_SUBTOPIC:
        return RelationType.CHILD
    if kind_ik in (EdgeKind.TOPIC_TOPIC, EdgeKind.TOPIC_RELATED):
        return RelationType.SEMANTIC_NEIGHBOR

    return RelationType.DISTANT_RELATED


def r_intent(
    source: str,
    target: str,
    intent: TraversalIntent,
    graph: "Graph",
    *,
    clarify_similarity_threshold: Optional[float] = None,
) -> float:
    """Intent-relation score: +1 if (source, target) relation matches intent, else 0.

    R_intent(i, k) in the policy score. For CLARIFY, only scores 1.0 when
    semantic neighbor edge weight >= clarify_similarity_threshold (if set).
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
        if clarify_similarity_threshold is not None:
            weight = graph.get_neighbors(source).get(target)
            if weight is None or weight < clarify_similarity_threshold:
                return 0.0
        return 1.0
    # APPLY could also allow distant_related for "related domain"; optional
    if intent == TraversalIntent.APPLY and rel == RelationType.DISTANT_RELATED:
        return 0.5  # weaker bonus for distant

    return 0.0
