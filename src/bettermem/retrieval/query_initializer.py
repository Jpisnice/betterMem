from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Tuple

from bettermem.core.nodes import NodeId
from bettermem.retrieval.intent import TraversalIntent

if TYPE_CHECKING:
    from bettermem.core.navigation_policy import SemanticState
from bettermem.indexing.keyword_mapper import KeywordToTopicMapper
from bettermem.topic_modeling.base import BaseTopicModel


class QueryInitializer:
    """Initialize query priors and starting state for traversal."""

    def __init__(
        self,
        topic_model: BaseTopicModel,
        keyword_mapper: KeywordToTopicMapper | None = None,
    ) -> None:
        self._topic_model = topic_model
        self._keyword_mapper = keyword_mapper

    def topic_prior(self, text: str) -> Mapping[NodeId, float]:
        """Compute P0 over topic nodes given a query.

        For hierarchical models, prior is over topic nodes (path IDs);
        topic_id is the same as node_id.
        """
        model_dist = self._topic_model.get_topic_distribution_for_query(text)
        prior: dict[NodeId, float] = {}
        for tid, p in model_dist.items():
            if p <= 0.0:
                continue
            prior[tid] = prior.get(tid, 0.0) + float(p)

        if self._keyword_mapper is not None:
            kw_dist = self._keyword_mapper.topic_prior_from_query(text)
            for node_id, p in kw_dist.items():
                prior[node_id] = prior.get(node_id, 0.0) + float(p)

        total = sum(prior.values())
        if total <= 0.0:
            return {}
        inv_total = 1.0 / total
        return {k: v * inv_total for k, v in prior.items()}

    def initial_state(
        self, text: str, intent: Optional[TraversalIntent] = None
    ) -> Tuple[tuple[NodeId, NodeId] | None, Mapping[NodeId, float]]:
        """Choose (v_{t-2}, v_{t-1}) and prior. When intent is set, use intent-aware start and rolled prior."""
        prior = self.topic_prior(text)
        if not prior:
            return None, prior

        rollup = getattr(
            self._topic_model, "rollup_leaf_prior_to_ancestors", None
        )
        if callable(rollup) and intent is not None and intent != TraversalIntent.NEUTRAL:
            prior_rolled = rollup(prior)
            if prior_rolled:
                prior = prior_rolled

        if intent is None or intent == TraversalIntent.NEUTRAL:
            if len(prior) < 2:
                return None, prior
            sorted_topics = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
            v_t2, _ = sorted_topics[0]
            v_t1, _ = sorted_topics[1]
            return (v_t2, v_t1), prior

        hierarchy = getattr(self._topic_model, "get_hierarchy", None)
        get_leaves = getattr(self._topic_model, "get_leaf_topic_ids", None)
        leaf_ids = set(get_leaves()) if callable(get_leaves) else set()
        has_children = set()
        if callable(hierarchy):
            has_children = set(hierarchy())

        sorted_prior = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
        if not sorted_prior:
            return None, prior

        if intent == TraversalIntent.DEEPEN:
            non_leaves_with_children = [
                (tid, p) for tid, p in sorted_prior if tid in has_children
            ]
            if non_leaves_with_children:
                start = non_leaves_with_children[0][0]
            else:
                start = sorted_prior[0][0]
        elif intent == TraversalIntent.BROADEN:
            start = sorted_prior[0][0]
        elif intent == TraversalIntent.COMPARE:
            leaf_prior_only = [(tid, p) for tid, p in sorted_prior if tid in leaf_ids]
            if leaf_prior_only:
                start = leaf_prior_only[0][0]
            else:
                start = sorted_prior[0][0]
        elif intent in (TraversalIntent.CLARIFY, TraversalIntent.APPLY):
            start = sorted_prior[0][0]
        else:
            start = sorted_prior[0][0]

        return (start, start), prior

    def semantic_state(
        self, text: str, prior: Optional[Mapping[NodeId, float]] = None
    ) -> "SemanticState":
        """Build semantic state S_t for intent-conditioned navigation: query embedding + prior."""
        from bettermem.core.navigation_policy import SemanticState

        if prior is None:
            prior = self.topic_prior(text)
        query_embedding: list | None = None
        if hasattr(self._topic_model, "embed_query"):
            q = self._topic_model.embed_query(text)
            if q is not None:
                query_embedding = list(q)
        return SemanticState(
            query_embedding=query_embedding,
            path_history=[],
            prior=dict(prior) if prior is not None else {},
        )

