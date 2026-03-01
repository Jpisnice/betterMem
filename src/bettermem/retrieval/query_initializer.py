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
        """Choose (v_{t-2}, v_{t-1}) and prior.

        Start node is chosen from the *leaf* prior (before rollup) so that
        each intent begins at the depth where it has room to navigate:
        - DEEPEN: parent of best leaf (room to descend into children)
        - BROADEN: best leaf (room to ascend to parent)
        - COMPARE: best leaf (room to move to siblings)
        - APPLY / CLARIFY: best leaf (follow semantic edges)

        The full prior passed to the policy is rolled up to ancestors so the
        walk can score non-leaf topics too.
        """
        leaf_prior = self.topic_prior(text)
        if not leaf_prior:
            return None, leaf_prior

        rollup = getattr(
            self._topic_model, "rollup_leaf_prior_to_ancestors", None
        )

        if intent is None or intent == TraversalIntent.NEUTRAL:
            if len(leaf_prior) < 2:
                return None, leaf_prior
            sorted_topics = sorted(leaf_prior.items(), key=lambda kv: kv[1], reverse=True)
            v_t2, _ = sorted_topics[0]
            v_t1, _ = sorted_topics[1]
            return (v_t2, v_t1), leaf_prior

        prior = leaf_prior
        if callable(rollup):
            prior_rolled = rollup(leaf_prior)
            if prior_rolled:
                prior = prior_rolled

        hierarchy = getattr(self._topic_model, "get_hierarchy", None)
        get_parents_fn = getattr(self._topic_model, "get_parents", None)
        get_leaves = getattr(self._topic_model, "get_leaf_topic_ids", None)
        leaf_ids = set(get_leaves()) if callable(get_leaves) else set()
        has_children = set()
        if callable(hierarchy):
            has_children = set(hierarchy())

        sorted_leaf = sorted(leaf_prior.items(), key=lambda kv: kv[1], reverse=True)
        if not sorted_leaf:
            return None, prior

        best_leaf = sorted_leaf[0][0]

        if intent == TraversalIntent.DEEPEN:
            if callable(get_parents_fn):
                parents = get_parents_fn(best_leaf)
                mid_level = [p for p in parents if p in has_children]
                if mid_level:
                    start = mid_level[0]
                elif parents:
                    start = parents[0]
                else:
                    start = best_leaf
            else:
                start = best_leaf

        elif intent == TraversalIntent.BROADEN:
            start = best_leaf

        elif intent == TraversalIntent.COMPARE:
            start = best_leaf

        elif intent in (TraversalIntent.CLARIFY, TraversalIntent.APPLY):
            start = best_leaf

        else:
            start = best_leaf

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

