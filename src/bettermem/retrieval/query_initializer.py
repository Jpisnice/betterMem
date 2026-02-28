from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Tuple

from bettermem.core.nodes import NodeId, make_subtopic_id, make_topic_id

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

    def _topic_id_to_node_id(self, tid: int) -> NodeId:
        """Map numeric topic id to graph node id (flat or hierarchical)."""
        if callable(getattr(self._topic_model, "get_hierarchy", None)):
            hierarchy = self._topic_model.get_hierarchy()
            if hierarchy:
                return make_subtopic_id(tid // 100, tid % 100)
        return make_topic_id(tid)

    def topic_prior(self, text: str) -> Mapping[NodeId, float]:
        """Compute P0 over topic nodes given a query.

        For hierarchical models, prior is over sub-topic nodes (coarse prior
        distributed to sub-topics via the model's P(sub | coarse, query)).
        """
        model_dist = self._topic_model.get_topic_distribution_for_query(text)
        prior: dict[NodeId, float] = {}
        for tid, p in model_dist.items():
            if p <= 0.0:
                continue
            node_id = self._topic_id_to_node_id(tid)
            prior[node_id] = prior.get(node_id, 0.0) + float(p)

        if self._keyword_mapper is not None:
            kw_dist = self._keyword_mapper.topic_prior_from_query(text)
            for tid, p in kw_dist.items():
                node_id = self._topic_id_to_node_id(tid)
                prior[node_id] = prior.get(node_id, 0.0) + float(p)

        total = sum(prior.values())
        if total <= 0.0:
            return {}
        inv_total = 1.0 / total
        return {k: v * inv_total for k, v in prior.items()}

    def initial_state(self, text: str) -> Tuple[tuple[NodeId, NodeId] | None, Mapping[NodeId, float]]:
        """Choose (v_{t-2}, v_{t-1}) from the top-2 topics."""
        prior = self.topic_prior(text)
        if len(prior) < 2:
            return None, prior

        sorted_topics = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
        v_t2, _ = sorted_topics[0]
        v_t1, _ = sorted_topics[1]
        return (v_t2, v_t1), prior

    def semantic_state(self, text: str) -> "SemanticState":
        """Build semantic state S_t for intent-conditioned navigation: query embedding + prior."""
        from bettermem.core.navigation_policy import SemanticState

        prior = self.topic_prior(text)
        query_embedding: list | None = None
        if hasattr(self._topic_model, "embed_query"):
            q = self._topic_model.embed_query(text)
            if q is not None:
                query_embedding = list(q)
        return SemanticState(
            query_embedding=query_embedding,
            path_history=[],
            prior=prior,
        )

