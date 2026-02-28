from __future__ import annotations

from typing import Mapping, Tuple

from bettermem.core.nodes import NodeId, make_topic_id
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
        """Compute P0 over topic nodes given a query."""
        model_dist = self._topic_model.get_topic_distribution_for_query(text)
        prior: dict[NodeId, float] = {
            make_topic_id(tid): float(p) for tid, p in model_dist.items() if p > 0.0
        }

        if self._keyword_mapper is not None:
            kw_dist = self._keyword_mapper.topic_prior_from_query(text)
            for tid, p in kw_dist.items():
                node_id = make_topic_id(tid)
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

