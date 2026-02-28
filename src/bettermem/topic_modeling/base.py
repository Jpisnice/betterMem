from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Optional, Sequence


class BaseTopicModel(ABC):
    """Abstract interface for topic models used by BetterMem.

    Implementations wrap concrete backends such as BERTopic or LDA and
    expose a unified API in terms of topic IDs and probability
    distributions. Optional get_centroid and embed_query enable
    intent-conditioned navigation for embedding-capable backends.
    """

    @abstractmethod
    def fit(self, documents: Iterable[str]) -> None:
        """Fit the topic model on the given corpus."""

    @abstractmethod
    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        """Return topic distributions for each chunk.

        Each element in the returned sequence is a mapping from topic id
        (int) to probability (float). Probabilities for each chunk should
        sum approximately to 1.
        """

    @abstractmethod
    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        """Return representative keywords for a topic."""

    @abstractmethod
    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        """Return a topic probability distribution for a query."""

    def get_centroid(self, topic_id: int) -> Optional[Sequence[float]]:
        """Return the embedding-space centroid for a topic, if available.

        Used by intent-conditioned navigation for semantic scoring.
        Default returns None (non-embedding backends).
        """
        return None

    def embed_query(self, text: str) -> Optional[Sequence[float]]:
        """Return the query embedding vector, if available.

        Used by intent-conditioned navigation for relevance term cos(μ_k, q).
        Default returns None (non-embedding backends).
        """
        return None

    def get_coarse_centroid(self, coarse_id: int) -> Optional[Sequence[float]]:
        """Return the embedding centroid for a coarse (parent) topic, if available.

        Used when storing centroids on coarse-level topic nodes in hierarchical models.
        Default returns None.
        """
        return None

