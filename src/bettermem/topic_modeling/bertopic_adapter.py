from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from .base import BaseTopicModel


class BERTopicAdapter(BaseTopicModel):
    """Adapter around BERTopic exposing the BaseTopicModel interface.

    This class assumes BERTopic is installed in the environment. If it is
    not available, instantiation will raise an ImportError.
    """

    def __init__(self, *args, **kwargs) -> None:
        try:
            from bertopic import BERTopic  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "BERTopicAdapter requires the 'bertopic' extra to be installed."
            ) from exc

        self._bertopic_cls = BERTopic
        self._model = BERTopic(*args, **kwargs)

    def fit(self, documents: Iterable[str]) -> None:
        self._model.fit(list(documents))

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        texts = list(chunks)
        topics, probs = self._model.transform(texts)
        # BERTopic returns a 2D array of probabilities
        distributions: List[Mapping[int, float]] = []
        if probs is None:
            for topic in topics:
                distributions.append({int(topic): 1.0})
        else:
            for row in probs:
                dist: Mapping[int, float] = {
                    int(idx): float(p) for idx, p in enumerate(row) if p > 0.0
                }
                distributions.append(dist)
        return distributions

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        tuples = self._model.get_topic(topic_id) or []
        return [term for term, _ in tuples[:top_k]]

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        _, probs = self._model.transform([text])
        if probs is None:
            return {}
        row = probs[0]
        return {int(idx): float(p) for idx, p in enumerate(row) if p > 0.0}

