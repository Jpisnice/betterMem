from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from .base import BaseTopicModel


class LDAAdapter(BaseTopicModel):
    """Adapter around a Gensim LDA model exposing BaseTopicModel.

    This implementation expects a preconfigured Gensim dictionary and LDA
    model instance to be provided, so BetterMem does not own tokenization
    or preprocessing choices.
    """

    def __init__(self, dictionary, lda_model) -> None:  # type: ignore[no-untyped-def]
        self._dict = dictionary
        self._lda = lda_model

    def fit(self, documents: Iterable[str]) -> None:
        # For simplicity, assume the underlying LDA model is already trained.
        # If training is required, it should be done externally and passed in.
        _ = list(documents)

    def _text_to_bow(self, text: str):
        tokens = text.split()
        return self._dict.doc2bow(tokens)

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        dists: List[Mapping[int, float]] = []
        for text in chunks:
            bow = self._text_to_bow(text)
            topic_dist = self._lda.get_document_topics(bow)
            dists.append({int(tid): float(p) for tid, p in topic_dist})
        return dists

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        terms = self._lda.show_topic(topic_id, topn=top_k)
        return [term for term, _ in terms]

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        bow = self._text_to_bow(text)
        topic_dist = self._lda.get_document_topics(bow)
        return {int(tid): float(p) for tid, p in topic_dist}

