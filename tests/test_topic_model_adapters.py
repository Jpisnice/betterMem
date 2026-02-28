from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import pytest

from bettermem.topic_modeling.lda_adapter import LDAAdapter


class DummyDictionary:
    """Minimal stub matching the gensim Dictionary API we rely on."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def doc2bow(self, tokens: Iterable[str]):
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        bow = []
        for term, count in counts.items():
            if term not in self._vocab:
                self._vocab[term] = len(self._vocab)
            bow.append((self._vocab[term], count))
        return bow


class DummyLdaModel:
    """Minimal stub matching the gensim LDA API used by LDAAdapter."""

    def get_document_topics(self, bow) -> Sequence[tuple[int, float]]:  # type: ignore[override]
        # Return a tiny fixed distribution irrespective of input
        return [(0, 0.7), (1, 0.3)]

    def show_topic(self, topic_id: int, topn: int = 10) -> List[tuple[str, float]]:
        return [(f"kw{topic_id}", 1.0) for _ in range(topn)]


def test_lda_adapter_transform_and_query_distribution() -> None:
    dictionary = DummyDictionary()
    lda = DummyLdaModel()
    adapter = LDAAdapter(dictionary=dictionary, lda_model=lda)

    corpus = ["alpha beta", "gamma delta"]
    adapter.fit(corpus)  # no-op but should not fail

    dists = adapter.transform(["chunk one", "chunk two"])
    assert len(dists) == 2
    for dist in dists:
        assert set(dist.keys()) == {0, 1}

    q_dist = adapter.get_topic_distribution_for_query("alpha")
    assert set(q_dist.keys()) == {0, 1}


@pytest.mark.skipif(
    pytest.importorskip("bertopic", reason="BERTopic not installed") is None,  # type: ignore[comparison-overlap]
    reason="BERTopic not available",
)
def test_bertopic_adapter_basic() -> None:
    """Smoke test for BERTopicAdapter when bertopic is installed.

    This test is skipped automatically if the optional dependency
    is not available in the environment.
    """
    from bettermem.topic_modeling.bertopic_adapter import BERTopicAdapter

    adapter = BERTopicAdapter()
    docs = ["short document", "another short document"]
    adapter.fit(docs)

    dists = adapter.transform(["query text"])
    assert len(dists) == 1
    assert dists[0]  # non-empty distribution

