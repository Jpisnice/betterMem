from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from bettermem.core.graph import Graph
from bettermem.core.nodes import ChunkNode, Node, NodeKind, make_chunk_id, make_topic_id
from bettermem.indexing.chunker import FixedWindowChunker
from bettermem.indexing.corpus_indexer import CorpusIndexer
from bettermem.indexing.keyword_mapper import KeywordToTopicMapper
from bettermem.learning.smoothing import additive_smoothing
from bettermem.retrieval.context_aggregator import ContextAggregator
from bettermem.retrieval.query_initializer import QueryInitializer
from bettermem.retrieval.scorer import QueryScorer
from bettermem.topic_modeling.base import BaseTopicModel


class DummyTopicModel(BaseTopicModel):
    """Minimal semantic-hierarchical topic model for testing."""

    def __init__(self) -> None:
        self._fit_called = False

    def fit(self, documents: Iterable[str]) -> None:
        _ = list(documents)
        self._fit_called = True

    def get_hierarchy(self) -> Mapping[int, List[int]]:
        # One coarse topic with one subtopic (encoded 0 = 0*100+0)
        return {0: [0]}

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        return [{0: 1.0} for _ in chunks]

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        return [f"kw{topic_id}"]

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        base = {0: 0.6, 100: 0.4} if len(text.split()) > 1 else {0: 1.0}
        return additive_smoothing({k: int(v * 10) for k, v in base.items()}, alpha=0.0)


def test_fixed_window_chunker_basic() -> None:
    chunker = FixedWindowChunker(window_size=3, overlap=1)
    corpus = ["one two three four five"]

    chunks = chunker.chunk(corpus)
    # Expect more than one overlapping chunk
    assert len(chunks) >= 2
    assert chunks[0].document_id == "0"
    assert chunks[0].position == 0


def test_corpus_indexer_builds_graph_and_transitions() -> None:
    graph = Graph()
    tm = DummyTransitionModel()
    chunker = FixedWindowChunker(window_size=3, overlap=1)
    topic_model = DummyTopicModel()

    indexer = CorpusIndexer(
        chunker=chunker,
        topic_model=topic_model,
        graph=graph,
        transition_model=tm,
    )

    corpus = ["alpha beta gamma", "delta epsilon zeta"]
    indexer.build_index(corpus)

    # Should have created at least one topic node and some chunk nodes
    assert graph.count_nodes(NodeKind.TOPIC) >= 1
    assert graph.count_nodes(NodeKind.CHUNK) >= 1

    # Transition model should have been fit with at least one sequence
    assert tm.fit_called


class DummyTransitionModel:
    """Lightweight stand-in to observe CorpusIndexer behaviour."""

    def __init__(self) -> None:
        self.fit_called = False
        self.sequences: list[Sequence[str]] = []

    def fit(
        self,
        sequences: Iterable[Sequence[str]],
        *,
        structural_groups_per_seq: Iterable[Sequence[int]] | None = None,
        **kwargs: object,
    ) -> None:
        self.fit_called = True
        self.sequences = list(sequences)


def test_context_aggregator_diversity_and_top_k() -> None:
    graph = Graph()
    # Two documents, multiple chunks each
    chunks: list[ChunkNode] = []
    for doc_id in ("doc1", "doc2"):
        for pos in range(3):
            node = ChunkNode(
                id=make_chunk_id(len(chunks)),
                document_id=doc_id,
                position=pos,
                metadata={"text": f"{doc_id}-{pos}"},
            )
            graph.add_node(node)
            chunks.append(node)

    scores = {c.id: float(i + 1) for i, c in enumerate(chunks)}
    aggregator = ContextAggregator(graph=graph)

    top_k = 4
    diverse = aggregator.select(scores, top_k=top_k, diversity=True)
    assert len(diverse) <= top_k
    # With diversity, we should see chunks from both documents
    doc_ids = {c.document_id for c in diverse}
    assert doc_ids == {"doc1", "doc2"}

    greedy = aggregator.select(scores, top_k=top_k, diversity=False)
    assert len(greedy) == top_k


def test_keyword_to_topic_mapper_prior() -> None:
    mapper = KeywordToTopicMapper()
    mapper.fit({0: ["python", "ai"], 1: ["java"]})

    prior = mapper.topic_prior_from_query("python ai")
    assert prior
    # All mass should be on topic 0 for this toy setup
    assert set(prior.keys()) == {0}


def test_query_initializer_topic_prior_and_initial_state() -> None:
    topic_model = DummyTopicModel()
    mapper = KeywordToTopicMapper()
    mapper.fit({0: ["alpha"], 1: ["beta"]})
    initializer = QueryInitializer(topic_model=topic_model, keyword_mapper=mapper)

    pair, prior = initializer.initial_state("alpha beta")
    assert prior
    assert pair is not None


def test_query_scorer_add_visit_counts() -> None:
    scorer = QueryScorer()
    scorer.add_visit_counts({"n1": 2, "n2": 3})

    scores = scorer.scores()
    assert scores["n1"] == 2.0
    assert scores["n2"] == 3.0

