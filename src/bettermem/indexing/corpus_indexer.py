from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from bettermem.core.graph import Graph
from bettermem.core.nodes import ChunkNode, TopicNode, make_chunk_id, make_topic_id
from bettermem.core.transition_model import TransitionModel
from bettermem.indexing.chunker import BaseChunker, Chunk
from bettermem.topic_modeling.base import BaseTopicModel


class CorpusIndexer:
    """Orchestrates corpus chunking, topic assignment, graph building, and transitions."""

    def __init__(
        self,
        *,
        chunker: BaseChunker,
        topic_model: BaseTopicModel,
        graph: Graph,
        transition_model: TransitionModel,
    ) -> None:
        self._chunker = chunker
        self._topic_model = topic_model
        self._graph = graph
        self._transition_model = transition_model

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def transition_model(self) -> TransitionModel:
        return self._transition_model

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, corpus: Iterable[str]) -> None:
        """End-to-end indexing pipeline."""
        # 1. Chunk corpus
        chunks = self._chunker.chunk(corpus)

        # 2. Fit topic model and get topic distributions for chunks
        #    (fit on original corpus text for richer representations)
        texts = list(corpus)
        if texts:
            self._topic_model.fit(texts)

        chunk_texts = [c.text for c in chunks]
        topic_dists = self._topic_model.transform(chunk_texts)

        # 3. Create topic nodes and chunk nodes, and connect topic–chunk edges
        self._add_nodes_and_edges(chunks, topic_dists)

        # 4. Build topic sequences and fit transition model
        sequences = self._build_topic_sequences(chunks, topic_dists)
        self._transition_model.fit(sequences)

    def _add_nodes_and_edges(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
    ) -> None:
        # Collect topic IDs
        topic_ids = set()
        for dist in topic_dists:
            topic_ids.update(dist.keys())

        # Create TopicNodes
        for tid in sorted(topic_ids):
            node_id = make_topic_id(tid)
            topic_node = TopicNode(id=node_id, label=f"Topic {tid}")
            self._graph.add_node(topic_node)

        # Create ChunkNodes and connect topics to chunks
        for chunk, dist in zip(chunks, topic_dists):
            chunk_node = ChunkNode(
                id=make_chunk_id(len(self._graph.nodes)),
                document_id=chunk.document_id,
                position=chunk.position,
                metadata={"text": chunk.text},
            )
            self._graph.add_node(chunk_node)

            for tid, weight in dist.items():
                topic_node_id = make_topic_id(tid)
                # Topic-to-chunk edge weighted by assignment probability
                self._graph.add_edge(topic_node_id, chunk_node.id, weight=weight)

    def _build_topic_sequences(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
    ) -> List[Sequence[str]]:
        """Build sequences of topic node ids from chunk topic distributions."""
        # Group chunks by document and sort by position
        by_doc: dict[str, List[Tuple[Chunk, dict[int, float]]]] = {}
        for chunk, dist in zip(chunks, topic_dists):
            by_doc.setdefault(chunk.document_id, []).append((chunk, dist))

        sequences: List[Sequence[str]] = []
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda x: x[0].position)
            seq: List[str] = []
            for _chunk, dist in doc_chunks:
                if not dist:
                    continue
                # Use the most probable topic for the sequence
                top_topic = max(dist.items(), key=lambda kv: kv[1])[0]
                seq.append(make_topic_id(top_topic))
            if len(seq) >= 2:
                sequences.append(seq)
        return sequences

