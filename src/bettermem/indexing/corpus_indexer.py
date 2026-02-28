from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from bettermem.core.edges import EdgeKind
from bettermem.core.graph import Graph
from bettermem.core.nodes import (
    ChunkNode,
    NodeKind,
    TopicNode,
    get_topic_centroid,
    make_chunk_id,
    make_subtopic_id,
    make_topic_id,
)
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
        sequences, structural_groups_per_seq = self._build_topic_sequences(chunks, topic_dists)
        self._transition_model.fit(sequences, structural_groups_per_seq=structural_groups_per_seq)

    def _add_nodes_and_edges(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
    ) -> None:
        hierarchy = getattr(self._topic_model, "get_hierarchy", None)
        get_hierarchy = hierarchy if callable(hierarchy) else None
        hierarchy_map = get_hierarchy() if get_hierarchy else None

        if hierarchy_map is not None and len(hierarchy_map) > 0:
            self._add_nodes_and_edges_hierarchical(chunks, topic_dists, hierarchy_map)
        else:
            self._add_nodes_and_edges_flat(chunks, topic_dists)

        self._add_structural_chunk_edges(chunks)
        self._add_topic_topic_semantic_edges()

    def _add_nodes_and_edges_flat(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
    ) -> None:
        topic_ids = set()
        for dist in topic_dists:
            topic_ids.update(dist.keys())
        for tid in sorted(topic_ids):
            node_id = make_topic_id(tid)
            meta: dict = {}
            if hasattr(self._topic_model, "get_centroid"):
                cent = self._topic_model.get_centroid(tid)
                if cent is not None:
                    meta["centroid"] = list(cent)
            self._graph.add_node(TopicNode(id=node_id, label=f"Topic {tid}", metadata=meta))
        for chunk, dist in zip(chunks, topic_dists):
            chunk_node = ChunkNode(
                id=make_chunk_id(len(self._graph.nodes)),
                document_id=chunk.document_id,
                position=chunk.position,
                metadata={"text": chunk.text},
            )
            self._graph.add_node(chunk_node)
            for tid, weight in dist.items():
                self._graph.add_edge(
                    make_topic_id(tid),
                    chunk_node.id,
                    weight=weight,
                    kind=EdgeKind.TOPIC_CHUNK,
                )

    def _add_nodes_and_edges_hierarchical(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
        hierarchy_map: dict[int, list[int]],
    ) -> None:
        for coarse_id, sub_ids in hierarchy_map.items():
            coarse_node_id = make_topic_id(coarse_id)
            coarse_meta: dict = {}
            if hasattr(self._topic_model, "get_coarse_centroid"):
                cent = self._topic_model.get_coarse_centroid(coarse_id)
                if cent is not None:
                    coarse_meta["centroid"] = list(cent)
            self._graph.add_node(
                TopicNode(
                    id=coarse_node_id,
                    label=f"Topic {coarse_id}",
                    level=0,
                    metadata=coarse_meta,
                )
            )
            for encoded_sub in sub_ids:
                sub_idx = encoded_sub % 100
                sub_node_id = make_subtopic_id(coarse_id, sub_idx)
                sub_meta: dict = {}
                if hasattr(self._topic_model, "get_centroid"):
                    cent = self._topic_model.get_centroid(encoded_sub)
                    if cent is not None:
                        sub_meta["centroid"] = list(cent)
                self._graph.add_node(
                    TopicNode(
                        id=sub_node_id,
                        label=f"Topic {coarse_id}:{sub_idx}",
                        level=1,
                        parent_id=coarse_node_id,
                        keywords=self._topic_model.get_topic_keywords(encoded_sub, top_k=5)
                        if hasattr(self._topic_model, "get_topic_keywords")
                        else None,
                        metadata=sub_meta,
                    )
                )
                self._graph.add_edge(
                    coarse_node_id,
                    sub_node_id,
                    weight=1.0,
                    kind=EdgeKind.TOPIC_SUBTOPIC,
                )
        for chunk, dist in zip(chunks, topic_dists):
            chunk_node = ChunkNode(
                id=make_chunk_id(len(self._graph.nodes)),
                document_id=chunk.document_id,
                position=chunk.position,
                metadata={"text": chunk.text},
            )
            self._graph.add_node(chunk_node)
            for encoded_tid, weight in dist.items():
                coarse_id = encoded_tid // 100
                sub_idx = encoded_tid % 100
                sub_node_id = make_subtopic_id(coarse_id, sub_idx)
                if self._graph.get_node(sub_node_id) is not None:
                    self._graph.add_edge(
                        sub_node_id,
                        chunk_node.id,
                        weight=weight,
                        kind=EdgeKind.TOPIC_CHUNK,
                    )

    def _add_structural_chunk_edges(self, chunks: Sequence[Chunk]) -> None:
        by_doc: dict[str, List[Tuple[Chunk, int]]] = {}
        for chunk in chunks:
            doc_id = chunk.document_id
            group = chunk.structural_group if chunk.structural_group is not None else -1
            by_doc.setdefault(doc_id, []).append((chunk, group))
        chunk_id_by_doc_pos: dict[Tuple[str, int], str] = {}
        for node in self._graph.iter_nodes():
            if not isinstance(node, ChunkNode):
                continue
            doc_id = node.document_id
            pos = node.position
            if doc_id is not None and pos is not None:
                chunk_id_by_doc_pos[(doc_id, pos)] = node.id
        for doc_id, doc_chunks in by_doc.items():
            doc_chunks.sort(key=lambda x: x[0].position)
            for i in range(len(doc_chunks) - 1):
                (c_a, g_a), (c_b, g_b) = doc_chunks[i], doc_chunks[i + 1]
                if g_a != g_b or g_a < 0:
                    continue
                id_a = chunk_id_by_doc_pos.get((doc_id, c_a.position))
                id_b = chunk_id_by_doc_pos.get((doc_id, c_b.position))
                if id_a is None or id_b is None:
                    continue
                if self._graph.get_node(id_a) is None or self._graph.get_node(id_b) is None:
                    continue
                self._graph.add_edge(
                    id_a,
                    id_b,
                    weight=1.0,
                    kind=EdgeKind.CHUNK_CHUNK_STRUCTURAL,
                )

    def _add_topic_topic_semantic_edges(
        self,
        *,
        min_cosine: float = 0.0,
        top_k_per_node: Optional[int] = 20,
    ) -> None:
        """Add TOPIC_TOPIC edges between topic nodes with centroids; weight = cos(μ_i, μ_j)."""
        topic_ids_with_centroid: List[str] = []
        centroids: List[Sequence[float]] = []
        for node in self._graph.iter_nodes():
            if node.kind != NodeKind.TOPIC:
                continue
            cent = get_topic_centroid(node)
            if cent is not None:
                topic_ids_with_centroid.append(node.id)
                centroids.append(cent)

        n = len(topic_ids_with_centroid)
        if n < 2:
            return

        def cos_sim(a: Sequence[float], b: Sequence[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na <= 0 or nb <= 0:
                return 0.0
            return dot / (na * nb)

        for i in range(n):
            id_i = topic_ids_with_centroid[i]
            mu_i = centroids[i]
            candidates: List[Tuple[float, str]] = []
            for j in range(n):
                if i == j:
                    continue
                w = cos_sim(mu_i, centroids[j])
                if w >= min_cosine:
                    candidates.append((w, topic_ids_with_centroid[j]))
            if top_k_per_node is not None and len(candidates) > top_k_per_node:
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:top_k_per_node]
            for w, id_j in candidates:
                self._graph.add_edge(id_i, id_j, weight=float(w), kind=EdgeKind.TOPIC_TOPIC)

    def _build_topic_sequences(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[int, float]],
    ) -> Tuple[List[Sequence[str]], List[List[int]]]:
        """Build sequences of topic node ids and parallel structural group per position."""
        by_doc: dict[str, List[Tuple[Chunk, dict[int, float]]]] = {}
        for chunk, dist in zip(chunks, topic_dists):
            by_doc.setdefault(chunk.document_id, []).append((chunk, dist))

        hierarchy_map = None
        if callable(getattr(self._topic_model, "get_hierarchy", None)):
            hierarchy_map = self._topic_model.get_hierarchy()
        use_subtopic_id = hierarchy_map is not None and len(hierarchy_map) > 0

        sequences: List[Sequence[str]] = []
        structural_groups_per_seq: List[List[int]] = []
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda x: x[0].position)
            seq: List[str] = []
            groups: List[int] = []
            for chunk, dist in doc_chunks:
                if not dist:
                    continue
                top_encoded = max(dist.items(), key=lambda kv: kv[1])[0]
                if use_subtopic_id:
                    seq.append(make_subtopic_id(top_encoded // 100, top_encoded % 100))
                else:
                    seq.append(make_topic_id(top_encoded))
                g = chunk.structural_group if chunk.structural_group is not None else -1
                groups.append(g)
            if len(seq) >= 2:
                sequences.append(seq)
                structural_groups_per_seq.append(groups)
        return sequences, structural_groups_per_seq

