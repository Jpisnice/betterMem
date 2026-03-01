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
    make_topic_path_id,
)

# Semantic hierarchical traversal: topic model must provide get_hierarchy().
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
    def build_index(
        self,
        corpus: Iterable[str],
        *,
        neighbor_top_k: Optional[int] = None,
        neighbor_min_cosine: Optional[float] = None,
        topic_chunk_top_m: Optional[int] = None,
        topic_chunk_min_prob: Optional[float] = None,
        topic_chunk_ancestor_decay: Optional[float] = None,
    ) -> None:
        """End-to-end indexing pipeline."""
        # 1. Chunk corpus
        chunks = self._chunker.chunk(corpus)
        chunk_texts = [c.text for c in chunks]

        # 2. Fit topic model on chunks so clustering discovers multiple topics
        #    even from a single long document.
        if chunk_texts:
            self._topic_model.fit(chunk_texts)

        topic_dists = self._topic_model.transform(chunk_texts)

        # 3. Create topic nodes and chunk nodes, and connect topic–chunk edges
        self._add_nodes_and_edges(
            chunks,
            topic_dists,
            neighbor_top_k=neighbor_top_k,
            neighbor_min_cosine=neighbor_min_cosine,
            topic_chunk_top_m=topic_chunk_top_m,
            topic_chunk_min_prob=topic_chunk_min_prob,
            topic_chunk_ancestor_decay=topic_chunk_ancestor_decay,
        )

        # 4. Build topic sequences and fit transition model
        sequences, structural_groups_per_seq = self._build_topic_sequences(chunks, topic_dists)
        self._transition_model.fit(sequences, structural_groups_per_seq=structural_groups_per_seq)

    def _add_nodes_and_edges(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[str, float]],
        *,
        neighbor_top_k: Optional[int] = None,
        neighbor_min_cosine: Optional[float] = None,
        topic_chunk_top_m: Optional[int] = None,
        topic_chunk_min_prob: Optional[float] = None,
        topic_chunk_ancestor_decay: Optional[float] = None,
    ) -> None:
        get_hierarchy = getattr(self._topic_model, "get_hierarchy", None)
        if not callable(get_hierarchy):
            raise ValueError(
                "Semantic hierarchical traversal requires a topic model with get_hierarchy() "
                "(e.g. SemanticHierarchicalTopicModel)."
            )
        hierarchy_map = get_hierarchy()
        if not hierarchy_map:
            raise ValueError("Topic model get_hierarchy() returned empty; cannot build graph.")
        self._add_nodes_and_edges_hierarchical(
            chunks,
            topic_dists,
            hierarchy_map,
            topic_chunk_top_m=topic_chunk_top_m,
            topic_chunk_min_prob=topic_chunk_min_prob,
            topic_chunk_ancestor_decay=topic_chunk_ancestor_decay,
        )
        self._add_structural_chunk_edges(chunks)
        self._graph.build_topic_indexes()
        self._add_topic_topic_semantic_edges(
            top_k_per_node=neighbor_top_k,
            min_cosine=neighbor_min_cosine if neighbor_min_cosine is not None else 0.0,
        )

    def _add_nodes_and_edges_hierarchical(
        self,
        chunks: Sequence[Chunk],
        topic_dists: Sequence[dict[str, float]],
        hierarchy_map: dict[str, list[str]],
        *,
        topic_chunk_top_m: Optional[int] = None,
        topic_chunk_min_prob: Optional[float] = None,
        topic_chunk_ancestor_decay: Optional[float] = None,
    ) -> None:
        chunk_texts = [c.text for c in chunks]
        embeddings: Optional[Sequence[Sequence[float]]] = None
        if hasattr(self._topic_model, "embed_texts") and chunk_texts:
            emb = self._topic_model.embed_texts(chunk_texts)
            if emb is not None:
                embeddings = emb

        get_all_topic_ids = getattr(self._topic_model, "get_all_topic_ids", None)
        if not callable(get_all_topic_ids):
            raise ValueError("Topic model must provide get_all_topic_ids().")
        all_topic_ids = get_all_topic_ids()
        if not all_topic_ids:
            raise ValueError("Topic model get_all_topic_ids() returned empty.")

        child_to_parent: dict[str, str] = {}
        for parent_id, child_ids in hierarchy_map.items():
            for cid in child_ids:
                child_to_parent[cid] = parent_id

        def depth(topic_id: str) -> int:
            parts = topic_id[2:].split(".") if topic_id.startswith("t:") else []
            return len([p for p in parts if p.isdigit()])

        get_parents = getattr(self._topic_model, "get_parents", None)
        get_leaf_topic_ids = getattr(self._topic_model, "get_leaf_topic_ids", None)
        leaf_ids_set: set[str] = set()
        if callable(get_leaf_topic_ids):
            leaf_ids_set = set(get_leaf_topic_ids())

        top_m = 2 if topic_chunk_top_m is None else max(1, topic_chunk_top_m)
        min_prob = 0.15 if topic_chunk_min_prob is None else topic_chunk_min_prob
        ancestor_decay = 0.7 if topic_chunk_ancestor_decay is None else topic_chunk_ancestor_decay

        sorted_topic_ids = sorted(all_topic_ids, key=depth)
        for topic_id in sorted_topic_ids:
            parent_id = child_to_parent.get(topic_id)
            level = depth(topic_id)
            meta: dict = {}
            if hasattr(self._topic_model, "get_centroid"):
                cent = self._topic_model.get_centroid(topic_id)
                if cent is not None:
                    meta["centroid"] = list(cent)
            keywords = None
            if hasattr(self._topic_model, "get_topic_keywords"):
                keywords = self._topic_model.get_topic_keywords(topic_id, top_k=5)
            if callable(get_parents):
                parent_ids_list = get_parents(topic_id)
            else:
                parent_ids_list = [parent_id] if parent_id is not None else []
            self._graph.add_node(
                TopicNode(
                    id=topic_id,
                    label=f"Topic {topic_id}",
                    level=level,
                    parent_ids=parent_ids_list,
                    keywords=keywords,
                    metadata=meta,
                )
            )
            for pid in parent_ids_list:
                self._graph.add_edge(
                    pid,
                    topic_id,
                    weight=1.0,
                    kind=EdgeKind.TOPIC_SUBTOPIC,
                )

        for chunk_idx, (chunk, dist) in enumerate(zip(chunks, topic_dists)):
            chunk_emb = None
            if embeddings is not None and chunk_idx < len(embeddings):
                chunk_emb = embeddings[chunk_idx]
            chunk_node = ChunkNode(
                id=make_chunk_id(chunk_idx),
                document_id=chunk.document_id,
                position=chunk.position,
                embedding=chunk_emb,
                metadata={"text": chunk.text},
            )
            self._graph.add_node(chunk_node)

            # Sparse topic–chunk: top M leaf topics above threshold, then best leaf + ancestors with decay
            leaf_dist = {tid: p for tid, p in dist.items() if tid in leaf_ids_set} if leaf_ids_set else dict(dist)
            items = sorted(leaf_dist.items(), key=lambda kv: kv[1], reverse=True)
            items = [(tid, p) for tid, p in items[:top_m] if p >= min_prob]

            for leaf_id, weight in items:
                topic_node = self._graph.get_node(leaf_id)
                if topic_node is None:
                    continue
                self._graph.add_edge(leaf_id, chunk_node.id, weight=weight, kind=EdgeKind.TOPIC_CHUNK)
                if isinstance(topic_node, TopicNode) and chunk_node.id not in topic_node.chunk_ids:
                    topic_node.chunk_ids.append(chunk_node.id)

                # Attach to ancestors with decayed weight (hierarchical recall)
                if ancestor_decay <= 0:
                    continue
                current = leaf_id
                anc_weight = float(weight)
                while True:
                    parents = get_parents(current) if callable(get_parents) else []
                    if not parents:
                        break
                    anc_weight *= ancestor_decay
                    if anc_weight < 1e-9:
                        break
                    for pid in parents:
                        p_node = self._graph.get_node(pid)
                        if p_node is not None:
                            self._graph.add_edge(pid, chunk_node.id, weight=anc_weight, kind=EdgeKind.TOPIC_CHUNK)
                            if isinstance(p_node, TopicNode) and chunk_node.id not in p_node.chunk_ids:
                                p_node.chunk_ids.append(chunk_node.id)
                    current = parents[0]

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
                self._graph.add_edge(
                    id_b,
                    id_a,
                    weight=1.0,
                    kind=EdgeKind.CHUNK_CHUNK_STRUCTURAL,
                )

    def _add_topic_topic_semantic_edges(
        self,
        *,
        min_cosine: float = 0.0,
        top_k_per_node: Optional[int] = 20,
        filter_ancestor_descendant: bool = True,
    ) -> None:
        """Add TOPIC_TOPIC edges via ANN kNN over topic centroids; weight = cos(μ_i, μ_j)."""
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

        import numpy as np

        try:
            import hnswlib
        except ImportError:
            self._add_topic_topic_semantic_edges_bruteforce(
                topic_ids_with_centroid, centroids, min_cosine=min_cosine, top_k_per_node=top_k_per_node
            )
            return

        dim = len(centroids[0])
        data = np.array(centroids, dtype=np.float32)
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(data, np.arange(n))
        k_query = (top_k_per_node + 2) if top_k_per_node is not None else (n + 1)
        k_query = min(k_query, n)
        index.set_ef(max(50, k_query * 2))

        for i in range(n):
            id_i = topic_ids_with_centroid[i]
            labels, distances = index.knn_query(data[i : i + 1], k=k_query)
            labels = labels[0]
            distances = distances[0]
            candidates: List[Tuple[float, str]] = []
            for idx, (j, dist) in enumerate(zip(labels, distances)):
                if j == i:
                    continue
                if j < 0:
                    continue
                sim = 1.0 - float(dist)
                if sim < min_cosine:
                    continue
                id_j = topic_ids_with_centroid[j]
                if filter_ancestor_descendant and (
                    self._graph.is_ancestor(id_j, id_i) or self._graph.is_ancestor(id_i, id_j)
                ):
                    continue
                candidates.append((sim, id_j))
            if top_k_per_node is not None and len(candidates) > top_k_per_node:
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:top_k_per_node]
            for w, id_j in candidates:
                self._graph.add_edge(id_i, id_j, weight=float(w), kind=EdgeKind.TOPIC_TOPIC)

    def _add_topic_topic_semantic_edges_bruteforce(
        self,
        topic_ids_with_centroid: List[str],
        centroids: List[Sequence[float]],
        *,
        min_cosine: float = 0.0,
        top_k_per_node: Optional[int] = 20,
    ) -> None:
        """Fallback O(n²) when hnswlib not available."""
        n = len(topic_ids_with_centroid)

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
        topic_dists: Sequence[dict[str, float]],
    ) -> Tuple[List[Sequence[str]], List[List[int]]]:
        """Build sequences of leaf topic node ids and structural groups."""
        by_doc: dict[str, List[Tuple[Chunk, dict[str, float]]]] = {}
        for chunk, dist in zip(chunks, topic_dists):
            by_doc.setdefault(chunk.document_id, []).append((chunk, dist))

        sequences: List[Sequence[str]] = []
        structural_groups_per_seq: List[List[int]] = []
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda x: x[0].position)
            seq: List[str] = []
            groups: List[int] = []
            for chunk, dist in doc_chunks:
                if not dist:
                    continue
                top_topic_id = max(dist.items(), key=lambda kv: kv[1])[0]
                seq.append(top_topic_id)
                g = chunk.structural_group if chunk.structural_group is not None else -1
                groups.append(g)
            if len(seq) >= 2:
                sequences.append(seq)
                structural_groups_per_seq.append(groups)
        return sequences, structural_groups_per_seq

