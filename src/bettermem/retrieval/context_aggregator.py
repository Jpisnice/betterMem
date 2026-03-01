from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Set, Tuple

from bettermem.core.edges import EdgeKind
from bettermem.core.graph import Graph
from bettermem.core.nodes import ChunkNode, NodeId, NodeKind


@dataclass
class ContextWindow:
    """A contiguous or structurally expanded window of chunks with a single score."""

    chunks: List[ChunkNode]
    score: float

    def __iter__(self) -> Iterable[ChunkNode]:
        return iter(self.chunks)


class ContextAggregator:
    """Select final chunks or context windows from scored nodes."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def _get_structural_neighbors(self, chunk_id: NodeId, half_size: int = 2) -> Set[NodeId]:
        """Return chunk ids reachable by following CHUNK_CHUNK_STRUCTURAL edges within half_size steps."""
        if half_size <= 0:
            return {chunk_id}
        seen: Set[NodeId] = {chunk_id}
        queue: deque[Tuple[NodeId, int]] = deque([(chunk_id, 0)])
        while queue:
            nid, d = queue.popleft()
            if d >= half_size:
                continue
            for neigh_id in self._graph.get_neighbors(nid):
                neigh = self._graph.get_node(neigh_id)
                if neigh is None or neigh.kind != NodeKind.CHUNK:
                    continue
                kind = self._graph.get_edge_kind(nid, neigh_id)
                if kind != EdgeKind.CHUNK_CHUNK_STRUCTURAL:
                    continue
                if neigh_id in seen:
                    continue
                seen.add(neigh_id)
                queue.append((neigh_id, d + 1))
        return seen

    def _chunks_to_ordered_window(self, chunk_ids: Set[NodeId]) -> List[ChunkNode]:
        """Order chunk ids by (document_id, position) and return ChunkNode list."""
        nodes: List[Tuple[Optional[str], Optional[int], ChunkNode]] = []
        for cid in chunk_ids:
            node = self._graph.get_node(cid)
            if isinstance(node, ChunkNode):
                nodes.append((node.document_id, node.position, node))
        nodes.sort(key=lambda x: (x[0] or "", x[1] if x[1] is not None else -1))
        return [n[2] for n in nodes]

    def _window_score(
        self,
        chunk_ids: Set[NodeId],
        scores: Mapping[NodeId, float],
        section_coherence_bonus: float = 0.1,
    ) -> float:
        """S_window(W) = avg(chunk scores) + section_coherence_bonus * coherence(W)."""
        if not chunk_ids:
            return 0.0
        total = sum(scores.get(cid, 0.0) for cid in chunk_ids)
        avg = total / len(chunk_ids)
        coherence = 1.0
        doc_ids: Set[Optional[str]] = set()
        for cid in chunk_ids:
            node = self._graph.get_node(cid)
            if isinstance(node, ChunkNode):
                doc_ids.add(node.document_id)
        if len(doc_ids) > 1:
            coherence = 0.5
        return avg + section_coherence_bonus * coherence

    def select(
        self,
        scores: Mapping[NodeId, float],
        *,
        top_k: int = 8,
        diversity: bool = True,
        use_windows: bool = True,
        window_half_size: int = 2,
    ) -> List[ContextWindow]:
        """Return top-k context windows (or single-chunk windows when use_windows=False).

        When use_windows is True: rank seed chunks, expand each by ±window_half_size
        structural neighbors, form windows, score them, and return top-k windows.
        """
        chunk_scores: List[Tuple[ChunkNode, float]] = []
        for node_id, score in scores.items():
            node = self._graph.get_node(node_id)
            if isinstance(node, ChunkNode):
                chunk_scores.append((node, float(score)))

        if not chunk_scores:
            return []

        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        if not use_windows:
            if not diversity:
                return [
                    ContextWindow(chunks=[c], score=s)
                    for c, s in chunk_scores[:top_k]
                ]
            selected: List[ContextWindow] = []
            per_doc: dict[str, int] = {}
            for chunk, sc in chunk_scores:
                doc_id = chunk.document_id or ""
                if per_doc.get(doc_id, 0) >= max(1, top_k // 2):
                    continue
                selected.append(ContextWindow(chunks=[chunk], score=sc))
                per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
                if len(selected) >= top_k:
                    break
            return selected

        # Window mode: expand seeds, merge into windows, score, dedupe, top-k
        seen_window_keys: Set[frozenset] = set()
        window_list: List[ContextWindow] = []
        for chunk, _ in chunk_scores[: max(top_k * 3, 20)]:
            seed_id = chunk.id
            neighbor_ids = self._get_structural_neighbors(seed_id, half_size=window_half_size)
            key = frozenset(neighbor_ids)
            if key in seen_window_keys:
                continue
            seen_window_keys.add(key)
            ordered = self._chunks_to_ordered_window(neighbor_ids)
            if not ordered:
                continue
            sc = self._window_score(neighbor_ids, scores, section_coherence_bonus=0.1)
            window_list.append(ContextWindow(chunks=ordered, score=sc))
        window_list.sort(key=lambda w: w.score, reverse=True)
        return window_list[:top_k]
