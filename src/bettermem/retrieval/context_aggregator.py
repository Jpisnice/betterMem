from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple

from bettermem.core.graph import Graph
from bettermem.core.nodes import ChunkNode, Node, NodeId, NodeKind


class ContextAggregator:
    """Select final chunks from scored nodes."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def select(
        self,
        scores: Mapping[NodeId, float],
        *,
        top_k: int = 8,
        diversity: bool = True,
    ) -> List[ChunkNode]:
        """Return top-k chunk nodes according to scores, with optional diversity."""
        # Filter to chunk nodes only
        chunk_scores: List[Tuple[ChunkNode, float]] = []
        for node_id, score in scores.items():
            node = self._graph.get_node(node_id)
            if isinstance(node, ChunkNode):
                chunk_scores.append((node, float(score)))

        if not chunk_scores:
            return []

        # Initial ranking by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        if not diversity:
            return [c for c, _ in chunk_scores[:top_k]]

        # Simple diversity heuristic: limit chunks per document
        selected: List[ChunkNode] = []
        per_doc: dict[str, int] = {}
        for chunk, _score in chunk_scores:
            doc_id = chunk.document_id or ""
            if per_doc.get(doc_id, 0) >= max(1, top_k // 2):
                continue
            selected.append(chunk)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            if len(selected) >= top_k:
                break

        return selected

