from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from .edges import Edge, EdgeKind
from .nodes import ChunkNode, KeywordNode, Node, NodeId, NodeKind, TopicNode


class Graph:
    """Directed weighted graph over topic, chunk, and keyword nodes.

    Internally, nodes are stored in a registry, and outgoing edges are
    represented as sparse adjacency maps:

        neighbors[i][j] = weight_ij

    which can be viewed as entries of an adjacency matrix A where
    A_ij = w_ij.
    """

    def __init__(self) -> None:
        self._nodes: Dict[NodeId, Node] = {}
        self._neighbors: Dict[NodeId, Dict[NodeId, float]] = defaultdict(dict)
        self._edge_kinds: Dict[Tuple[NodeId, NodeId], EdgeKind] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        """Register a node in the graph."""
        self._nodes[node.id] = node
        self._neighbors.setdefault(node.id, {})

    def get_node(self, node_id: NodeId) -> Optional[Node]:
        return self._nodes.get(node_id)

    def iter_nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    def add_edge(
        self,
        source: NodeId,
        target: NodeId,
        weight: float = 1.0,
        kind: Optional[EdgeKind] = None,
    ) -> None:
        """Add or update a directed weighted edge."""
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("Both source and target must exist in the graph.")

        current = self._neighbors[source].get(target, 0.0)
        self._neighbors[source][target] = current + float(weight)
        if kind is not None:
            self._edge_kinds[(source, target)] = kind

    def get_neighbors(self, node_id: NodeId) -> Mapping[NodeId, float]:
        """Return outgoing neighbors and their weights for a node."""
        return self._neighbors.get(node_id, {})

    def get_edge_kind(self, source: NodeId, target: NodeId) -> Optional[EdgeKind]:
        return self._edge_kinds.get((source, target))

    def iter_edges(self) -> Iterable[Edge]:
        for source, targets in self._neighbors.items():
            for target, weight in targets.items():
                yield Edge(
                    source=source,
                    target=target,
                    weight=weight,
                    kind=self._edge_kinds.get((source, target)),
                )

    # ------------------------------------------------------------------
    # Normalization and pruning
    # ------------------------------------------------------------------
    def normalize_edges(self) -> None:
        """Normalize outgoing edge weights so each node's outgoing sum is 1."""
        for source, targets in self._neighbors.items():
            total = sum(targets.values())
            if total <= 0.0:
                continue
            inv_total = 1.0 / total
            for target in list(targets.keys()):
                targets[target] = targets[target] * inv_total

    def prune_edges(self, threshold: float) -> None:
        """Remove edges with weight strictly below the given threshold."""
        for source, targets in list(self._neighbors.items()):
            to_delete = [t for t, w in targets.items() if w < threshold]
            for t in to_delete:
                targets.pop(t, None)
                self._edge_kinds.pop((source, t), None)
            if not targets:
                self._neighbors.pop(source, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> Mapping[NodeId, Node]:
        return self._nodes

    @property
    def adjacency(self) -> Mapping[NodeId, Mapping[NodeId, float]]:
        return self._neighbors

    def count_nodes(self, kind: Optional[NodeKind] = None) -> int:
        if kind is None:
            return len(self._nodes)
        return sum(1 for node in self._nodes.values() if node.kind == kind)

    def count_edges(self) -> int:
        return sum(len(targets) for targets in self._neighbors.values())

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Mapping[str, object]:
        """Return a JSON-serializable representation of the graph."""
        nodes_payload = []
        for node in self._nodes.values():
            base = {
                "id": node.id,
                "kind": node.kind.value,
                "metadata": dict(node.metadata),
            }
            if isinstance(node, TopicNode):
                base["label"] = node.label
                base["keywords"] = node.keywords
            elif isinstance(node, ChunkNode):
                base["document_id"] = node.document_id
                base["position"] = node.position
            elif isinstance(node, KeywordNode):
                base["term"] = node.term
            nodes_payload.append(base)

        edges_payload = [
            {
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
                "kind": e.kind.value if e.kind is not None else None,
            }
            for e in self.iter_edges()
        ]

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Graph":
        """Rebuild a Graph instance from a serialized representation."""
        graph = cls()

        for n in payload.get("nodes", []) or []:  # type: ignore[assignment]
            nid = n.get("id")  # type: ignore[assignment]
            kind_str = n.get("kind")  # type: ignore[assignment]
            metadata = dict(n.get("metadata", {}) or {})  # type: ignore[assignment]

            if nid is None or kind_str is None:
                continue

            kind = NodeKind(kind_str)
            if kind == NodeKind.TOPIC:
                node = TopicNode(
                    id=nid,
                    label=n.get("label"),  # type: ignore[arg-type]
                    keywords=n.get("keywords"),  # type: ignore[arg-type]
                    metadata=metadata,
                )
            elif kind == NodeKind.CHUNK:
                node = ChunkNode(
                    id=nid,
                    document_id=n.get("document_id"),  # type: ignore[arg-type]
                    position=n.get("position"),  # type: ignore[arg-type]
                    metadata=metadata,
                )
            elif kind == NodeKind.KEYWORD:
                node = KeywordNode(
                    id=nid,
                    term=n.get("term"),  # type: ignore[arg-type]
                    metadata=metadata,
                )
            else:
                node = Node(id=nid, kind=kind, metadata=metadata)

            graph.add_node(node)

        for e in payload.get("edges", []) or []:  # type: ignore[assignment]
            source = e.get("source")  # type: ignore[assignment]
            target = e.get("target")  # type: ignore[assignment]
            weight = float(e.get("weight", 1.0))  # type: ignore[arg-type]
            kind_str = e.get("kind")  # type: ignore[assignment]
            edge_kind = EdgeKind(kind_str) if kind_str is not None else None
            if source is None or target is None:
                continue
            graph.add_node(graph.get_node(source) or Node(source, NodeKind.TOPIC))  # safety
            graph.add_node(graph.get_node(target) or Node(target, NodeKind.TOPIC))
            graph.add_edge(source, target, weight=weight, kind=edge_kind)

        return graph

