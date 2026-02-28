from __future__ import annotations

import pytest

from bettermem.core.edges import EdgeKind
from bettermem.core.graph import Graph
from bettermem.core.nodes import ChunkNode, KeywordNode, NodeKind, TopicNode, make_chunk_id, make_keyword_id, make_topic_id
from bettermem.storage.sparse_tensor import SparseTensor3D


def _build_small_graph() -> Graph:
    graph = Graph()

    t0 = TopicNode(id=make_topic_id(0), label="T0")
    t1 = TopicNode(id=make_topic_id(1), label="T1")
    c0 = ChunkNode(id=make_chunk_id(0), document_id="doc0", position=0)
    k0 = KeywordNode(id=make_keyword_id("kw"), term="kw")

    for node in (t0, t1, c0, k0):
        graph.add_node(node)

    graph.add_edge(t0.id, t1.id, weight=2.0, kind=EdgeKind.TOPIC_TOPIC)
    graph.add_edge(t0.id, c0.id, weight=1.0, kind=EdgeKind.TOPIC_CHUNK)
    graph.add_edge(k0.id, t0.id, weight=1.0, kind=EdgeKind.KEYWORD_TOPIC)

    return graph


def test_graph_add_and_count_nodes_edges() -> None:
    graph = _build_small_graph()

    assert graph.count_nodes() == 4
    assert graph.count_nodes(NodeKind.TOPIC) == 2
    assert graph.count_nodes(NodeKind.CHUNK) == 1
    assert graph.count_nodes(NodeKind.KEYWORD) == 1

    # Three edges added
    assert graph.count_edges() == 3


def test_graph_normalize_and_prune_edges() -> None:
    graph = _build_small_graph()
    graph.normalize_edges()

    # Outgoing edges from t0 should sum to 1
    neighbors = graph.get_neighbors(make_topic_id(0))
    assert pytest.approx(sum(neighbors.values()), rel=1e-6) == 1.0  # type: ignore[name-defined]

    # Prune with a threshold that removes only the smaller edge
    graph.prune_edges(threshold=0.5)
    neighbors_after = graph.get_neighbors(make_topic_id(0))
    # Only the larger edge (originally weight 2.0) should remain
    assert len(neighbors_after) == 1


def test_graph_roundtrip_serialization() -> None:
    graph = _build_small_graph()
    payload = graph.to_dict()
    restored = Graph.from_dict(payload)

    assert restored.count_nodes() == graph.count_nodes()
    assert restored.count_edges() == graph.count_edges()

    # Check that node kinds survive roundtrip
    kinds = {n.kind for n in graph.iter_nodes()}
    restored_kinds = {n.kind for n in restored.iter_nodes()}
    assert kinds == restored_kinds


def test_sparse_tensor_roundtrip() -> None:
    # Build a small tensor
    i = make_topic_id(0)
    j = make_topic_id(1)
    k1 = make_chunk_id(0)
    k2 = make_chunk_id(1)

    tensor = SparseTensor3D(
        data={
            (i, j): {k1: 0.3, k2: 0.7},
        }
    )

    payload = tensor.to_dict()
    restored = SparseTensor3D.from_dict(payload)

    assert restored.data == tensor.data

