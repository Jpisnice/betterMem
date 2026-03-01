from __future__ import annotations

from pathlib import Path

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.core.graph import Graph
from bettermem.core.nodes import Node, NodeKind
from bettermem.core.transition_model import TransitionModel
from bettermem.storage.persistence import (
    GRAPH_JOBLIB,
    TRANSITION_JOBLIB,
    load_index,
    save_index,
)


def _build_sample_graph_and_model() -> tuple[Graph, TransitionModel]:
    graph = Graph()
    for nid in ("n1", "n2"):
        graph.add_node(Node(id=nid, kind=NodeKind.TOPIC))
    graph.add_edge("n1", "n2", weight=1.0)

    tm = TransitionModel()
    tm.fit([["n1", "n2"]])
    return graph, tm


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    graph, tm = _build_sample_graph_and_model()
    cfg = BetterMemConfig(order=2)

    save_dir = tmp_path / "index"
    save_index(str(save_dir), graph=graph, transition_model=tm, config=cfg)

    assert (save_dir / GRAPH_JOBLIB).exists()
    assert (save_dir / TRANSITION_JOBLIB).exists()
    assert (save_dir / "config.json").exists()

    loaded_graph, loaded_tm, loaded_cfg = load_index(str(save_dir))

    assert loaded_cfg is not None
    assert isinstance(loaded_cfg, BetterMemConfig)
    assert loaded_graph.count_nodes() == graph.count_nodes()
    assert loaded_graph.count_edges() == graph.count_edges()

    # Transition probabilities for observed pair should match
    probs_orig = tm.transition_prob("n1", "n2").probs
    probs_loaded = loaded_tm.transition_prob("n1", "n2").probs
    assert probs_orig == probs_loaded


def test_bettermem_save_and_load_client(tmp_path: Path) -> None:
    graph, tm = _build_sample_graph_and_model()
    cfg = BetterMemConfig(max_steps=16)

    client = BetterMem(config=cfg)
    # Wire in the prebuilt components
    client._graph = graph  # type: ignore[attr-defined]
    client._transition_model = tm  # type: ignore[attr-defined]

    save_dir = tmp_path / "client_index"
    client.save(str(save_dir))

    loaded = BetterMem.load(str(save_dir))
    assert isinstance(loaded, BetterMem)
    assert loaded.config.max_steps == cfg.max_steps
    assert loaded._graph is not None  # type: ignore[attr-defined]
    assert loaded._transition_model is not None  # type: ignore[attr-defined]

