from __future__ import annotations

import json
import os
from typing import Any, Optional, Tuple

import joblib
import numpy as np

from bettermem.api.config import BetterMemConfig
from bettermem.core.graph import Graph
from bettermem.core.transition_model import TransitionModel

# Filenames for joblib persistence
GRAPH_JOBLIB = "graph.joblib"
TRANSITION_JOBLIB = "transition.joblib"
CONFIG_JSON = "config.json"


def _graph_payload_for_joblib(graph: Graph) -> dict[str, Any]:
    """Build graph payload and convert embedding lists to numpy for efficient joblib storage."""
    payload = graph.to_dict()
    for node in payload.get("nodes", []) or []:
        emb = node.get("embedding")
        if emb is not None and isinstance(emb, (list, tuple)):
            node["embedding"] = np.asarray(emb, dtype=np.float64)
    return payload


def save_index(
    path: str,
    *,
    graph: Graph,
    transition_model: TransitionModel,
    config: Optional[BetterMemConfig] = None,
) -> None:
    """Persist graph, transition model, and optional config to a directory.

    The directory will contain:
      - graph.joblib
      - transition.joblib
      - config.json (optional, for human-readable config)
    """
    os.makedirs(path, exist_ok=True)

    graph_path = os.path.join(path, GRAPH_JOBLIB)
    tm_path = os.path.join(path, TRANSITION_JOBLIB)
    cfg_path = os.path.join(path, CONFIG_JSON)

    graph_payload = _graph_payload_for_joblib(graph)
    joblib.dump(graph_payload, graph_path, compress=("gzip", 3))

    joblib.dump(transition_model.to_dict(), tm_path, compress=("gzip", 3))

    if config is not None:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f)


def load_index(
    path: str,
    *,
    mmap_mode: Optional[str] = None,
) -> Tuple[Graph, TransitionModel, Optional[BetterMemConfig]]:
    """Load graph, transition model, and optional config from a directory.

    Expects graph.joblib and transition.joblib (joblib format only).

    Parameters
    ----------
    path
        Directory containing the index files.
    mmap_mode
        If set (e.g. 'r'), passed to joblib.load for memory-mapped loading
        of large arrays.
    """
    graph_path = os.path.join(path, GRAPH_JOBLIB)
    tm_path = os.path.join(path, TRANSITION_JOBLIB)
    cfg_path = os.path.join(path, CONFIG_JSON)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"{GRAPH_JOBLIB} not found in {path!r}")
    kwargs = {} if mmap_mode is None else {"mmap_mode": mmap_mode}
    graph_payload = joblib.load(graph_path, **kwargs)
    graph = Graph.from_dict(graph_payload)

    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"{TRANSITION_JOBLIB} not found in {path!r}")
    tm_payload = joblib.load(tm_path, **kwargs)
    transition_model = TransitionModel.from_dict(tm_payload)

    config: Optional[BetterMemConfig] = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_payload = json.load(f)
        config = BetterMemConfig(**cfg_payload)

    return graph, transition_model, config
