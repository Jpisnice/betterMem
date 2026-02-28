from __future__ import annotations

import json
import os
from typing import Optional, Tuple

from bettermem.api.config import BetterMemConfig
from bettermem.core.graph import Graph
from bettermem.core.transition_model import TransitionModel


def save_index(
    path: str,
    *,
    graph: Graph,
    transition_model: TransitionModel,
    config: Optional[BetterMemConfig] = None,
) -> None:
    """Persist graph, transition model, and optional config to a directory.

    The directory will contain:
      - graph.json
      - transition.json
      - config.json (optional)
    """
    os.makedirs(path, exist_ok=True)

    graph_path = os.path.join(path, "graph.json")
    tm_path = os.path.join(path, "transition.json")
    cfg_path = os.path.join(path, "config.json")

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f)

    with open(tm_path, "w", encoding="utf-8") as f:
        json.dump(transition_model.to_dict(), f)

    if config is not None:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f)


def load_index(path: str) -> Tuple[Graph, TransitionModel, Optional[BetterMemConfig]]:
    """Load graph, transition model, and optional config from a directory."""
    graph_path = os.path.join(path, "graph.json")
    tm_path = os.path.join(path, "transition.json")
    cfg_path = os.path.join(path, "config.json")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_payload = json.load(f)
    graph = Graph.from_dict(graph_payload)

    with open(tm_path, "r", encoding="utf-8") as f:
        tm_payload = json.load(f)
    transition_model = TransitionModel.from_dict(tm_payload)

    config: Optional[BetterMemConfig] = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_payload = json.load(f)
        config = BetterMemConfig(**cfg_payload)

    return graph, transition_model, config

