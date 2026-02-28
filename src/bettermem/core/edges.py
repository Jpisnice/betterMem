from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .nodes import NodeId


class EdgeKind(str, Enum):
    TOPIC_TOPIC = "topic-topic"
    TOPIC_SUBTOPIC = "topic-subtopic"
    TOPIC_CHUNK = "topic-chunk"
    CHUNK_CHUNK = "chunk-chunk"
    CHUNK_CHUNK_STRUCTURAL = "chunk-chunk-structural"
    KEYWORD_TOPIC = "keyword-topic"
    KEYWORD_CHUNK = "keyword-chunk"


@dataclass(slots=True)
class Edge:
    """Directed weighted edge between two nodes."""

    source: NodeId
    target: NodeId
    weight: float = 1.0
    kind: Optional[EdgeKind] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

