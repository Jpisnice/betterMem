from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class NodeKind(str, Enum):
    TOPIC = "topic"
    CHUNK = "chunk"
    KEYWORD = "keyword"


NodeId = str


@dataclass(slots=True)
class Node:
    """Generic graph node used internally by BetterMem."""

    id: NodeId
    kind: NodeKind
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TopicNode(Node):
    """Specialized node representing a latent topic."""

    label: Optional[str] = None
    keywords: Optional[list[str]] = None

    def __init__(
        self,
        id: NodeId,
        *,
        label: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Explicitly call the base dataclass __init__ instead of super()
        # to avoid interactions between dataclass(slots=True) and zero-arg super.
        Node.__init__(self, id=id, kind=NodeKind.TOPIC, metadata=dict(metadata or {}))
        self.label = label
        self.keywords = list(keywords) if keywords is not None else None


@dataclass(slots=True)
class ChunkNode(Node):
    """Node corresponding to a text chunk derived from the corpus."""

    document_id: Optional[str] = None
    position: Optional[int] = None

    def __init__(
        self,
        id: NodeId,
        *,
        document_id: Optional[str] = None,
        position: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        Node.__init__(self, id=id, kind=NodeKind.CHUNK, metadata=dict(metadata or {}))
        self.document_id = document_id
        self.position = position


@dataclass(slots=True)
class KeywordNode(Node):
    """Optional node representing a lexical keyword."""

    term: str | None = None

    def __init__(
        self,
        id: NodeId,
        *,
        term: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        Node.__init__(self, id=id, kind=NodeKind.KEYWORD, metadata=dict(metadata or {}))
        self.term = term


def make_topic_id(index: int) -> NodeId:
    return f"t:{index}"


def make_chunk_id(index: int) -> NodeId:
    return f"c:{index}"


def make_keyword_id(term: str) -> NodeId:
    return f"k:{term}"

