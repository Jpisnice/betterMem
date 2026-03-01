from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


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
    """Topic node for semantic hierarchical traversal (coarse or subtopic).

    Stores centroid for navigation scoring and chunk_ids for fast lookup
    of associated chunks without scanning edges.
    """

    label: Optional[str] = None
    keywords: Optional[list[str]] = None
    level: int = 0
    parent_id: Optional[NodeId] = None
    chunk_ids: List[NodeId] = field(default_factory=list)

    def __init__(
        self,
        id: NodeId,
        *,
        label: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        level: int = 0,
        parent_id: Optional[NodeId] = None,
        chunk_ids: Optional[Sequence[NodeId]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        Node.__init__(self, id=id, kind=NodeKind.TOPIC, metadata=dict(metadata or {}))
        self.label = label
        self.keywords = list(keywords) if keywords is not None else None
        self.level = level
        self.parent_id = parent_id
        self.chunk_ids = list(chunk_ids) if chunk_ids is not None else []


@dataclass(slots=True)
class ChunkNode(Node):
    """Node for a corpus chunk with optional stored embedding reference.

    text is in metadata['text']; embedding can be stored at index time
    for explain/retrieval without recomputing.
    """

    document_id: Optional[str] = None
    position: Optional[int] = None
    embedding: Optional[Sequence[float]] = None

    def __init__(
        self,
        id: NodeId,
        *,
        document_id: Optional[str] = None,
        position: Optional[int] = None,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        Node.__init__(self, id=id, kind=NodeKind.CHUNK, metadata=dict(metadata or {}))
        self.document_id = document_id
        self.position = position
        self.embedding = list(embedding) if embedding is not None else None


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


def make_subtopic_id(coarse: int, sub: int) -> NodeId:
    return f"t:{coarse}:{sub}"


def make_chunk_id(index: int) -> NodeId:
    return f"c:{index}"


def make_keyword_id(term: str) -> NodeId:
    return f"k:{term}"


def get_topic_centroid(node: Node) -> Optional[Sequence[float]]:
    """Return the embedding centroid for a topic node if stored in metadata.

    Centroids are stored as metadata['centroid'] (list of floats) during
    indexing when the topic model provides get_centroid(). Returns None
    for non-topic nodes or when no centroid is stored.
    """
    if not isinstance(node, TopicNode):
        return None
    raw = node.metadata.get("centroid")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and all(isinstance(x, (int, float)) for x in raw):
        return raw
    return None

