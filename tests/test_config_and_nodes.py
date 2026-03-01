import pytest

from bettermem.api.config import BetterMemConfig
from bettermem.core.nodes import (
    ChunkNode,
    KeywordNode,
    Node,
    NodeKind,
    TopicNode,
    make_chunk_id,
    make_keyword_id,
    make_topic_id,
)
from pydantic import ValidationError


def test_config_defaults() -> None:
    cfg = BetterMemConfig()
    assert 0.0 <= cfg.smoothing_lambda <= 1.0
    assert cfg.max_steps >= 1
    assert cfg.navigation_alpha >= 0


def test_config_validation_bounds() -> None:
    with pytest.raises(ValidationError):
        BetterMemConfig(smoothing_lambda=1.5)


def test_node_kinds_and_ids() -> None:
    t_id = make_topic_id(0)
    c_id = make_chunk_id(1)
    k_id = make_keyword_id("hello")

    assert t_id.startswith("t:")
    assert c_id.startswith("c:")
    assert k_id.startswith("k:")

    topic = TopicNode(id=t_id, label="Topic 0", keywords=["a", "b"])
    chunk = ChunkNode(id=c_id, document_id="doc-1", position=0)
    keyword = KeywordNode(id=k_id, term="hello")
    generic = Node(id="x", kind=NodeKind.TOPIC)

    assert topic.kind is NodeKind.TOPIC
    assert chunk.kind is NodeKind.CHUNK
    assert keyword.kind is NodeKind.KEYWORD
    assert generic.kind is NodeKind.TOPIC

