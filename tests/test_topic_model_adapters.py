from __future__ import annotations

import pytest

from bettermem.topic_modeling.semantic_hierarchical import SemanticHierarchicalTopicModel


def test_semantic_hierarchical_adapter_basic() -> None:
    """Smoke test for SemanticHierarchicalTopicModel: fit, transform, hierarchy, query prior."""
    model = SemanticHierarchicalTopicModel(n_coarse=2, n_fine_per_coarse=2, random_state=42)
    docs = ["first short document", "second short document", "third one", "fourth chunk"]
    model.fit(docs)

    dists = model.transform(["query text"])
    assert len(dists) == 1
    assert dists[0]
    assert abs(sum(dists[0].values()) - 1.0) < 1e-5

    hierarchy = model.get_hierarchy()
    assert hierarchy
    for parent_id, child_ids in hierarchy.items():
        assert isinstance(parent_id, str)
        assert parent_id.startswith("t:")
        assert isinstance(child_ids, list)
        assert len(child_ids) > 0
        for cid in child_ids:
            assert isinstance(cid, str)

    q_dist = model.get_topic_distribution_for_query("another query")
    assert q_dist
    assert abs(sum(q_dist.values()) - 1.0) < 1e-5

    # Centroid and embed_query available (embedding backend)
    for tid in list(q_dist.keys())[:1]:
        cent = model.get_centroid(tid)
        assert cent is None or (isinstance(cent, (list, tuple)) and len(cent) > 0)
    q_emb = model.embed_query("test")
    assert q_emb is None or (isinstance(q_emb, (list, tuple)) and len(q_emb) > 0)
