from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.topic_modeling.base import BaseTopicModel


class DemoTopicModel(BaseTopicModel):
    """Lightweight semantic-hierarchical topic model for end-to-end API testing.

    Two coarse topics (t:0, t:1), one leaf each (t:0.0, t:1.0). Chunks alternate
    between them; query prior places mass on both for traversal.
    """

    def __init__(self) -> None:
        self._fit_called = False

    def fit(self, documents: Iterable[str]) -> None:
        _ = list(documents)
        self._fit_called = True

    def get_hierarchy(self) -> Mapping[str, Sequence[str]]:
        return {"t:0": ["t:0.0"], "t:1": ["t:1.0"]}

    def get_all_topic_ids(self) -> List[str]:
        return ["t:0", "t:1", "t:0.0", "t:1.0"]

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[str, float]]:
        dists: List[Mapping[str, float]] = []
        for idx, _ in enumerate(chunks):
            tid = "t:0.0" if idx % 2 == 0 else "t:1.0"
            dists.append({tid: 1.0})
        return dists

    def get_topic_keywords(self, topic_id: str, top_k: int = 10) -> List[str]:
        return [f"kw{topic_id}"] * top_k

    def get_topic_distribution_for_query(self, text: str) -> Mapping[str, float]:
        return {"t:0.0": 0.6, "t:1.0": 0.4}

    def get_centroid(self, topic_id: str):
        return None


def _load_project_readme() -> str:
    root = Path(__file__).resolve().parents[1]
    readme_path = root / "README.md"
    return readme_path.read_text(encoding="utf-8")


def test_api_end_to_end_usage(tmp_path: Path) -> None:
    """End-to-end usage test exercising the BetterMem API surface.

    Builds an index, runs queries via intent-conditioned navigation,
    inspects explanations, and verifies persistence through save/load.
    """
    corpus_document = _load_project_readme()
    query_text = "probabilistic topic transition graph"

    topic_model = DemoTopicModel()
    config = BetterMemConfig(
        max_steps=16,
        navigation_temperature=0.5,
        navigation_greedy=True,
    )

    client = BetterMem(config=config, topic_model=topic_model)

    # Build an index over the README document
    client.build_index([corpus_document])

    # Intent-conditioned navigation (default)
    results = client.query(
        query_text,
        top_k=5,
        path_trace=True,
    )
    assert results

    explanation = client.explain()
    assert explanation is not None
    assert explanation.get("strategy") == "intent_conditioned"
    assert "intent" in explanation
    assert explanation.get("prior")
    assert "paths" in explanation

    # Persist the index and configuration and reload via the API
    save_dir = tmp_path / "api_index"
    client.save(str(save_dir))

    reloaded = BetterMem.load(str(save_dir))
    reloaded_results = reloaded.query(query_text, top_k=3)
    assert reloaded_results

