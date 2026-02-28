from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.topic_modeling.base import BaseTopicModel


class DemoTopicModel(BaseTopicModel):
    """Lightweight topic model used for end-to-end API testing.

    It creates two topics and assigns chunks alternately to topic 0 and 1,
    while the query distribution always places non-zero mass on both topics.
    This ensures all traversal strategies have a sensible prior/state to work with
    without requiring heavy external dependencies.
    """

    def __init__(self) -> None:
        self._fit_called = False

    def fit(self, documents: Iterable[str]) -> None:
        _ = list(documents)
        self._fit_called = True

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        dists: List[Mapping[int, float]] = []
        for idx, _ in enumerate(chunks):
            topic_id = idx % 2
            dists.append({topic_id: 1.0})
        return dists

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        return [f"kw{topic_id}"] * top_k

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        # Simple fixed two-topic prior used for initialization
        # Topic 0 gets slightly higher mass than topic 1.
        return {0: 0.6, 1: 0.4}


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
        order=2,
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

