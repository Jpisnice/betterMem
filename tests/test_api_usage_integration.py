from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.topic_modeling.base import BaseTopicModel


class DemoTopicModel(BaseTopicModel):
    """Lightweight semantic-hierarchical topic model for end-to-end API testing.

    Two coarse topics, one subtopic each (encoded 0 and 100). Chunks alternate
    between them; query prior places mass on both for traversal.
    """

    def __init__(self) -> None:
        self._fit_called = False

    def fit(self, documents: Iterable[str]) -> None:
        _ = list(documents)
        self._fit_called = True

    def get_hierarchy(self) -> Mapping[int, Sequence[int]]:
        return {0: [0], 1: [100]}  # coarse 0 -> sub 0, coarse 1 -> sub 100

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        dists: List[Mapping[int, float]] = []
        for idx, _ in enumerate(chunks):
            encoded = 0 if idx % 2 == 0 else 100
            dists.append({encoded: 1.0})
        return dists

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        return [f"kw{topic_id}"] * top_k

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        return {0: 0.6, 100: 0.4}


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

