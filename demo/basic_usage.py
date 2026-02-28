from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.core.nodes import ChunkNode
from bettermem.topic_modeling.base import BaseTopicModel


class ToyTopicModel(BaseTopicModel):
    """Minimal topic model for console demos.

    - Creates two topics (0 and 1).
    - Alternates assignments for successive chunks so both topics appear.
    - Provides a simple two-topic prior for queries.
    """

    def fit(self, documents: Iterable[str]) -> None:
        # No heavy training; we just consume the iterator.
        _ = list(documents)

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        dists: List[Mapping[int, float]] = []
        for idx, _ in enumerate(chunks):
            if idx % 2 == 0:
                dists.append({0: 0.7, 1: 0.3})
            else:
                dists.append({0: 0.3, 1: 0.7})
        return dists

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        return [f"kw{topic_id}"] * top_k

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        # Fixed two-topic prior used to seed traversal.
        return {0: 0.6, 1: 0.4}


CORPUS_PATH = Path(__file__).resolve().parent / "attentionIsAllYouNeed.txt"


def _load_corpus() -> str:
    """Load the demo corpus (Attention Is All You Need paper)."""
    return CORPUS_PATH.read_text(encoding="utf-8", errors="replace")


def _print_results(label: str, contexts: Sequence[ChunkNode]) -> None:
    print(f"\n=== {label} ===")
    if not contexts:
        print("No contexts returned.")
        return

    for i, chunk in enumerate(contexts, start=1):
        text = str(chunk.metadata.get("text", "")) if chunk.metadata is not None else ""
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        print(f"[{i}] source={CORPUS_PATH.name}, doc_id={chunk.document_id}, position={chunk.position}")
        print(f"    {snippet}")


def main() -> None:
    corpus_document = _load_corpus()
    query_text = "attention mechanism transformer self-attention"

    config = BetterMemConfig(
        order=2,
        traversal_strategy="personalized_pagerank",
        max_steps=16,
        beam_width=2,
        exploration_factor=0.1,
    )
    topic_model = ToyTopicModel()
    client = BetterMem(config=config, topic_model=topic_model)

    print(f"Building index over {CORPUS_PATH.name} using BetterMem...")
    client.build_index([corpus_document])
    print("Index built.\n")

    # 1) Personalized PageRank (default strategy)
    ppr_results = client.query(
        query_text,
        strategy="personalized_pagerank",
        top_k=5,
    )
    _print_results("Personalized PageRank results", ppr_results)  # type: ignore[arg-type]

    # 2) Random walk with explanation trace
    rw_results = client.query(
        query_text,
        strategy="random_walk",
        steps=8,
        top_k=5,
        path_trace=True,
    )
    _print_results("Random walk results", rw_results)  # type: ignore[arg-type]

    explanation = client.explain() or {}
    print("\n--- Explanation (random_walk) ---")
    print(f"Strategy: {explanation.get('strategy')}")
    prior = explanation.get("prior", {})
    print(f"Prior topics: {list(prior.keys())[:5]}")
    paths = explanation.get("paths", [])
    if paths:
        print(f"Number of paths: {len(paths)}")
        print(f"First path (up to 10 nodes): {paths[0][:10]}")

    # 3) Beam search strategy
    beam_results = client.query(
        query_text,
        strategy="beam",
        steps=8,
        top_k=5,
        path_trace=True,
    )
    _print_results("Beam search results", beam_results)  # type: ignore[arg-type]

    # 4) Persistence: save and reload client
    save_dir = Path(__file__).resolve().parent / "demo_index"
    print(f"\nSaving index to: {save_dir}")
    client.save(str(save_dir))

    print("Reloading client from disk...")
    reloaded = BetterMem.load(str(save_dir))
    reload_results = reloaded.query(
        query_text,
        strategy="personalized_pagerank",
        top_k=3,
    )
    _print_results("Reloaded client results", reload_results)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

