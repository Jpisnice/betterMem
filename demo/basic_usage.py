from __future__ import annotations

from pathlib import Path
from typing import Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.core.nodes import ChunkNode
from bettermem.indexing.structure_aware_chunker import StructureAwareChunker
from bettermem.topic_modeling.semantic_hierarchical import SemanticHierarchicalTopicModel


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
        # Avoid Windows console encoding errors: keep ASCII + replace others with '?'.
        snippet = "".join(c if ord(c) < 128 else "?" for c in snippet)
        print(f"[{i}] source={CORPUS_PATH.name}, doc_id={chunk.document_id}, position={chunk.position}")
        print(f"    {snippet}")


def main() -> None:
    corpus_document = _load_corpus()
    query_text = "attention mechanism transformer self-attention"

    config = BetterMemConfig(
        order=2,
        max_steps=16,
        navigation_temperature=0.8,
        navigation_greedy=False,
    )
    topic_model = SemanticHierarchicalTopicModel(
        n_coarse=5,
        n_fine_per_coarse=3,
        random_state=42,
    )
    chunker = StructureAwareChunker(window_size=200, overlap=50)
    client = BetterMem(config=config, topic_model=topic_model)

    print(f"Building index over {CORPUS_PATH.name} using BetterMem...")
    client.build_index([corpus_document], chunker=chunker)
    print("Index built.")
    print("Discovered hierarchy (coarse -> sub-topics, sample keywords):")
    for coarse_id, sub_ids in topic_model.get_hierarchy().items():
        kws = topic_model.get_topic_keywords(sub_ids[0], top_k=5) if sub_ids else []
        print(f"  Coarse {coarse_id} -> {sub_ids}: e.g. {kws}")
    print()

    # 1) Intent-conditioned navigation (default)
    results = client.query(
        query_text,
        top_k=5,
    )
    _print_results("Intent-conditioned results", results)  # type: ignore[arg-type]

    # 2) Same query with path trace to inspect navigation
    results_with_trace = client.query(
        query_text,
        steps=8,
        top_k=5,
        path_trace=True,
    )
    _print_results("Intent-conditioned (path trace)", results_with_trace)  # type: ignore[arg-type]

    explanation = client.explain() or {}
    print("\n--- Explanation ---")
    print(f"Strategy: {explanation.get('strategy')}")
    print(f"Intent: {explanation.get('intent')}")
    prior = explanation.get("prior", {})
    print(f"Prior topics: {list(prior.keys())[:5]}")
    paths = explanation.get("paths", [])
    if paths:
        print(f"Number of paths: {len(paths)}")
        print(f"First path (up to 10 nodes): {paths[0][:10]}")

    # 4) Persistence: save and reload client
    save_dir = Path(__file__).resolve().parent / "demo_index"
    print(f"\nSaving index to: {save_dir}")
    client.save(str(save_dir))

    print("Reloading client from disk...")
    reloaded = BetterMem.load(str(save_dir))
    reload_results = reloaded.query(query_text, top_k=3)
    _print_results("Reloaded client results", reload_results)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

