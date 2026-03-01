"""
Demo: intent-conditioned navigation over the Attention Is All You Need paper.

Shows how each traversal intent (deepen, broaden, compare, apply, clarify, neutral)
steers the graph walk and retrieves different context. Each section runs a query
with that intent and prints the path taken plus a short explanation of the intent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.core.nodes import ChunkNode
from bettermem.indexing.chunker import ParagraphSentenceChunker
from bettermem.retrieval.context_aggregator import ContextWindow
from bettermem.retrieval.intent import TraversalIntent
from bettermem.topic_modeling.semantic_hierarchical import SemanticHierarchicalTopicModel


CORPUS_PATH = Path(__file__).resolve().parent / "attentionIsAllYouNeed.txt"

# Short explanations for each intent (for the demo output)
INTENT_EXPLANATIONS = {
    TraversalIntent.NEUTRAL: (
        "No structural bias; next topic is chosen by relevance to the query and "
        "continuity with the current topic. Good for general retrieval."
    ),
    TraversalIntent.DEEPEN: (
        "Prefers moving to child nodes (more specific subtopics). Use when you "
        "want more detail or a deeper explanation of the current topic."
    ),
    TraversalIntent.BROADEN: (
        "Prefers moving to the parent node (broader topic). Use when you want "
        "the big picture or how the current topic fits in context."
    ),
    TraversalIntent.COMPARE: (
        "Prefers moving to sibling nodes (same parent). Use when you want to "
        "compare alternatives or see related subtopics at the same level."
    ),
    TraversalIntent.APPLY: (
        "Prefers moving to semantically related topics (e.g. via topic-topic edges). "
        "Use when you want applications, examples, or related domains."
    ),
    TraversalIntent.CLARIFY: (
        "Prefers high-similarity semantic neighbors. Use when the current topic "
        "is unclear and you want a closely related explanation."
    ),
}


def _load_corpus() -> str:
    """Load the demo corpus (Attention Is All You Need paper)."""
    return CORPUS_PATH.read_text(encoding="utf-8", errors="replace")


def _print_results(
    label: str,
    contexts: Sequence[ContextWindow],
    max_items: int = 3,
) -> None:
    print(f"  Top results:")
    if not contexts:
        print("    (none)")
        return
    chunks_shown = 0
    for w in contexts:
        if chunks_shown >= max_items:
            break
        for chunk in w.chunks:
            if chunks_shown >= max_items:
                break
            text = str(chunk.metadata.get("text", "")) if chunk.metadata else ""
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            snippet = "".join(c if ord(c) < 128 else "?" for c in snippet)
            chunks_shown += 1
            print(f"    [{chunks_shown}] {snippet}")


def _print_explanation(explanation: dict) -> None:
    """Print path and prior from the last query explanation."""
    path = explanation.get("path") or []
    prior = explanation.get("prior", {})
    print(f"  Path ({len(path)} nodes): {path[:8]}{'...' if len(path) > 8 else ''}")
    if prior:
        top = sorted(prior.items(), key=lambda x: -x[1])[:3]
        print(f"  Prior (top): {[(n, round(p, 3)) for n, p in top]}")


def main() -> None:
    corpus_document = _load_corpus()
    base_query = "attention mechanism transformer training"

    config = BetterMemConfig.debug_preset()
    topic_model = SemanticHierarchicalTopicModel(
        n_coarse=10,
        n_fine_per_coarse=4,
        random_state=42,
    )
    chunker = ParagraphSentenceChunker(max_tokens=200)
    client = BetterMem(config=config, topic_model=topic_model)

    print(f"Building index over {CORPUS_PATH.name} using BetterMem...")
    client.build_index([corpus_document], chunker=chunker)
    print("Index built.")
    print("Discovered hierarchy (coarse -> sub-topics, sample keywords):")
    for coarse_id, sub_ids in topic_model.get_hierarchy().items():
        kws = topic_model.get_topic_keywords(sub_ids[0], top_k=4) if sub_ids else []
        print(f"  Coarse {coarse_id} -> {sub_ids}: e.g. {kws}")
    print()

    # Showcase each intent type with the same base query
    intents_to_show = [
        TraversalIntent.NEUTRAL,
        TraversalIntent.DEEPEN,
        TraversalIntent.BROADEN,
        TraversalIntent.COMPARE,
        TraversalIntent.APPLY,
        TraversalIntent.CLARIFY,
    ]

    for intent in intents_to_show:
        title = f"Intent: {intent.value.upper()}"
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(f"  Explanation: {INTENT_EXPLANATIONS[intent]}")
        print()

        results = client.query(
            base_query,
            intent=intent,
            top_k=5,
            path_trace=True,
            diversity=False,
        )
        explanation = client.explain() or {}
        _print_explanation(explanation)
        print()
        _print_results(f"Results ({intent.value})", results, max_items=3)

    # Persistence
    save_dir = Path(__file__).resolve().parent / "demo_index"
    print("\n" + "=" * 60)
    print("Persistence")
    print("=" * 60)
    print(f"Saving index to: {save_dir}")
    client.save(str(save_dir))
    print("Reloading client from disk...")
    reloaded = BetterMem.load(str(save_dir))
    reload_results = reloaded.query(base_query, top_k=3, diversity=False)
    _print_results("Reloaded client (neutral)", reload_results, max_items=2)


if __name__ == "__main__":
    main()
