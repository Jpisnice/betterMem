from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from bettermem.api.client import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.core.nodes import ChunkNode
from bettermem.topic_modeling.base import BaseTopicModel

# Common stopwords so topics are built from content words in the file.
_STOP = frozenset(
    "a an the and or but in on at to for of with by from as is was are were be been being have has had do does did will would could should may might must can this that these those it its i we you he she they".split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenize; keep only alphabetic tokens of length >= 2."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP]


class ContentTopicModel(BaseTopicModel):
    """Topic model that derives topics from the same corpus file.

    - fit(): builds vocabulary from the document(s), then forms N topics by
      partitioning the top frequent content words. Each topic is a set of
      keywords from the file.
    - transform() / get_topic_distribution_for_query(): assign mass to topics
      by counting how many of each topic's keywords appear in the chunk/query.
    So all topics and assignments come from the same file's context.
    """

    def __init__(self, num_topics: int = 5, words_per_topic: int = 25) -> None:
        self.num_topics = max(1, num_topics)
        self.words_per_topic = max(1, words_per_topic)
        self._topic_keywords: List[List[str]] = []

    def fit(self, documents: Iterable[str]) -> None:
        all_tokens: List[str] = []
        for doc in documents:
            all_tokens.extend(_tokenize(doc))
        counts = Counter(all_tokens)
        # Top words by frequency, excluding stopwords (already filtered in _tokenize).
        top_words = [w for w, _ in counts.most_common(self.num_topics * self.words_per_topic)]
        self._topic_keywords = []
        for i in range(self.num_topics):
            start = i * self.words_per_topic
            end = min(start + self.words_per_topic, len(top_words))
            self._topic_keywords.append(top_words[start:end])
        # Ensure we have exactly num_topics (pad with first topic's words if needed).
        while len(self._topic_keywords) < self.num_topics:
            self._topic_keywords.append(self._topic_keywords[0][: self.words_per_topic])

    def _text_to_topic_scores(self, text: str) -> Mapping[int, float]:
        tokens = set(_tokenize(text))
        if not tokens:
            # Uniform over topics if no tokens.
            n = len(self._topic_keywords)
            return {i: 1.0 / n for i in range(n)}
        scores: List[float] = []
        for kw_list in self._topic_keywords:
            overlap = sum(1 for w in kw_list if w in tokens)
            scores.append(overlap + 0.1)  # smoothing
        total = sum(scores)
        if total <= 0:
            n = len(self._topic_keywords)
            return {i: 1.0 / n for i in range(n)}
        return {i: scores[i] / total for i in range(len(scores))}

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        return [self._text_to_topic_scores(chunk) for chunk in chunks]

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        if 0 <= topic_id < len(self._topic_keywords):
            return self._topic_keywords[topic_id][:top_k]
        return []

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        return self._text_to_topic_scores(text)


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
        traversal_strategy="personalized_pagerank",
        max_steps=16,
        beam_width=2,
        exploration_factor=0.1,
    )
    topic_model = ContentTopicModel(num_topics=5, words_per_topic=25)
    client = BetterMem(config=config, topic_model=topic_model)

    print(f"Building index over {CORPUS_PATH.name} using BetterMem...")
    client.build_index([corpus_document])
    print("Index built.")
    print("Topics derived from this file (sample keywords per topic):")
    for tid in range(min(5, topic_model.num_topics)):
        kws = topic_model.get_topic_keywords(tid, top_k=8)
        print(f"  Topic {tid}: {kws}")
    print()

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

