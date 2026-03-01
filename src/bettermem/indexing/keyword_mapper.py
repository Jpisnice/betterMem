from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping


class KeywordToTopicMapper:
    """Lightweight keyword-to-topic prior estimator.

    This implementation approximates

        P(t | q) ∝ Σ_{w in q} TFIDF(w, t)

    using simple term frequencies over topic keyword lists. A more
    sophisticated TF-IDF or embedding-based scheme can be introduced
    later behind the same API.
    """

    def __init__(self) -> None:
        # topic_id (node id str) -> Counter(term -> weight)
        self._topic_term_weights: Dict[str, Counter[str]] = {}

    def fit(self, topic_keywords: Mapping[str, Iterable[str]]) -> None:
        """Register keywords for each topic (key = topic/node id string)."""
        self._topic_term_weights.clear()
        for tid, terms in topic_keywords.items():
            self._topic_term_weights[str(tid)] = Counter(str(t) for t in terms)

    def topic_prior_from_query(self, text: str) -> Mapping[str, float]:
        """Compute an unnormalized topic prior from query terms."""
        if not self._topic_term_weights:
            return {}

        terms = [t.lower() for t in text.split()]
        scores: Dict[str, float] = defaultdict(float)
        for tid, counter in self._topic_term_weights.items():
            for term in terms:
                if term in counter:
                    scores[tid] += float(counter[term])

        total = sum(scores.values())
        if total <= 0.0:
            return {}
        inv_total = 1.0 / total
        return {tid: s * inv_total for tid, s in scores.items()}

