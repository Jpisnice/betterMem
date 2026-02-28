"""Traversal intent and heuristic classifier for intent-conditioned navigation."""

from __future__ import annotations

import re
from enum import Enum
from typing import Sequence


class TraversalIntent(str, Enum):
    """User intent for graph navigation (textbook-browsing style)."""

    DEEPEN = "deepen"  # "Explain this more" -> move to child
    BROADEN = "broaden"  # "What is the big picture?" -> move to parent
    COMPARE = "compare"  # "How is this different?" -> move to sibling
    APPLY = "apply"  # "Where is this used?" -> move to related domain
    CLARIFY = "clarify"  # "I didn't understand that" -> high-similarity neighbor
    NEUTRAL = "neutral"  # Relevance only; no structural bias


# Heuristic phrase sets (lowercased) per intent
_DEEPEN_PHRASES = (
    "explain more", "go deeper", "more detail", "in depth", "elaborate",
    "expand on", "tell me more", "what exactly", "how does it work",
    "break down", "dive into", "drill down",
)
_BROADEN_PHRASES = (
    "big picture", "overview", "high level", "in general", "broaden",
    "context", "how does this fit", "wider", "summary", "summarize",
)
_COMPARE_PHRASES = (
    "compare", "comparison", "different", "difference", "vs ", " versus ",
    "alternative", "alternatives", "other approach", "similar", "contrast",
)
_APPLY_PHRASES = (
    "where is this used", "application", "apply", "example", "use case",
    "in practice", "real world", "how to use", "when to use",
)
_CLARIFY_PHRASES = (
    "didn't understand", "don't understand", "unclear", "clarify",
    "what do you mean", "confused", "simpler", "simple explanation",
)


def classify_intent_heuristic(text: str) -> TraversalIntent:
    """Classify query text to traversal intent using keyword/phrase heuristics.

    Checks query (lowercased) for intent-indicative phrases. First match wins
    in order: deepen, broaden, compare, apply, clarify; otherwise NEUTRAL.
    """
    if not text or not text.strip():
        return TraversalIntent.NEUTRAL

    lower = text.lower().strip()
    # Normalize whitespace for phrase matching
    lower_norm = re.sub(r"\s+", " ", lower)

    for phrase in _DEEPEN_PHRASES:
        if phrase in lower_norm:
            return TraversalIntent.DEEPEN
    for phrase in _BROADEN_PHRASES:
        if phrase in lower_norm:
            return TraversalIntent.BROADEN
    for phrase in _COMPARE_PHRASES:
        if phrase in lower_norm:
            return TraversalIntent.COMPARE
    for phrase in _APPLY_PHRASES:
        if phrase in lower_norm:
            return TraversalIntent.APPLY
    for phrase in _CLARIFY_PHRASES:
        if phrase in lower_norm:
            return TraversalIntent.CLARIFY

    return TraversalIntent.NEUTRAL
