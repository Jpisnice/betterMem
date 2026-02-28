from __future__ import annotations

from typing import Dict, Hashable, Mapping


def additive_smoothing(
    counts: Mapping[Hashable, int],
    alpha: float,
) -> Dict[Hashable, float]:
    """Apply additive (Laplace) smoothing and normalize to probabilities."""
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    smoothed = {k: float(v) + alpha for k, v in counts.items()}
    total = sum(smoothed.values())
    if total <= 0.0:
        return {}
    inv_total = 1.0 / total
    return {k: v * inv_total for k, v in smoothed.items()}


def dirichlet_smoothing(
    counts: Mapping[Hashable, int],
    prior_mass: float,
    vocab_size: int,
) -> Dict[Hashable, float]:
    """Dirichlet-style smoothing over a finite vocabulary.

    This is equivalent to additive smoothing with alpha = prior_mass / V
    when a uniform prior over the vocabulary of size V is assumed.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if prior_mass < 0.0:
        raise ValueError("prior_mass must be non-negative.")

    alpha = prior_mass / float(vocab_size)
    return additive_smoothing(counts, alpha)

