from __future__ import annotations

import math
from typing import Dict, Hashable, Mapping


def entropy(dist: Mapping[Hashable, float]) -> float:
    """Compute Shannon entropy H(P) in nats."""
    h = 0.0
    for p in dist.values():
        if p > 0.0:
            h -= p * math.log(p)
    return h


def temperature_rescale(
    dist: Mapping[Hashable, float],
    temperature: float,
) -> Dict[Hashable, float]:
    """Rescale a distribution with a temperature parameter."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if not dist:
        return {}

    # Softmax over log-probabilities scaled by 1 / temperature
    scaled = {k: math.log(max(p, 1e-12)) / temperature for k, p in dist.items()}
    max_log = max(scaled.values())
    exp_values = {k: math.exp(v - max_log) for k, v in scaled.items()}
    total = sum(exp_values.values())
    if total <= 0.0:
        return {}
    inv_total = 1.0 / total
    return {k: v * inv_total for k, v in exp_values.items()}

