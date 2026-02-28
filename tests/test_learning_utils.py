from __future__ import annotations

import math

import pytest

from bettermem.learning.entropy_control import entropy, temperature_rescale
from bettermem.learning.reinforcement import SimpleReinforcementLearner
from bettermem.learning.smoothing import additive_smoothing, dirichlet_smoothing


def test_additive_smoothing_and_normalization() -> None:
    counts = {"a": 1, "b": 1}
    probs = additive_smoothing(counts, alpha=1.0)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)
    assert all(v > 0.0 for v in probs.values())


def test_additive_smoothing_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError):
        additive_smoothing({"a": 1}, alpha=-0.1)


def test_dirichlet_smoothing_behaviour() -> None:
    counts = {"a": 1, "b": 0}
    probs = dirichlet_smoothing(counts, prior_mass=1.0, vocab_size=2)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)


def test_entropy_and_temperature_rescale() -> None:
    dist = {"a": 0.25, "b": 0.75}
    h = entropy(dist)
    assert h > 0.0

    colder = temperature_rescale(dist, temperature=0.5)
    hotter = temperature_rescale(dist, temperature=2.0)

    # Distribution remains normalized
    assert math.isclose(sum(colder.values()), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(hotter.values()), 1.0, rel_tol=1e-6)

    # More peaked at lower temperature
    assert colder["b"] > dist["b"]
    assert hotter["b"] < dist["b"]


def test_temperature_rescale_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        temperature_rescale({"a": 1.0}, temperature=0.0)


def test_simple_reinforcement_learner_adjustment_factors() -> None:
    learner = SimpleReinforcementLearner()
    trajectory = ["i", "j", "k", "k"]

    learner.update_trajectory(trajectory, reward=1.0)
    factors = learner.adjustment_factors("i", "j")

    # Positive reward should yield factor > 1.0 for visited transitions
    assert "k" in factors
    assert factors["k"] > 1.0

