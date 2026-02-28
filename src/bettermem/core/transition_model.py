from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from .nodes import NodeId
from bettermem.learning import smoothing


TripleKey = tuple[NodeId, NodeId]


@dataclass
class TransitionDistribution:
    """Container for a transition distribution over next nodes."""

    probs: Dict[NodeId, float]

    def is_empty(self) -> bool:
        return not self.probs


class TransitionModel:
    """Second-order Markov transition model over graph nodes.

    The model estimates conditional probabilities of the form
    P(k | i, j) where i, j, k are node identifiers, based on
    observed sequences of nodes.
    """

    def __init__(
        self,
        smoothing_lambda: float = 0.7,
        additive_alpha: float = 0.0,
    ) -> None:
        if not (0.0 <= smoothing_lambda <= 1.0):
            raise ValueError("smoothing_lambda must be in [0, 1].")

        self.smoothing_lambda = float(smoothing_lambda)
        self.additive_alpha = float(additive_alpha)

        # (i, j) -> {k: count_ijk}
        self._second_order_counts: DefaultDict[TripleKey, Dict[NodeId, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # j -> {k: count_jk}
        self._first_order_counts: DefaultDict[NodeId, Dict[NodeId, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Cached normalized distributions
        self._second_order_probs: Dict[TripleKey, Dict[NodeId, float]] = {}
        self._first_order_probs: Dict[NodeId, Dict[NodeId, float]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        sequences: Iterable[Sequence[NodeId]],
        *,
        structural_groups_per_seq: Optional[Iterable[Sequence[int]]] = None,
        structural_penalty: float = 0.3,
    ) -> None:
        """Estimate second-order and first-order counts from sequences.

        Parameters
        ----------
        sequences:
            Iterable of node-id sequences, typically at the topic level.
        structural_groups_per_seq:
            Optional parallel iterable: one sequence of structural group ids per
            position for each sequence. Transitions that cross section boundaries
            (different group) are weighted by structural_penalty.
        structural_penalty:
            Weight applied to counts when transition crosses a section boundary.
        """
        self._second_order_counts.clear()
        self._first_order_counts.clear()
        self._second_order_probs.clear()
        self._first_order_probs.clear()

        groups_iter = iter(structural_groups_per_seq) if structural_groups_per_seq else None

        for seq in sequences:
            if len(seq) < 2:
                continue

            groups: Optional[Sequence[int]] = None
            if groups_iter is not None:
                try:
                    groups = next(groups_iter)
                except StopIteration:
                    pass
            if groups is not None and len(groups) != len(seq):
                groups = None

            # First-order counts
            for idx, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
                w = 1.0
                if groups is not None and idx + 1 < len(groups):
                    if groups[idx] != groups[idx + 1]:
                        w = structural_penalty
                self._first_order_counts[a][b] += w

            # Second-order counts
            if len(seq) < 3:
                continue
            for idx, (i, j, k) in enumerate(zip(seq[:-2], seq[1:-1], seq[2:])):
                w = 1.0
                if groups is not None and idx + 2 < len(groups):
                    if groups[idx + 1] != groups[idx + 2]:
                        w = structural_penalty
                key = (i, j)
                self._second_order_counts[key][k] += w

    # ------------------------------------------------------------------
    # Probability distributions
    # ------------------------------------------------------------------
    def _normalize_counts(
        self,
        counts: Mapping[NodeId, int],
    ) -> Dict[NodeId, float]:
        if self.additive_alpha > 0.0:
            return smoothing.additive_smoothing(counts, self.additive_alpha)

        total = sum(counts.values())
        if total <= 0:
            return {}
        inv_total = 1.0 / float(total)
        return {k: v * inv_total for k, v in counts.items()}

    def _get_second_order_probs(self, i: NodeId, j: NodeId) -> Dict[NodeId, float]:
        key = (i, j)
        if key not in self._second_order_probs:
            counts = self._second_order_counts.get(key, {})
            self._second_order_probs[key] = self._normalize_counts(counts)
        return self._second_order_probs[key]

    def _get_first_order_probs(self, j: NodeId) -> Dict[NodeId, float]:
        if j not in self._first_order_probs:
            counts = self._first_order_counts.get(j, {})
            self._first_order_probs[j] = self._normalize_counts(counts)
        return self._first_order_probs[j]

    def transition_prob(self, i: NodeId, j: NodeId) -> TransitionDistribution:
        """Return the blended distribution P(k | i, j).

        The distribution is computed as:

            P = λ P2 + (1 - λ) P1

        where:
            - P2 is the second-order distribution P(k | i, j)
            - P1 is the first-order distribution P(k | j)
        """
        p2 = self._get_second_order_probs(i, j)
        p1 = self._get_first_order_probs(j)

        if not p2 and not p1:
            return TransitionDistribution(probs={})

        lam = self.smoothing_lambda
        if not p2:
            blended = dict(p1)
        elif not p1:
            blended = dict(p2)
        else:
            blended: Dict[NodeId, float] = {}
            keys = set(p2.keys()) | set(p1.keys())
            for k in keys:
                blended[k] = lam * p2.get(k, 0.0) + (1.0 - lam) * p1.get(k, 0.0)

        # Ensure numerical stability (optional renormalization)
        total = sum(blended.values())
        if total > 0.0 and abs(total - 1.0) > 1e-9:
            inv_total = 1.0 / total
            blended = {k: v * inv_total for k, v in blended.items()}

        return TransitionDistribution(probs=blended)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_next(
        self,
        i: NodeId,
        j: NodeId,
        *,
        exploration_factor: float = 0.0,
        rng: Optional["random.Random"] = None,
    ) -> Optional[NodeId]:
        """Sample the next node k ~ P(k | i, j) with optional exploration.

        Parameters
        ----------
        i, j:
            Previous two node identifiers.
        exploration_factor:
            With probability `exploration_factor`, sample uniformly from
            the support instead of using the weighted distribution.
        rng:
            Optional random number generator compatible with random.Random.
        """
        import random

        rng = rng or random
        dist = self.transition_prob(i, j).probs
        if not dist:
            return None

        nodes = list(dist.keys())

        if exploration_factor > 0.0 and rng.random() < exploration_factor:
            return rng.choice(nodes)

        weights = [dist[n] for n in nodes]
        # random.choices is available from Python 3.6 onwards
        return rng.choices(nodes, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Persistence (wired in Stage 8)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, MutableMapping]:
        """Return a serializable representation of the model.

        Actual disk persistence is implemented in the storage layer.
        """
        return {
            "smoothing_lambda": self.smoothing_lambda,
            "second_order_counts": {
                f"{i}|{j}": dict(counts)
                for (i, j), counts in self._second_order_counts.items()
            },
            "first_order_counts": {
                j: dict(counts) for j, counts in self._first_order_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TransitionModel":
        model = cls(smoothing_lambda=float(data.get("smoothing_lambda", 0.7)))

        second_raw = data.get("second_order_counts", {}) or {}
        for key, counts in second_raw.items():  # type: ignore[assignment]
            i, j = key.split("|", 1)
            for k, c in counts.items():  # type: ignore[assignment]
                model._second_order_counts[(i, j)][k] = int(c)

        first_raw = data.get("first_order_counts", {}) or {}
        for j, counts in first_raw.items():  # type: ignore[assignment]
            for k, c in counts.items():  # type: ignore[assignment]
                model._first_order_counts[j][k] = int(c)

        # Clear probability caches; they will be recomputed lazily.
        model._second_order_probs.clear()
        model._first_order_probs.clear()

        return model

