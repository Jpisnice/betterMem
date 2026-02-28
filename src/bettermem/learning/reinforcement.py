from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Mapping, Tuple

from bettermem.core.nodes import NodeId

State = Tuple[NodeId, NodeId]


class SimpleReinforcementLearner:
    """Minimal reinforcement hook for reweighting transitions.

    This learner accumulates average rewards for observed transitions
    (i, j -> k). It can then be used to compute a multiplicative
    adjustment over base transition probabilities.
    """

    def __init__(self) -> None:
        # (i, j, k) -> sum_reward, count
        self._sum_rewards: DefaultDict[Tuple[NodeId, NodeId, NodeId], float] = defaultdict(
            float
        )
        self._counts: DefaultDict[Tuple[NodeId, NodeId, NodeId], int] = defaultdict(int)

    def update_trajectory(
        self,
        trajectory: Iterable[NodeId],
        reward: float,
    ) -> None:
        """Record a trajectory and its scalar reward."""
        nodes = list(trajectory)
        if len(nodes) < 3:
            return
        for i, j, k in zip(nodes[:-2], nodes[1:-1], nodes[2:]):
            key = (i, j, k)
            self._sum_rewards[key] += reward
            self._counts[key] += 1

    def adjustment_factors(
        self,
        i: NodeId,
        j: NodeId,
    ) -> Mapping[NodeId, float]:
        """Return multiplicative adjustment factors for successors of (i, j)."""
        factors: Dict[NodeId, float] = {}
        for (si, sj, k), s in self._sum_rewards.items():
            if si == i and sj == j:
                c = self._counts[(si, sj, k)]
                avg_reward = s / max(c, 1)
                # Map average reward to a positive factor; 1.0 means neutral.
                factors[k] = max(0.0, 1.0 + avg_reward)
        return factors

