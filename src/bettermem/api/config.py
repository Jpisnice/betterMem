from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BetterMemConfig(BaseModel):
    """Configuration for the BetterMem probabilistic topic transition graph retriever."""

    order: Literal[1, 2] = Field(
        default=2,
        description="Markov order for transitions. 2 enables second-order topic transitions.",
    )
    smoothing_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Interpolation weight between second-order and first-order transitions.",
    )
    beam_width: int = Field(
        default=10,
        ge=1,
        description="Beam width for beam-search-based traversal.",
    )
    max_steps: int = Field(
        default=32,
        ge=1,
        description="Maximum number of traversal steps for random walks / PageRank iterations.",
    )
    transition_prune_threshold: float = Field(
        default=1e-6,
        ge=0.0,
        description="Drop transitions with probability below this threshold during pruning.",
    )
    traversal_strategy: Literal["beam", "random_walk", "personalized_pagerank"] = Field(
        default="personalized_pagerank",
        description="Deprecated; intent-conditioned navigation is used. Kept for backward compatibility.",
    )
    navigation_alpha: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for relevance term cos(μ_k, q) in intent-conditioned policy.",
    )
    navigation_beta: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for continuity term cos(μ_k, μ_i) in intent-conditioned policy.",
    )
    navigation_gamma: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for structural fit R_intent(i,k) in intent-conditioned policy.",
    )
    navigation_delta: float = Field(
        default=0.5,
        ge=0.0,
        description="Repetition penalty (quadratic in visit count) in intent-conditioned policy.",
    )
    navigation_backtrack_penalty: float = Field(
        default=5.0,
        ge=0.0,
        description="Penalty for moving back to the previous node (prevents parent-child oscillation).",
    )
    navigation_novelty_bonus: float = Field(
        default=0.3,
        ge=0.0,
        description="Bonus added for candidates not yet visited (encourages exploration).",
    )
    navigation_prior_weight: float = Field(
        default=0.2,
        ge=0.0,
        description="Weight for the topic prior P(topic|query) in candidate scoring.",
    )
    navigation_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature for next-step selection (lower = greedier).",
    )
    navigation_greedy: bool = Field(
        default=False,
        description="If True, always take argmax next step; else sample from policy.",
    )
    entropy_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of entropy-based regularization in scoring.",
    )
    exploration_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Controls stochastic exploration in next-step sampling.",
    )
    # Optional reserved fields for future extensions
    index_storage_path: Optional[str] = Field(
        default=None,
        description="Optional default path for saving/loading indices.",
    )

