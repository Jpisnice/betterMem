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
        description="Default traversal strategy for queries.",
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
    topic_model_backend: Literal["bertopic", "lda"] = Field(
        default="bertopic",
        description="Default topic modeling backend to use for indexing and query priors.",
    )
    # Optional reserved fields for future extensions
    index_storage_path: Optional[str] = Field(
        default=None,
        description="Optional default path for saving/loading indices.",
    )

