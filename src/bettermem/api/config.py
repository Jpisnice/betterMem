from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BetterMemConfig(BaseModel):
    """Configuration for the BetterMem semantic hierarchical retriever.

    All settings apply to the intent-conditioned traversal over the
    hierarchical topic graph (second-order Markov transitions, policy scoring).
    """

    smoothing_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Interpolation weight between second-order and first-order transitions.",
    )
    max_steps: int = Field(
        default=32,
        ge=1,
        description="Maximum number of steps for intent-conditioned traversal.",
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
    index_storage_path: Optional[str] = Field(
        default=None,
        description="Optional default path for saving/loading indices.",
    )
    neighbor_top_k: int = Field(
        default=20,
        ge=1,
        description="Max number of semantic topic-topic neighbors per node (ANN kNN).",
    )
    neighbor_min_cosine: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Min cosine similarity for topic-topic semantic edges.",
    )
    clarify_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Min similarity for clarify intent (high-similarity semantic neighbor).",
    )
    hierarchy_max_depth: int = Field(
        default=3,
        ge=1,
        description="Max depth for recursive topic clustering.",
    )
    min_cluster_size: int = Field(
        default=2,
        ge=1,
        description="Min cluster size before stopping recursive split.",
    )
    transition_policy_mix_eta: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend: P = eta*P_policy + (1-eta)*P_markov; 1.0 = policy only.",
    )
    dag_tau: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, add multi-parent (DAG) edges where cos(centroid, other) >= dag_tau; None = tree only.",
    )

