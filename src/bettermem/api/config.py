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
    topic_chunk_top_m: int = Field(
        default=2,
        ge=1,
        description="Max number of leaf topics to attach per chunk (sparse topic–chunk edges).",
    )
    topic_chunk_min_prob: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Min topic probability to attach a topic–chunk edge.",
    )
    topic_chunk_ancestor_decay: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Decay factor for attaching chunk to ancestors of best leaf (hierarchical recall).",
    )
    rerank_weight_topic: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for topic-projected score in chunk reranking (hybrid with cosine).",
    )
    rerank_weight_cosine: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for query–chunk cosine similarity in chunk reranking.",
    )
    rerank_weight_bm25: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 in chunk reranking; 0 = disable BM25.",
    )

    @classmethod
    def debug_preset(cls) -> "BetterMemConfig":
        """Preset for debugging intent alignment: short deterministic runs, no Markov blend.

        Use with single-document demos and set diversity=False in query().
        """
        return cls(
            max_steps=5,
            navigation_greedy=True,
            navigation_temperature=0.35,
            navigation_gamma=1.2,
            navigation_novelty_bonus=0.05,
            navigation_prior_weight=0.3,
            neighbor_top_k=6,
            neighbor_min_cosine=0.35,
            clarify_similarity_threshold=0.8,
            transition_policy_mix_eta=1.0,
            dag_tau=None,
        )

