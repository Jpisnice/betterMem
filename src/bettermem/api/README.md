# API

User-facing interface and configuration.

## Modules

- **`config.py`**: **BetterMemConfig** (Pydantic) for the semantic hierarchical retriever:
  - Transition: `smoothing_lambda`; traversal: `max_steps`, `transition_policy_mix_eta`.
  - Navigation: `navigation_alpha`, `navigation_beta`, `navigation_gamma`, `navigation_delta`, `navigation_backtrack_penalty`, `navigation_novelty_bonus`, `navigation_prior_weight`, `navigation_temperature`, `navigation_greedy`; `clarify_similarity_threshold`.
  - Graph/indexing: `neighbor_top_k`, `neighbor_min_cosine`, `topic_chunk_top_m`, `topic_chunk_min_prob`, `topic_chunk_ancestor_decay`; `hierarchy_max_depth`, `min_cluster_size`, `dag_tau`.
  - Rerank: `rerank_weight_topic`, `rerank_weight_cosine`, `rerank_weight_bm25`.
  - **debug_preset()**: short deterministic runs (e.g. max_steps=5, greedy, no Markov blend) for demos.
- **`client.py`**: **BetterMem**:
  - **build_index(corpus, chunker=...)**: build graph and transition model via CorpusIndexer (default topic model and chunker if not provided).
  - **query(text, steps=..., top_k=..., diversity=..., path_trace=..., intent=...)**: intent-conditioned walk, then project visit counts to chunk space (with per-topic normalization), rerank, and select contexts.
  - **explain(include_embeddings=...)**: last query’s intent, prior, path, path_steps, chunks_along_path.
  - **save(path)** / **load(path)**: persist and reload via `bettermem.storage.persistence`. Loaded client has no topic model (queries use uniform topic prior when needed).

## How it fits

- Entry point for applications: configure, build index, query, explain, save/load. Composes core, topic_modeling, indexing, retrieval, learning (for transition smoothing), and storage.
