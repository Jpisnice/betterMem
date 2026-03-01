Public API layer
================

The `bettermem.api` package exposes the **user-facing interface** and
configuration for the library.

Modules
-------

- `config.py`:
  - Defines `BetterMemConfig` (Pydantic) for the **semantic hierarchical** retriever only:
    - `smoothing_lambda`, `max_steps` (transition and traversal).
    - Navigation policy: `navigation_alpha`, `navigation_beta`, `navigation_gamma`,
      `navigation_delta`, `navigation_backtrack_penalty`, `navigation_novelty_bonus`,
      `navigation_prior_weight`, `navigation_temperature`, `navigation_greedy`.
    - Optional `index_storage_path`.
  - Single design: intent-conditioned traversal over the hierarchical topic graph.
- `client.py`:
  - `BetterMem` facade:
    - `build_index(corpus, chunker=...)`: build hierarchical topic graph and second-order transitions.
    - `query(text, steps=..., top_k=..., diversity=..., path_trace=..., intent=...)`: intent-conditioned retrieval.
    - `explain(include_embeddings=...)`: structured explanation of last query (path, prior, intent).
    - `save(path)` / `load(path)`: persist and reload via `bettermem.storage.persistence`.

How it fits into the system
---------------------------

- Primary entry point for external callers:
  - Import `BetterMem` and `BetterMemConfig`.
  - Configure semantic hierarchical traversal (navigation knobs, max_steps).
  - Build indices, run queries, retrieve explanations.
- Internally, the API layer composes all other subpackages:
  `core`, `topic_modeling`, `indexing`, `learning`, `retrieval`,
  and `storage`.

