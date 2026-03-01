Public API layer
================

The `bettermem.api` package exposes the **user-facing interface** and
configuration for the library.

Modules
-------

- `config.py`:
  - Defines `BetterMemConfig` (Pydantic model) with knobs such as:
    - `order` (1 or 2), `smoothing_lambda`, `beam_width`,
      `max_steps`, `transition_prune_threshold`.
    - `traversal_strategy` (e.g. beam, random walk, personalized PageRank).
    - `entropy_penalty`, `exploration_factor`.
    - Navigation and traversal knobs (no separate topic backend; semantic hierarchical model is used by default).
  - This captures the high-level behaviour of the system in a validated,
    serializable form.
- `client.py`:
  - Implements the `BetterMem` facade class:
    - `build_index(corpus)`: runs the indexing pipeline
      (`Chunker`, `CorpusIndexer`, topic modeling) to construct the graph
      and transition model.
    - `query(text, ...)`: executes the query pipeline:
      `QueryInitializer` → `TraversalEngine` → `QueryScorer` →
      `ContextAggregator`, returning ranked chunks.
    - `explain()`: returns a structured explanation of the last query,
      including priors, chosen strategy, and traversal artifacts.
    - `save(path)` / `load(path)`: persist and reload indices via
      `bettermem.storage.persistence`.

How it fits into the system
---------------------------

- This is the primary entry point for external callers:
  - Import `BetterMem` and `BetterMemConfig`.
  - Configure traversal and topic modeling backends.
  - Build indices, run queries, and retrieve explanations.
- Internally, the API layer composes all other subpackages:
  `core`, `topic_modeling`, `indexing`, `learning`, `retrieval`,
  and `storage`.

