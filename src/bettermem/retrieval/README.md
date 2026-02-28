Retrieval layer
===============

The `bettermem.retrieval` package turns **graph traversals** into ranked
context chunks and explanations.

Modules
-------

- `query_initializer.py`:
  - Combines topic-model output and optional keyword mapping to produce a
    **topic prior** \(P_0(t \mid q)\) over topic nodes.
  - Chooses an initial second-order **state**
    \((v_{t-2}, v_{t-1})\) as the top-2 topics for the query.
- `scorer.py`:
  - `QueryScorer` aggregates contributions from:
    - Beam search paths: converts cumulative log-probabilities into
      weights and updates node scores.
    - Visit counts: from random walks or PageRank.
  - Conceptually approximates
    \(\text{Score}(c) = \sum_{\text{paths}} P(\text{path} \to c)\).
- `context_aggregator.py`:
  - Filters scores down to **chunk nodes** and selects:
    - Top-\(k\) chunks by score.
    - Optionally enforces document-level diversity.

Math and concepts
-----------------

Given:

- A traversal result (paths or visit counts) over nodes in
  \(G = (V, E)\).
- Node types \(V = T \cup C\) (topics and chunks).

The retrieval layer:

1. Converts path or visit statistics into a scalar **score** per node,
   with chunk nodes as the primary targets.
2. Aggregates and ranks chunk scores to form the final retrieval set.

How it fits into the system
---------------------------

- Used by `BetterMem.query` after the traversal engine produces paths, visit
  counts, or PageRank scores.
- Works directly with:
  - The **graph** (`bettermem.core.graph.Graph`) to resolve node types and
    metadata.
  - The **query initializer** to ground traversal in query-specific topic
    priors.
- Feeds the selected chunks back to the API layer, along with optional
  explanation data (paths, transitions, topic transitions).

