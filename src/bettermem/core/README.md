Core module
===========

The `bettermem.core` package contains the **graph and traversal primitives**
on which the rest of the library is built.

- `nodes.py`: defines node types (`TopicNode`, `ChunkNode`, optional
  `KeywordNode`) and IDs (`t:i`, `c:j`, `k:term`). Conceptually, the node set
  is \(V = T \cup C\).
- `edges.py`: defines directed weighted edges with kinds
  (topic→topic, topic→chunk, chunk→chunk), forming the edge set \(E\).
- `graph.py`: maintains a sparse adjacency structure
  \(A_{ij} = w_{ij}\) via `neighbors[i][j] = weight`, and supports adding
  nodes/edges, normalizing outgoing weights, pruning small edges, and
  serializing/deserializing the graph.
- `transition_model.py`: implements the **second-order Markov model**
  \(P(k \mid i,j)\) with counts
  \(\text{count}(i,j,k)\), smoothed and blended as
  \(P = \lambda P_2 + (1 - \lambda) P_1\) where
  \(P_2(k \mid i,j) = \text{count}(i,j,k)/\sum_k \text{count}(i,j,k)\) and
  \(P_1(k \mid j)\) is first-order.
- `traversal_engine.py`: provides traversal algorithms on top of the graph
  and transition model:
  - Beam search over paths (summing log \(P(k\mid i,j)\)).
  - Second-order random walk \(v_t \sim P(v_t \mid v_{t-1}, v_{t-2})\).
  - Personalized PageRank \(R = \alpha T R + (1-\alpha) P_0\) using the
    adjacency as a first-order transition matrix.

How it fits into the system
---------------------------

- The **indexing pipeline** (`bettermem.indexing`) uses `Graph` and node
  types to materialize topic/chunk nodes and edges from a corpus.
- The **transition model** is trained from topic sequences and then used
  by the **traversal engine** during querying.
- The **retrieval layer** (`bettermem.retrieval`) consumes graph-level
  traversal results and maps them back to chunk nodes for scoring and
  context selection.

