# Core

Graph and traversal primitives used by the rest of the library.

## Modules

- **`nodes.py`**: Node types (`TopicNode`, `ChunkNode`, optional `KeywordNode`) and ID helpers. Topic IDs are path-style (e.g. `t:0`, `t:0.1`, `t:0.1.2`). `TopicNode` has `parent_ids`, `chunk_ids`, level, optional centroid in metadata.
- **`edges.py`**: Edge kinds: `TOPIC_SUBTOPIC`, `TOPIC_CHUNK`, `TOPIC_TOPIC`, `CHUNK_CHUNK_STRUCTURAL`, etc.
- **`graph.py`**: Sparse directed graph (adjacency, edge kinds). Builds topic indexes: `get_parents`, `get_children`, `get_siblings`, `is_ancestor` for hierarchy traversal. Serialization via `to_dict` / `from_dict`.
- **`transition_model.py`**: Second-order Markov model P(k|i,j) over topic sequences; uses `bettermem.learning.smoothing`. Blended with first-order P(k|j). Fitted from document topic sequences produced by the indexer.
- **`navigation_policy.py`**: **IntentConditionedPolicy** scores the next topic: relevance (cos(μ_k,q)), continuity (cos(μ_k,μ_i)), R_intent(i,k), novelty, prior, minus repetition and backtrack penalties. **Intent filter**: at each step candidates are filtered by intent (child/parent/sibling/semantic neighbor) with structured fallbacks; relation type uses hierarchy (parent/child/sibling) before edge kind so siblings are not misclassified as semantic neighbors.
- **`traversal_engine.py`**: **TraversalEngine.intent_conditioned_navigate**: from a start node, for each step gets policy distribution (over intent-filtered candidates), optionally blends with Markov transition, then steps (greedy or sample). Returns paths and visit counts used by the retrieval layer.

## How it fits

- **Indexing** creates topic and chunk nodes and edges (including TOPIC_SUBTOPIC for hierarchy).
- **Query**: initializer provides start node and prior; policy + traversal engine produce visit counts over topic nodes; retrieval projects these to chunk scores (with per-topic normalization) and selects contexts.
