# Retrieval

Turns graph traversals into ranked context chunks and explanations.

## Modules

- **`query_initializer.py`**:
  - **Topic prior** P(t|q) from the topic model (e.g. leaf distribution for the query).
  - For non-NEUTRAL intents, prior is **rolled up** to ancestors so the policy can score any topic; **start node** is chosen from the **leaf** prior so the walk has room to navigate (e.g. BROADEN starts at best leaf to ascend; DEEPEN at parent of best leaf to descend).
  - **Semantic state**: query embedding, path history, prior for the policy.
- **`scorer.py`**: **QueryScorer** aggregates **visit counts** from the traversal into node scores. Used by the client to get topic-level scores before projecting to chunks.
- **`context_aggregator.py`**: Takes **chunk-space** scores (from the client’s `_scores_to_chunk_space`). Selects top-k chunks or context windows; optional diversity and structural window expansion.
- **`intent.py`**: `TraversalIntent` enum and optional intent-from-query heuristics.
- **`relation.py`**: `get_relation_type` (parent/child/sibling/semantic/distant) with **hierarchy checked before edge kind**; `r_intent` for policy scoring.

## Chunk scoring (in API layer)

After the traversal, the client:

1. Converts visit counts to topic scores.
2. **Projects to chunk space** via topic→chunk edges: each topic’s score is **normalized by its chunk fan-out** (so root topics do not dominate).
3. Optionally **reranks** by blending topic score with query–chunk cosine (and placeholder BM25).
4. Passes chunk scores to **ContextAggregator.select** for top-k and optional diversity/windows.

## How it fits

- **QueryInitializer** is used by `BetterMem.query` to get start node and prior; policy and traversal run over the graph; retrieval consumes visit counts and graph to produce chunk scores and final contexts. **Explain** uses the same graph to attach path steps and chunks along the path.
