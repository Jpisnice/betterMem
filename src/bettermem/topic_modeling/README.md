# Topic modeling

Unified topic interface for indexing and query priors.

## Modules

- **`base.py`**: **BaseTopicModel** with `fit(documents)`, `transform(chunks)` → P(t|chunk) per chunk, `get_topic_distribution_for_query(text)`, and optional `get_topic_keywords`, `get_centroid`, `embed_query`, `embed_texts`, `get_hierarchy`, `get_all_topic_ids`, `get_leaf_topic_ids`, `get_parents`, `rollup_leaf_prior_to_ancestors` for hierarchical graphs.
- **`semantic_hierarchical.py`**: **SemanticHierarchicalTopicModel** (default):
  - Embeddings via sentence-transformers; recursive KMeans to build a **multi-level** topic tree (path IDs: `t:0`, `t:0.0`, `t:0.0.0`, …).
  - **Leaves**: topics with no children; **get_hierarchy** = parent → list of children; **get_parents**, **get_leaf_topic_ids**, **get_all_topic_ids**, **get_centroid**, **get_topic_keywords**.
  - **transform(chunks, temperature=0.1)**: P(leaf|chunk) using **cosine similarity** between chunk embedding and leaf centroids, then **softmax with temperature** (so distributions are peaked enough for topic–chunk edges). Returns one distribution per chunk.
  - **rollup_leaf_prior_to_ancestors**: roll leaf prior to all topics for policy prior.
  - Optional **dag_tau**: add multi-parent (DAG) edges when centroid similarity ≥ tau.

## How it fits

- **CorpusIndexer** uses `fit` and `transform` to create topic nodes and topic–chunk links. **QueryInitializer** uses `get_topic_distribution_for_query` (and optional rollup) for priors and start-node choice.
