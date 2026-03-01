Topic modeling layer
====================

The `bettermem.topic_modeling` package provides a unified topic interface
used for indexing and query priors.

- `base.py`: defines `BaseTopicModel` with:
  - `fit(documents)`: train the model on raw texts.
  - `transform(chunks)`: return topic distributions
    \(P(t \mid \text{chunk})\) for each chunk.
  - `get_topic_keywords(topic_id)`: expose top terms per topic.
  - `get_topic_distribution_for_query(text)`: compute query-level
    \(P(t \mid q)\) used for the query prior.
  - Optional `get_centroid`, `embed_query`, `embed_texts`, `get_hierarchy`
    for embedding-based navigation and hierarchical graphs.
- `semantic_hierarchical.py`: **SemanticHierarchicalTopicModel** — the
  default implementation. Uses sentence-transformers embeddings and
  two-level KMeans (coarse then fine per coarse) to build a topic
  hierarchy with centroids and topic–topic similarity edges.

Math and concepts
-----------------

- For each chunk, the model estimates a discrete distribution
  \(P(t \mid \text{chunk})\) over topic IDs.
- For a query \(q\), the model estimates \(P(t \mid q)\). Combined with
  keyword-based priors in `bettermem.indexing.keyword_mapper`, this forms
  the **query prior** \(P_0(t \mid q)\) used by traversal.

How it fits into the system
---------------------------

- The **corpus indexer** (`bettermem.indexing.corpus_indexer`) calls
  `fit` and `transform` to:
  - Create topic nodes.
  - Attach topic distributions to chunks.
  - Build topic sequences used to train the second-order transition model.
- The **query initializer** (`bettermem.retrieval.query_initializer`)
  uses `get_topic_distribution_for_query` to produce topic priors and
  initial topic states for traversal.

