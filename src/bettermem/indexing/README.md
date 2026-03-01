# Indexing

Builds the topic–chunk graph and topic sequences for the transition model.

## Modules

- **`chunker.py`**: `Chunk` dataclass and `BaseChunker` protocol. **ParagraphSentenceChunker** splits by paragraphs and sentences, respects `max_tokens`, keeps structural groups for chunk–chunk edges. Default chunker for the client.
- **`corpus_indexer.py`**: **CorpusIndexer** orchestrates:
  - Chunk corpus, fit topic model on chunk texts, **transform** (P(leaf|chunk) per chunk).
  - **Hierarchical nodes and edges**: for each topic in the hierarchy (from topic model `get_hierarchy`, `get_all_topic_ids`, etc.) add `TopicNode` with parent_ids, level, optional centroid/keywords; add TOPIC_SUBTOPIC edges. For each chunk, add `ChunkNode`; link to **top-M leaf topics** above a probability threshold, with **ancestor decay** so parent topics also get topic–chunk edges (for recall when the walk visits coarse nodes). Fallback: if no topic passes the threshold, link to the best leaf anyway.
  - **Structural chunk edges**: same-document, same structural-group consecutive chunks get CHUNK_CHUNK_STRUCTURAL edges.
  - **Topic–topic semantic edges**: kNN on topic centroids (cosine), excluding ancestor–descendant pairs; optional hnswlib.
  - **Topic sequences** per document (best topic per chunk) for the transition model.
- **`keyword_mapper.py`**: Optional keyword-based topic prior for queries (TFIDF-style over topic keywords).

## Concepts

- Topic model must provide `get_hierarchy()`, `get_all_topic_ids()`, and for hierarchical indexing `get_leaf_topic_ids()`, `get_parents()`, and optionally `get_centroid`, `embed_texts`.
- Chunk–topic links are sparse (top leaves + ancestors with decay) so retrieval can hit chunks when visiting any level of the tree.

## How it fits

- **BetterMem.build_index** creates graph and transition model, then calls the indexer with config (e.g. `topic_chunk_top_m`, `topic_chunk_min_prob`, `topic_chunk_ancestor_decay`). The resulting graph and transition model are used by the traversal engine and retrieval.
