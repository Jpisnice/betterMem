Indexing pipeline
=================

The `bettermem.indexing` package turns a raw text corpus into:

- Text **chunks**.
- Topic and chunk **nodes** in the graph.
- Topic–chunk **edges** and topic **sequences** for the transition model.

Modules
-------

- `chunker.py`:
  - Defines `Chunk` objects and a `BaseChunker` protocol.
  - Implements `FixedWindowChunker` that splits documents into overlapping
    windows of tokens \(\{c_1, \dots, c_n\}\).
- `keyword_mapper.py`:
  - Implements `KeywordToTopicMapper`, which approximates
    \(P(t \mid q) \propto \sum_{w \in q} \text{TFIDF}(w, t)\) using counts
    over topic keyword lists.
- `corpus_indexer.py`:
  - Orchestrates indexing with:
    - `build_index(corpus)`: chunk documents, fit topic model, produce
      chunk-level \(P(t \mid \text{chunk})\), create `TopicNode`s and
      `ChunkNode`s, and connect them with weighted edges.
    - Builds per-document sequences of **top topics** and passes them to
      the transition model as training sequences.

Math and concepts
-----------------

- Chunks are basic retrieval units, each linked to one or more topics by
  \(P(t \mid \text{chunk})\).
- For each document, the most probable topic per chunk yields a sequence
  \((t_1, t_2, \dots, t_m)\), which drives second-order count estimation
  for \(P(k \mid i,j)\).

How it fits into the system
---------------------------

- The indexer is called by `BetterMem.build_index` to construct:
  - The **graph** structure (`bettermem.core.graph.Graph`).
  - The **transition model** (`bettermem.core.transition_model.TransitionModel`).
- The resulting graph and transition model are later used by the traversal
  engine and retrieval layer during queries.

