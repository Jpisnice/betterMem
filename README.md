BetterMem
=========

BetterMem is an experimental Python library that implements a
probabilistic topic transition graph for retrieval. It treats knowledge
navigation as a discrete stochastic process over a graph of topic and
chunk nodes rather than pure vector similarity search.

High-level features:

- Second-order Markov transitions over topic sequences.
- Interpretable traversal paths via beam search and random walks.
- Personalized PageRank over the topic graph.
- Pluggable topic modeling backends (e.g. BERTopic, LDA).

The public entry point is the `BetterMem` client:

```python
from bettermem import BetterMem

client = BetterMem()
# client.build_index(corpus)
# results = client.query("Explain transformer attention")
```

Architecture
------------

```mermaid
flowchart TD
  user[User] --> client[BetterMemClient]
  corpus[RawDocuments] --> client

  client --> buildIndex[BuildIndex]
  buildIndex --> chunker[Chunker]
  chunker --> chunks[Chunks]
  buildIndex --> topicModel[TopicModel]
  topicModel --> chunkTopics["ChunkTopicDists"]
  chunks --> corpusIndexer[CorpusIndexer]
  chunkTopics --> corpusIndexer
  corpusIndexer --> graph["Graph G(V,E)"]
  corpusIndexer --> transModel["TransitionModel P(v_t|v_{t-1},v_{t-2})"]

  graph --> saveIndex[SaveIndex]
  transModel --> saveIndex
  saveIndex --> storage[IndexOnDisk]
  storage --> loadIndex[LoadIndex]
  loadIndex --> graph
  loadIndex --> transModel

  user --> queryText[QueryText]
  queryText --> client
  client --> queryHandler[Query]

  queryHandler --> qInit[QueryInitializer]
  qInit --> topicPrior["TopicPrior P0(t|q)"]
  topicPrior --> startState["InitialState (v_{t-2},v_{t-1})"]

  startState --> traversal[TraversalEngine]
  transModel --> traversal
  graph --> traversal

  traversal --> paths[VisitedPaths]
  paths --> scorer[QueryScorer]
  scorer --> nodeScores[NodeScores]

  nodeScores --> ctxAgg[ContextAggregator]
  graph --> ctxAgg
  ctxAgg --> results[ContextChunks]

  results --> client
  client --> user

  traversal --> explainData["Path+TransitionInfo"]
  explainData --> explainAPI[Explain]
  explainAPI --> user
```