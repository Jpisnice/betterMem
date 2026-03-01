BetterMem
=========

BetterMem is an experimental Python library that implements a
probabilistic topic transition graph for retrieval. It treats knowledge
navigation as a discrete stochastic process over a graph of topic and
chunk nodes rather than pure vector similarity search.

High-level features:

- **Intent-conditioned navigation**: goal-directed traversal in semantic space conditioned on user intent (deepen, broaden, compare, apply, clarify), approximating textbook-style browsing.
- Graph built from embedding space: topic centroids, parent-child hierarchy, and semantic (topic-topic) edges.
- Single policy P(v_{t+1} | v_t, I_t, S_t) with scoring: relevance + structural fit + novelty (repetition penalty).
- Semantic hierarchical topic model (embeddings + two-level clustering) for graph building and query priors.

The public entry point is the `BetterMem` client:

```python
from bettermem import BetterMem

client = BetterMem()
# client.build_index(corpus)
# results = client.query("Explain transformer attention")
```

For a more complete, runnable console demo, see `demo/basic_usage.py`:

```bash
uv run python demo/basic_usage.py
```

Intent-conditioned navigation
------------------------------

When you run a query, BetterMem navigates the topic graph using a **single policy** conditioned on your query and inferred **intent**. The policy scores each candidate next topic and chooses by softmax (greedy or sampled). This mirrors textbook browsing: choose a topic, then go deeper, broaden, compare alternatives, or clarify.

| Intent    | Graph relation preferred     | Basis |
|-----------|------------------------------|--------|
| **deepen**  | **Child** (subtopic of current) | Move to more specific subtopics; use when the user wants more detail or a deeper explanation. |
| **broaden**  | **Parent** (coarse topic of current) | Move to the broader topic; use for big picture, overview, or context. |
| **compare**  | **Sibling** (same parent as current) | Move to sibling subtopics; use to compare alternatives or related concepts at the same level. |
| **apply**    | **Semantic neighbor** (topic-topic edge) | Move to semantically related topics; use for applications, examples, or related domains. |
| **clarify**  | **Semantic neighbor** (high similarity) | Same as apply in structure; use when the current topic is unclear and the user wants a related explanation. |
| **neutral**  | None (relevance + continuity only) | No structural bias; next topic by relevance to the query and continuity with the current topic. |

**Policy** (at each step):  
Score(k) = α·cos(μ_k, q) + β·cos(μ_k, μ_i) + γ·R_intent(i,k) + novelty_bonus + prior_weight·prior(k) − repetition_penalty − backtrack_penalty.  
Scores are softmax-normalized; next node is argmax (greedy) or sampled. Intent is inferred from query text or set via `query(..., intent=TraversalIntent.DEEPEN)`.

### Navigation graph (conceptual)

Topic nodes and relation types form the graph that intents steer over: coarse topics have subtopics (child nodes); siblings share a parent; topic–topic edges connect semantically similar topics (centroid similarity).

```mermaid
graph TB
  subgraph hierarchy [Topic hierarchy]
    T0["t:0 coarse"]
    T00["t:0:0 sub"]
    T01["t:0:1 sub"]
    T02["t:0:2 sub"]
    T1["t:1 coarse"]
    T10["t:1:0 sub"]
    T0 -->|parent-child BROADEN up DEEPEN down| T00
    T0 --> T01
    T0 --> T02
    T1 --> T10
    T00 ---|sibling COMPARE| T01
    T00 ---|sibling| T02
  end
  T00 -.->|topic-topic APPLY CLARIFY| T10
```

Example:
Example: configure policy weights and use path trace:

```python
from bettermem import BetterMem
from bettermem.api.config import BetterMemConfig

client = BetterMem(
    config=BetterMemConfig(
        navigation_alpha=0.5,
        navigation_gamma=0.5,
        navigation_temperature=0.8,
        navigation_greedy=False,
    )
)
results = client.query("explain more about attention", path_trace=True)
# client.explain() includes "intent" and "paths"
```

Architecture
------------

```mermaid
flowchart TD

  subgraph inputLayer [Inputs]
    corpus["Raw Corpus Documents"]
    query["Query text q"]
  end

  subgraph indexPipeline [Indexing Pipeline]
    chunker["Chunker: fixed-window / sentence"]
    topicFit["TopicModel.fit on chunks"]
    topicXform["TopicModel.transform"]
    indexer["CorpusIndexer"]
  end

  corpus --> chunker
  chunker -->|"chunk texts"| topicFit
  chunker -->|"chunks c1 .. cn"| indexer
  topicFit --> topicXform
  topicXform -->|"P(t | chunk)"| indexer

  subgraph graphLayer ["Graph G = (V, E): topic and chunk nodes"]
    tNodes["Topic nodes T: coarse t:i, subtopics t:i:j"]
    cNodes["Chunk nodes C: c:0, c:1, ..."]
    adj["Edges: parent-child, topic-chunk, topic-topic semantic"]
  end

  indexer -->|"create nodes"| tNodes
  indexer -->|"create nodes"| cNodes
  tNodes --> adj
  cNodes --> adj

  subgraph transLayer ["Transition Model: 2nd-Order Markov"]
    seqs["Topic sequences per document"]
    p2["P2(k|i,j) = count(i,j,k) / Sum_k count(i,j,k)"]
    p1["P1(k|j) = count(j,k) / Sum_k count(j,k)"]
    blend["Blended: P = lambda * P2 + (1 - lambda) * P1"]
  end

  indexer -->|"extract sequences"| seqs
  seqs --> p2
  seqs --> p1
  p2 --> blend
  p1 --> blend

  subgraph learnLayer [Learning Layer]
    laplace["Laplace smoothing: (count + alpha) / (total + alpha * V)"]
    dirichlet["Dirichlet smoothing: alpha = prior_mass / V"]
    entCtrl["Entropy control: H(P) = -Sum p * log p"]
    tempRescale["Temperature rescaling"]
    rlHook["RL hook: pi(v_t | S) with reward signal"]
  end

  laplace --> p2
  dirichlet --> p1
  entCtrl --> tempRescale
  tempRescale -->|"adjust sharpness"| blend
  rlHook -->|"reweight transitions"| blend

  subgraph queryPipe [Query Pipeline]
    qInit["QueryInitializer"]
    topicPrior["Topic prior P0 and semantic state S_t"]
    intent["Intent I_t: deepen, broaden, compare, apply, clarify, neutral"]
  end

  query --> qInit
  topicFit -->|"trained model"| qInit
  qInit --> topicPrior
  qInit --> intent

  subgraph travEng [Traversal Engine]
    policy["Policy: Score(k) = alpha cos(mu_k,q) + beta cos(mu_k,mu_i) + gamma R_intent + novelty - penalties"]
    navigate["Navigate over graph: softmax(Score/T), greedy or sample"]
  end

  topicPrior --> navigate
  intent --> policy
  adj -->|"centroids, edges"| policy
  policy --> navigate

  subgraph scoreAgg [Scoring and Aggregation]
    scorer["Visit counts from path"]
    ctxAgg["ContextAggregator: top-K + diversity re-ranking"]
  end

  navigate --> scorer
  scorer --> ctxAgg
  adj -->|"chunk lookup"| ctxAgg

  subgraph outputLayer [Output]
    results["Retrieved Context Chunks"]
    explainOut["Explain: traversal paths + transition probs + topic keywords"]
  end

  ctxAgg --> results
  scorer --> explainOut

  subgraph storageLayer [Persistence]
    saveOp["Save: graph.joblib + transition.joblib + config.json"]
    loadOp["Load: reconstruct Graph + TransitionModel"]
  end

  adj -->|"serialize"| saveOp
  blend -->|"serialize"| saveOp
  loadOp -->|"deserialize"| adj
  loadOp -->|"deserialize"| blend
```

Index directories use joblib for the graph and transition model
(`graph.joblib`, `transition.joblib`). Optional `config.json` holds human-readable config.

Testing
-------

This project uses `pytest` and is configured for the `uv` package manager.

- **Install dependencies with dev extras (includes pytest):**

  ```bash
  uv sync --extra dev
  ```

- **Run the full test suite:**

  ```bash
  uv run pytest
  ```

- **Run a single test file (example):**

  ```bash
  uv run pytest tests/test_transition_and_traversal.py
  ```