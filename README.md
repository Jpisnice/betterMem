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
- Pluggable topic modeling backends (e.g. BERTopic, SemanticHierarchical with embeddings).

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

When you run a query, BetterMem navigates the topic graph using a **single policy** conditioned on your query and inferred **intent** (e.g. "explain more" → deepen, "big picture" → broaden, "compare" → siblings). This mirrors how humans browse textbooks: choose a topic, then decide whether to go deeper, broaden, compare alternatives, or clarify.

| Strategy | Best for | Behavior |
|----------|----------|----------|
| **beam** | Interpretable, deterministic results | Keeps the top-K highest-probability paths through the graph. Paths are built step-by-step using the learned topic transitions; only the most likely continuations are kept. |
| **random_walk** | Exploratory, single coherent path | Takes one stochastic walk through the graph, sampling the next topic from the transition model. Good when you want variety across queries or a single “storyline” of topics. |
| **personalized_pagerank** | Balanced relevance and coverage (default) | Spreads relevance from your query’s topic prior over the whole graph via repeated propagation. Produces a steady-state importance score per node rather than explicit paths. |

**Policy**: At each step the next topic is chosen by scoring candidates: **Relevance** cos(μ_k, q), **Structural fit** R_intent(i,k), and **Novelty** (repetition penalty). Scores are softmax-normalized; next node is argmax (greedy) or sampled.

**Intents** (inferred from query text, or overridden in `query(intent=...)`): see table above.

(Deprecated: **beam** — Use when you care about *why* something was retrieved. You get explicit paths (topic → topic → …) and can inspect them with `path_trace=True` and `explain()`. Results are deterministic for the same query and index. Tune with `beam_width` (more paths = more coverage, more cost) and `max_steps`.
- **random_walk** — Use when you want one coherent trail through the corpus or more diversity between repeated queries. Increase `exploration_factor` to make the walk less greedy and more exploratory. Paths are available in the explanation when `path_trace=True`.
- **personalized_pagerank** — Use as the default when you want a good balance of relevance and coverage without caring about a single path. It uses your full topic prior (all query-relevant topics) and the graph’s link structure. No path trace; explanations focus on the prior and strategy. Tune with `max_steps` (PageRank iterations).

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
    topicFit["TopicModel.fit: BERTopic or LDA"]
    topicXform["TopicModel.transform"]
    indexer["CorpusIndexer"]
  end

  corpus --> chunker
  corpus --> topicFit
  chunker -->|"chunks c1 .. cn"| indexer
  topicFit --> topicXform
  topicXform -->|"P(t | chunk)"| indexer

  subgraph graphLayer ["Graph G = (V, E) where V = T U C"]
    tNodes["TopicNodes T: t:0, t:1, ..."]
    cNodes["ChunkNodes C: c:0, c:1, ..."]
    adj["Weighted Directed Edges A_ij = w_ij"]
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
    intent["Intent I_t from query text"]
  end

  query --> qInit
  topicFit -->|"trained model"| qInit
  qInit --> topicPrior
  qInit --> intent

  subgraph travEng [Traversal Engine]
    policy["Intent-conditioned policy: Score(k) = alpha cos mu_k q + beta cos mu_k mu_i + gamma R_intent - delta repeat"]
    navigate["Navigate: softmax(Score/T), greedy or sample"]
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
    saveOp["Save: graph.json + transition.json + config.json"]
    loadOp["Load: reconstruct Graph + TransitionModel"]
  end

  adj -->|"serialize"| saveOp
  blend -->|"serialize"| saveOp
  loadOp -->|"deserialize"| adj
  loadOp -->|"deserialize"| blend
```

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