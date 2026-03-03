## BetterMem configuration guide

This guide explains how to configure the BetterMem client through `bettermem.api.config.BetterMemConfig`, what each field does, and how it changes behavior in practice.

The config controls **how the traversal walks the topic graph**, how **semantic neighbors** are wired, how **topics attach to chunks**, and how **final chunks are reranked**. You can treat it as the “personality and physics” of the retriever.

- **Default config**: balanced behavior for typical use.
- **`debug_preset`**: short, deterministic walks for demos and debugging alignment between **intent** and **navigation**.

```python
from bettermem import BetterMem
from bettermem.api.config import BetterMemConfig
from bettermem.topic_modeling.semantic_hierarchical import SemanticHierarchicalTopicModel
from bettermem.indexing.chunker import ParagraphSentenceChunker

config = BetterMemConfig()  # or BetterMemConfig.debug_preset()
topic_model = SemanticHierarchicalTopicModel(...)
chunker = ParagraphSentenceChunker(max_tokens=200)

client = BetterMem(config=config, topic_model=topic_model)
client.build_index(corpus, chunker=chunker)

results = client.query(
    "attention mechanism transformer training",
    top_k=5,
    intent="deepen",      # or TraversalIntent.DEEPEN
    path_trace=True,
)
```

---

## Big-picture mental model

At query time, BetterMem:

1. Builds a **topic prior** \(P(topic \mid query)\) from the topic model.
2. Chooses a **start topic** and **intent** (deepen, broaden, compare, apply, clarify, neutral).
3. Repeatedly scores candidate next topics with a **policy**:
$$
\text{Score}(k)
=
\alpha \cdot \cos(\mu_k, q)
+ \beta \cdot \cos(\mu_k, \mu_i)
+ \gamma \cdot R_{\text{intent}}(i, k)
+ \text{novelty\_bonus}
+ \text{prior\_weight}\cdot \text{prior}(k)
- \text{repetition/backtrack penalties}
$$

4. Optionally blends this policy with a **Markov transition model** over topic sequences.
5. Projects visited topic scores down to **chunks**, with per-topic normalization, and reranks final chunks by a hybrid of **topic score**, **query–chunk cosine**, and optionally **BM25**.

Each config key nudges one part of this pipeline.

---

## Quick-start recommendations

- **Start with defaults** (`BetterMemConfig()`) unless you have a concrete failure mode.
- Use **`BetterMemConfig.debug_preset()`** when:
  - You are running **single-document demos**.
  - You care more about **deterministic, interpretable paths** than recall.
  - You want to visually verify that **intent → graph motion** matches your expectations.
- In production, tune only a small number of knobs at a time:
  - **Traversal length**: `max_steps`
  - **Policy vs Markov**: `transition_policy_mix_eta`
  - **Exploration vs exploitation**: `navigation_temperature`, `navigation_greedy`, `navigation_novelty_bonus`, `navigation_delta`, `navigation_backtrack_penalty`
  - **Graph density**: `neighbor_top_k`, `neighbor_min_cosine`, `dag_tau`, `topic_chunk_top_m`, `topic_chunk_min_prob`, `topic_chunk_ancestor_decay`
  - **Final reranking**: `rerank_weight_topic`, `rerank_weight_cosine`, `rerank_weight_bm25`

---

## Traversal and Markov model

### `smoothing_lambda: float = 0.7`

- **What it does**: In the transition model, interpolates between **second-order** and **first-order** topic transitions.
- **Practical effect**:
  - Higher (\(\to 1\)): path depends more on the **last two topics** (more context-aware, but potentially sparse).
  - Lower (\(\to 0\)): path depends more on the **last topic only** (more robust but less structured).
- **When to tune**:
  - Increase when you have **rich, repeated topic sequences** and want the model to respect longer patterns.
  - Decrease when data is **sparse** or you see the walk getting stuck because higher-order transitions are under-trained.

### `max_steps: int = 32`

- **What it does**: Maximum number of traversal steps before the walk stops.
- **Practical effect**:
  - Lower: **shorter, more focused paths**, fewer topics visited; faster but lower recall.
  - Higher: **longer explorations**, more opportunities to discover diverse chunks; more compute.
- **Typical settings**:
  - Demos / debugging: `5–10` (or use `debug_preset`, which sets `max_steps=5`).
  - Production: `16–32` depending on performance budget and corpus size.

### `transition_policy_mix_eta: float = 1.0`

- **What it does**: Blends **policy-based navigation** and **Markov transitions**:
  - \(P = \eta \cdot P_{\text{policy}} + (1-\eta)\cdot P_{\text{markov}}\).
- **Practical effect**:
  - `1.0`: **pure policy**. Navigation is driven by query relevance, continuity, and intent; ignores learned transition statistics.
  - `0.0`: **pure Markov**. Navigation follows typical topic sequences seen in training, almost ignoring the query.
  - In between: policy proposes, Markov prior regularizes.
- **When to tune**:
  - Start at `1.0` during prototyping and debugging.
  - Move towards `0.7–0.9` once you have **stable transition statistics** and want navigation to follow natural discourse structure.

---

## Intent-conditioned policy weights

These parameters shape how the policy scores candidate next topics.

### `navigation_alpha: float = 0.5`

- **Meaning**: Weight of **relevance** \(\cos(\mu_k, q)\) between candidate topic centroid and query embedding.
- **Higher**:
  - Stays **closer to query semantics**, less willing to follow continuity or structural cues.
  - Good when the query is very specific and you trust the embedding model.
- **Lower**:
  - Lets **graph structure** and **intent** play a larger role.

### `navigation_beta: float = 0.3`

- **Meaning**: Weight of **continuity** \(\cos(\mu_k, \mu_i)\) between candidate topic and current topic.
- **Higher**:
  - Encourages **smooth, coherent walks** through local neighborhoods.
  - Reduces abrupt jumps to distant topics even if they match the query.
- **Lower**:
  - Allows more **teleportation** to far-but-relevant topics.

### `navigation_gamma: float = 0.5`

- **Meaning**: Weight for **structural fit** \(R_{\text{intent}}(i,k)\), which encodes whether the move matches the chosen intent (e.g. child for DEEPEN, parent for BROADEN, sibling for COMPARE, semantic neighbor for APPLY/CLARIFY).
- **Higher**:
  - Makes intent **very strong**: DEEPEN walks mostly downward, BROADEN mostly upward, etc.
  - Good when you want predictable structural behavior per intent.
- **Lower**:
  - Intent becomes a mild preference; relevance and continuity dominate.
- **Debug preset**: sets `navigation_gamma=1.2` so structural behavior is easy to see.

### `navigation_delta: float = 0.5`

- **Meaning**: Strength of **repetition penalty**, which grows with visit count to penalize revisiting the same topic repeatedly.
- **Higher**:
  - Strongly discourages revisiting already-visited nodes.
  - Leads to **broader exploration** but may cut off useful revisits.
- **Lower**:
  - Allows revisiting hubs; can be good when some topics are legitimately central.

### `navigation_backtrack_penalty: float = 5.0`

- **Meaning**: Extra penalty for stepping back to the **immediate previous node** (backtracking).
- **Practical effect**:
  - Prevents **oscillation** between parent and child topics.
  - If you set this very low, paths may bounce back and forth in small subtrees.

### `navigation_novelty_bonus: float = 0.3`

- **Meaning**: Positive bonus for **never-visited** candidate topics.
- **Practical effect**:
  - Encourages **exploration** of new parts of the graph.
  - If very high, you may over-explore and miss deep exploitation of good areas.
- **Debug preset**: `navigation_novelty_bonus=0.05` (lighter exploration; focus on structural behavior).

### `navigation_prior_weight: float = 0.2`

- **Meaning**: Weight for **topic prior** \(P(topic \mid query)\) in candidate scoring.
- **Practical effect**:
  - Higher: traversal remains more anchored to where the **topic model thinks the query lives**.
  - Lower: traversal is driven more by local structure and recent history.
- **Debug preset**: `navigation_prior_weight=0.3` to slightly strengthen the prior.

### `navigation_temperature: float = 1.0`

- **Meaning**: Softmax temperature for converting scores to probabilities.
- **Practical effect**:
  - Lower (< 1): distribution is **sharper**; picks high-scoring candidates more often.
  - Higher (> 1): distribution is **flatter**; more random, exploratory paths.
- **Debug preset**: `navigation_temperature=0.35` to make paths almost greedy but still numeric.

### `navigation_greedy: bool = False`

- **Meaning**: If `True`, always pick the **argmax** candidate (deterministic). If `False`, sample from the softmax distribution.
- **Practical effect**:
  - `True`: **reproducible** walks given the same random seed; easier to debug and visualize.
  - `False`: adds **stochasticity**, which can improve exploration and diversity across repeated queries.
- **Debug preset**: `navigation_greedy=True` for deterministic behavior.

---

## Graph shape and semantics

These settings control how the **topic graph** and **semantic neighbors** are constructed.

### `neighbor_top_k: int = 20`

- **Meaning**: Maximum number of **semantic neighbors** (topic–topic edges) per topic.
- **Practical effect**:
  - Higher: denser semantic graph; more cross-branch paths are possible.
  - Lower: sparser graph; traversal relies more on the tree hierarchy.
- **Debug preset**: `neighbor_top_k=6` for a lighter, easier-to-interpret neighbor graph.

### `neighbor_min_cosine: float = 0.0`

- **Meaning**: Minimum cosine similarity threshold for adding semantic edges.
- **Practical effect**:
  - Higher: only **very similar** topics are connected; fewer but stronger semantic shortcuts.
  - Lower: more edges, including weaker relationships; may introduce noise.
- **Debug preset**: `neighbor_min_cosine=0.35`.

### `clarify_similarity_threshold: float = 0.7`

- **Meaning**: Minimum similarity threshold used specifically for the **CLARIFY** intent.
- **Practical effect**:
  - Clarify moves prefer **high-similarity neighbors** above this cutoff, giving “explain this in slightly different words” behavior.
- **Debug preset**: `clarify_similarity_threshold=0.8` to make clarify jumps very local.

### `hierarchy_max_depth: int = 3`

- **Meaning**: Maximum depth for **recursive topic clustering** (topic tree).
- **Practical effect**:
  - Higher: deeper trees; more fine-grained subtopics.
  - Lower: shallower hierarchy; coarser topics.
- **Guideline**:
  - For small corpora: `2–3`.
  - For large corpora with diverse content: `3–4` (but beware complexity).

### `min_cluster_size: int = 2`

- **Meaning**: Minimum cluster size during recursive splitting.
- **Practical effect**:
  - Higher: prevents tiny clusters (noise) but may **merge distinct topics**.
  - Lower: allows narrow, specific topics but may lead to brittle, small clusters.

### `dag_tau: Optional[float] = None`

- **Meaning**: If set, allows the hierarchy to become a **DAG** (multi-parent topics) by adding extra parent edges when centroid cosine similarity ≥ `dag_tau`.
- **Practical effect**:
  - `None`: strict **tree**; each topic has exactly one parent.
  - Value in \([0,1]\): more **multi-parent relationships**; useful when topics belong naturally to multiple higher-level themes.
- **When to use**:
  - Set, e.g., `dag_tau=0.7` when you see many concepts that are clearly shared across branches and want **cross-cutting structure**.

---

## Topic–chunk attachment

These parameters control how chunk nodes are connected to topic nodes.

### `topic_chunk_top_m: int = 2`

- **Meaning**: Maximum number of **leaf topics** to which each chunk can attach directly.
- **Practical effect**:
  - Higher: each chunk participates in more topics, increasing recall but possibly blurring topic boundaries.
  - Lower: chunk is assigned to fewer topics; retrieval is more precise but may miss some contexts.

### `topic_chunk_min_prob: float = 0.15`

- **Meaning**: Minimum topic probability required to attach a chunk to a topic.
- **Practical effect**:
  - Higher: only strong chunk–topic associations are kept; fewer edges, higher precision.
  - Lower: more edges, including weaker associations; higher recall.

### `topic_chunk_ancestor_decay: float = 0.7`

- **Meaning**: Decay factor used when attaching chunks **up the hierarchy** from their best leaf topics.
- **Practical effect**:
  - Higher: ancestors retain more weight; chunks contribute more strongly to higher-level topics.
  - Lower: ancestor connections are weaker; retrieval focuses more on leaf-level topics.
- **Intuition**:
  - A chunk primarily “belongs” to its leaf topics, but some mass is propagated to parents so **BROADEN** can still find it via higher-level nodes.

---

## Chunk reranking

After traversal, BetterMem has topic visit counts. It then:

1. Projects topic scores down to chunks via topic–chunk edges (with per-topic normalization).
2. Optionally combines that score with **query–chunk cosine** and **BM25**.

These three weights control that final mix.

### `rerank_weight_topic: float = 0.6`

- **Meaning**: Weight of the **topic-projected score**.
- **Practical effect**:
  - Higher: results follow the **navigation path** more strongly; good when you trust the graph structure and intent behavior.
  - Lower: navigation plays a smaller role; retrieval becomes more “flat” embedding/BM25 driven.

### `rerank_weight_cosine: float = 0.4`

- **Meaning**: Weight of **query–chunk cosine**.
- **Practical effect**:
  - Higher: chunks that are directly close to the query in embedding space are prioritized even if the traversal didn’t spend much time near them.
  - Lower: more weight shifts to graph structure and/or BM25.

### `rerank_weight_bm25: float = 0.0`

- **Meaning**: Weight of **BM25** scores in the final ranking.
- **Practical effect**:
  - `0.0`: BM25 is **disabled**; retrieval is fully neural + graph-based.
  - `> 0`: BM25 contributes lexical matching, which can help with rare tokens, IDs, or when embeddings are weak for certain domains.
- **Guideline**:
  - Start with `0.0`; increase gradually (`0.1–0.3`) if you see failures on **keyword-heavy** or **ID-heavy** queries.

---

## Index storage and persistence

### `index_storage_path: Optional[str] = None`

- **Meaning**: Optional **default path** for saving/loading indices.
- **Practical effect**:
  - If set, `client.save()` and `BetterMem.load()` can use this as a default location (depending on how you wire them in your application).
  - Can be used to keep indexes for different corpora in well-known locations.

For more on persistence, see the main project README (`client.save(path)` / `BetterMem.load(path)`).

---

## Preset: `BetterMemConfig.debug_preset()`

The `debug_preset` classmethod returns a `BetterMemConfig` tuned for **short, deterministic, interpretable runs**:

- `max_steps=5` → shorter walks.
- `navigation_greedy=True` → argmax selection (no sampling).
- `navigation_temperature=0.35` → sharp distribution when sampling is used.
- `navigation_gamma=1.2` → strong structural fit to the chosen intent.
- `navigation_novelty_bonus=0.05` → limited exploration.
- `navigation_prior_weight=0.3` → slightly stronger query-based prior.
- `neighbor_top_k=6`, `neighbor_min_cosine=0.35` → fewer, stronger semantic neighbors.
- `clarify_similarity_threshold=0.8` → CLARIFY stays very local.
- `transition_policy_mix_eta=1.0` → pure policy (no Markov blend).
- `dag_tau=None` → strict tree hierarchy (no multi-parent topics).

**Use this when**:

- You are building **demos** (like the `demo/basic_usage.py` script).
- You want to verify that each **intent** (DEEPEN, BROADEN, COMPARE, APPLY, CLARIFY, NEUTRAL) produces the kind of path you expect.
- You are debugging **unexpected paths** and want to reduce sources of randomness.

**In production**, start from `BetterMemConfig()` and only borrow ideas from this preset (e.g. a higher `navigation_gamma` if you want stronger structural intent).

---

## Example tuning recipes

### 1. Make DEEPEN really drill down

- Increase `navigation_gamma` (e.g. `0.8–1.2`) so child moves are strongly preferred.
- Slightly decrease `navigation_beta` so continuity doesn’t drag you sideways.
- Keep `navigation_alpha` moderate so queries still matter.

### 2. Encourage more exploration across the graph

- Increase `navigation_novelty_bonus`.
- Decrease `navigation_delta` (repetition penalty) slightly.
- Increase `neighbor_top_k` or lower `neighbor_min_cosine` to add more semantic edges.

### 3. Make results more “flat embedding search”-like

- Increase `rerank_weight_cosine`.
- Decrease `rerank_weight_topic`.
- Optionally keep `transition_policy_mix_eta` high so paths are still intent-aware.

### 4. Add lexical robustness

- Set `rerank_weight_bm25` to something like `0.1–0.3`.
- Keep the sum of `(rerank_weight_topic, rerank_weight_cosine, rerank_weight_bm25)` reasonable (e.g. normalize in your application if you change them heavily).

---

## Where to go next

- Main project overview and quick start: `README.md` at the repository root.
- Implementation details:
  - `bettermem.api.config.BetterMemConfig` for the config model.
  - `bettermem.retrieval` for query initialization, scoring, and traversal.
  - `bettermem.topic_modeling` and `bettermem.indexing` for how topics and edges are built.

