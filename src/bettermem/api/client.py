from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bettermem.core.edges import EdgeKind
from bettermem.core.graph import Graph
from bettermem.core.navigation_policy import IntentConditionedPolicy
from bettermem.core.transition_model import TransitionModel
from bettermem.core.traversal_engine import TraversalEngine
from bettermem.indexing.chunker import ParagraphSentenceChunker
from bettermem.retrieval.intent import TraversalIntent, classify_intent_heuristic
from bettermem.indexing.corpus_indexer import CorpusIndexer
from bettermem.retrieval.context_aggregator import ContextAggregator, ContextWindow
from bettermem.retrieval.query_initializer import QueryInitializer
from bettermem.retrieval.scorer import QueryScorer
from bettermem.storage.persistence import load_index, save_index
from bettermem.topic_modeling.semantic_hierarchical import SemanticHierarchicalTopicModel
from .config import BetterMemConfig


class BetterMem:
    """High-level interface for the BetterMem semantic hierarchical retriever.

    Orchestrates indexing and intent-conditioned traversal over a hierarchical
    topic graph: coarse/fine topics, second-order Markov transitions, and
    policy-driven navigation (deepen, broaden, compare, apply, clarify, neutral).
    """

    def __init__(
        self,
        *,
        config: Optional[BetterMemConfig] = None,
        topic_model: Optional[Any] = None,
    ) -> None:
        """Initialize a BetterMem instance.

        Parameters
        ----------
        config:
            Optional BetterMemConfig. If omitted, defaults are used.
        topic_model:
            Optional semantic hierarchical topic model (e.g. SemanticHierarchicalTopicModel).
            When None, a default is created during build_index.
        """
        self.config: BetterMemConfig = config or BetterMemConfig()
        self._topic_model = topic_model

        # Internal components
        self._graph: Optional[Graph] = None
        self._transition_model: Optional[TransitionModel] = None
        self._indexer: Optional[CorpusIndexer] = None
        self._traversal_engine: Optional[TraversalEngine] = None
        self._context_aggregator: Optional[ContextAggregator] = None
        self._last_explanation: Any = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def build_index(
        self,
        corpus: Iterable[str],
        *,
        chunker: Optional[Any] = None,
    ) -> None:
        """Build the semantic hierarchical index over the corpus.

        Chunks documents, fits the hierarchical topic model, builds the
        topic–chunk graph, and estimates second-order transition probabilities.

        Parameters
        ----------
        corpus:
            Documents to index.
        chunker:
            Optional chunker. If None, uses ParagraphSentenceChunker() (paragraphs, whole sentences).
        """
        # Initialize components
        self._graph = Graph()
        self._transition_model = TransitionModel(
            smoothing_lambda=self.config.smoothing_lambda
        )

        if self._topic_model is None:
            self._topic_model = SemanticHierarchicalTopicModel(
                max_depth=self.config.hierarchy_max_depth,
                min_cluster_size=self.config.min_cluster_size,
                dag_tau=self.config.dag_tau,
            )

        if chunker is None:
            chunker = ParagraphSentenceChunker()
        self._indexer = CorpusIndexer(
            chunker=chunker,
            topic_model=self._topic_model,
            graph=self._graph,
            transition_model=self._transition_model,
        )
        self._indexer.build_index(
            corpus,
            neighbor_top_k=self.config.neighbor_top_k,
            neighbor_min_cosine=self.config.neighbor_min_cosine,
            topic_chunk_top_m=self.config.topic_chunk_top_m,
            topic_chunk_min_prob=self.config.topic_chunk_min_prob,
            topic_chunk_ancestor_decay=self.config.topic_chunk_ancestor_decay,
        )

        self._traversal_engine = TraversalEngine(
            graph=self._graph,
            transition_model=self._transition_model,
        )
        self._context_aggregator = ContextAggregator(graph=self._graph)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        *,
        steps: Optional[int] = None,
        top_k: int = 8,
        diversity: bool = True,
        path_trace: bool = False,
        intent: Optional[TraversalIntent] = None,
    ) -> Sequence[Any]:
        """Run a retrieval query with intent-conditioned traversal over the topic graph.

        Parameters
        ----------
        text:
            Natural language query.
        steps:
            Override for traversal steps; default from config.
        top_k:
            Number of chunk contexts to return.
        diversity:
            Apply diversity-aware selection to final contexts.
        path_trace:
            If True, explain() will include the traversal path.
        intent:
            Override traversal intent (deepen/broaden/compare/apply/clarify/neutral);
            default is inferred from query text.

        Returns
        -------
        Sequence of ContextWindow (each has .chunks and .score).
        """
        if self._graph is None or self._transition_model is None or self._traversal_engine is None:
            raise RuntimeError("Index has not been built or loaded.")

        steps = steps or self.config.max_steps
        intent_val = intent if intent is not None else classify_intent_heuristic(text)

        from bettermem.core.nodes import NodeKind

        prior: dict[str, float] = {}
        semantic_state = None
        start_node: Optional[str] = None

        if self._topic_model is not None:
            initializer = QueryInitializer(topic_model=self._topic_model)
            start_pair, prior_map = initializer.initial_state(text, intent=intent_val)
            prior = dict(prior_map)
            semantic_state = initializer.semantic_state(text, prior=prior)
            if start_pair is not None:
                start_node = start_pair[0]
            elif prior:
                start_node = max(prior.items(), key=lambda kv: kv[1])[0]

        if not prior:
            topic_ids = [
                nid
                for nid, node in self._graph.nodes.items()
                if node.kind == NodeKind.TOPIC
            ]
            if not topic_ids:
                return []
            uniform = 1.0 / float(len(topic_ids))
            prior = {nid: uniform for nid in topic_ids}
            start_node = topic_ids[0]
        if semantic_state is None:
            from bettermem.core.navigation_policy import SemanticState
            semantic_state = SemanticState(path_history=[], prior=prior)

        if start_node is None:
            return []

        policy = IntentConditionedPolicy(
            self._graph,
            alpha=self.config.navigation_alpha,
            beta=self.config.navigation_beta,
            gamma=self.config.navigation_gamma,
            delta=self.config.navigation_delta,
            backtrack_penalty=self.config.navigation_backtrack_penalty,
            novelty_bonus=self.config.navigation_novelty_bonus,
            prior_weight=self.config.navigation_prior_weight,
            clarify_similarity_threshold=self.config.clarify_similarity_threshold,
        )
        traversal_res = self._traversal_engine.intent_conditioned_navigate(
            start_nodes=[start_node],
            steps=steps,
            intent=intent_val,
            semantic_state=semantic_state,
            policy=policy,
            temperature=self.config.navigation_temperature,
            greedy=self.config.navigation_greedy,
            transition_policy_mix_eta=self.config.transition_policy_mix_eta,
        )
        scorer = QueryScorer()
        scorer.add_visit_counts(traversal_res.visit_counts)
        scores = scorer.scores()
        scores = self._scores_to_chunk_space(scores)
        scores = self._rerank_chunk_scores(text, scores)
        contexts = self._context_aggregator.select(
            scores,
            top_k=top_k,
            diversity=diversity,
        )
        self._last_explanation = self._build_structured_explanation(
            paths=traversal_res.paths if path_trace else [],
            prior=prior,
            intent_val=intent_val,
            include_embeddings=True,
        )
        return list(contexts)

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def explain(self, *, include_embeddings: bool = True) -> Any:
        """Return a structured explanation of the last query.

        When path_trace was True, includes: intent, topic prior, path (topic node ids),
        path_steps (topic labels/keywords/level and chunks per step), and
        chunks_along_path. Set include_embeddings=False to omit embedding fields.
        """
        if self._last_explanation is None:
            return None
        if include_embeddings:
            return self._last_explanation

        def strip_embeddings(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
            if isinstance(obj, list):
                return [strip_embeddings(x) for x in obj]
            return obj

        return strip_embeddings(dict(self._last_explanation))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_structured_explanation(
        self,
        paths: List[List[str]],
        prior: Dict[str, float],
        intent_val: TraversalIntent,
        *,
        include_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Build a structured explanation using stored chunk and embedding references."""
        from bettermem.core.nodes import ChunkNode, NodeKind, TopicNode

        explanation: Dict[str, Any] = {
            "strategy": "intent_conditioned",
            "intent": intent_val.value,
            "prior": prior,
        }
        if not paths:
            return explanation

        explanation["path"] = paths[0] if paths else []
        explanation["paths"] = paths
        path_node_ids = explanation["path"]
        if not path_node_ids or self._graph is None:
            return explanation

        path_steps: List[Dict[str, Any]] = []
        chunk_ids_needing_embed: List[tuple[int, int, str]] = []  # (step_idx, chunk_idx, text)

        for step_idx, node_id in enumerate(path_node_ids):
            node = self._graph.get_node(node_id)
            if node is None or node.kind != NodeKind.TOPIC:
                path_steps.append({"node_id": node_id, "label": None, "keywords": None, "level": None, "parent_id": None, "chunks": []})
                continue

            topic = node if isinstance(node, TopicNode) else None
            step_info: Dict[str, Any] = {
                "node_id": node_id,
                "label": topic.label if topic else None,
                "keywords": topic.keywords if topic else None,
                "level": topic.level if topic else None,
                "parent_id": topic.parent_id if topic else None,
                "chunks": [],
            }

            neighbors = self._graph.get_neighbors(node_id)
            chunk_id_weight_pairs: List[Tuple[str, float]] = []
            if topic and topic.chunk_ids:
                weight_by_neigh = dict(neighbors)
                for cid in topic.chunk_ids:
                    chunk_id_weight_pairs.append((cid, weight_by_neigh.get(cid, 1.0)))
            else:
                for neigh_id, weight in neighbors.items():
                    if self._graph.get_edge_kind(node_id, neigh_id) != EdgeKind.TOPIC_CHUNK:
                        continue
                    chunk_id_weight_pairs.append((neigh_id, float(weight)))

            for neigh_id, weight in chunk_id_weight_pairs:
                neigh = self._graph.get_node(neigh_id)
                if neigh is None or neigh.kind != NodeKind.CHUNK:
                    continue
                chunk = neigh if isinstance(neigh, ChunkNode) else None
                text = (chunk.metadata or {}).get("text", "") if chunk else ""
                text_snippet = (text[:500] + "...") if len(text) > 500 else text
                chunk_entry: Dict[str, Any] = {
                    "chunk_id": neigh_id,
                    "text_snippet": text_snippet,
                    "document_id": getattr(chunk, "document_id", None) if chunk else None,
                    "position": getattr(chunk, "position", None) if chunk else None,
                    "topic_weight": float(weight),
                }
                if include_embeddings and chunk is not None and getattr(chunk, "embedding", None) is not None:
                    chunk_entry["embedding"] = list(chunk.embedding)
                elif include_embeddings and text and self._topic_model is not None:
                    chunk_ids_needing_embed.append((step_idx, len(step_info["chunks"]), text))
                step_info["chunks"].append(chunk_entry)

            path_steps.append(step_info)

        if chunk_ids_needing_embed and self._topic_model is not None and hasattr(self._topic_model, "embed_texts"):
            texts_to_embed = [t for (_, _, t) in chunk_ids_needing_embed]
            embeddings = self._topic_model.embed_texts(texts_to_embed)
            if embeddings is not None and len(embeddings) == len(chunk_ids_needing_embed):
                for (step_idx, chunk_idx, _), emb in zip(chunk_ids_needing_embed, embeddings):
                    path_steps[step_idx]["chunks"][chunk_idx]["embedding"] = list(emb)

        explanation["path_steps"] = path_steps

        seen_chunk_ids: set = set()
        all_chunk_entries = []
        for step_idx, step in enumerate(path_steps):
            for c in step.get("chunks", []):
                chunk_id = c.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunk_entries.append({**c, "from_step": step_idx})
        explanation["chunks_along_path"] = all_chunk_entries
        return explanation

    def _scores_to_chunk_space(self, scores: dict[str, float]) -> dict[str, float]:
        """Project topic-node visit scores into chunk space via topic→chunk edges.

        Intent-conditioned traversal visits topic nodes; this distributes
        their scores to chunk nodes using TOPIC_CHUNK edge weights.
        """
        if self._graph is None:
            return {}

        from bettermem.core.nodes import NodeKind  # local import to avoid cycles

        chunk_scores: dict[str, float] = {}

        for node_id, score in scores.items():
            node = self._graph.get_node(node_id)
            if node is None:
                continue

            if node.kind == NodeKind.CHUNK:
                chunk_scores[node_id] = chunk_scores.get(node_id, 0.0) + float(score)
            elif node.kind == NodeKind.TOPIC:
                neighbors = self._graph.get_neighbors(node_id)
                for neigh_id, weight in neighbors.items():
                    neigh = self._graph.get_node(neigh_id)
                    if neigh is None or neigh.kind != NodeKind.CHUNK:
                        continue
                    chunk_scores[neigh_id] = chunk_scores.get(neigh_id, 0.0) + float(
                        score
                    ) * float(weight)

        return chunk_scores

    def _rerank_chunk_scores(
        self, query_text: str, chunk_scores: dict[str, float]
    ) -> dict[str, float]:
        """Rerank chunk scores by blending topic-projected score with query–chunk cosine (and optional BM25).

        S_chunk(c) = w_topic * norm(topic_score(c)) + w_cosine * norm(cos(q, c)) [+ w_bm25 * norm(bm25)].
        Weights from config; scores normalized to [0, 1] before blending.
        """
        if not chunk_scores:
            return chunk_scores
        wt = self.config.rerank_weight_topic
        wc = self.config.rerank_weight_cosine
        wb = self.config.rerank_weight_bm25
        total_w = wt + wc + wb
        if total_w <= 0:
            return chunk_scores

        from bettermem.core.nodes import ChunkNode

        if self._graph is None or self._topic_model is None:
            return chunk_scores

        query_emb: Optional[Sequence[float]] = None
        if wc > 0 or wb > 0:
            if hasattr(self._topic_model, "embed_query"):
                query_emb = self._topic_model.embed_query(query_text)
            if query_emb is not None:
                query_emb = list(query_emb)

        # Normalize topic scores to [0, 1]
        t_max = max(chunk_scores.values()) if chunk_scores else 0.0
        if t_max <= 0:
            norm_topic = {cid: 0.0 for cid in chunk_scores}
        else:
            norm_topic = {cid: s / t_max for cid, s in chunk_scores.items()}

        # Cosine similarity query–chunk; normalize to [0, 1]
        norm_cosine: dict[str, float] = {}
        if wc > 0 and query_emb is not None:
            chunk_ids = list(chunk_scores.keys())
            embeddings: List[Optional[Sequence[float]]] = []
            need_embed: List[Tuple[str, str]] = []
            for cid in chunk_ids:
                node = self._graph.get_node(cid)
                if isinstance(node, ChunkNode) and node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    text = ""
                    if node is not None:
                        raw = (node.metadata or {}).get("text")
                        text = raw if isinstance(raw, str) else ""
                    embeddings.append(None)
                    need_embed.append((cid, text))
            if need_embed and hasattr(self._topic_model, "embed_texts"):
                texts_to_embed = [t for (_, t) in need_embed]
                emb_list = self._topic_model.embed_texts(texts_to_embed)
                if emb_list is not None:
                    for (cid, _), emb in zip(need_embed, emb_list):
                        idx = chunk_ids.index(cid)
                        embeddings[idx] = list(emb) if emb is not None else None
            q_norm = (sum(x * x for x in query_emb)) ** 0.5
            if q_norm <= 0:
                q_norm = 1e-12
            cosines: List[float] = []
            for i, cid in enumerate(chunk_ids):
                emb = embeddings[i] if i < len(embeddings) else None
                if emb is None or len(emb) != len(query_emb):
                    cosines.append(0.0)
                    continue
                dot = sum(a * b for a, b in zip(query_emb, emb))
                n = (sum(x * x for x in emb)) ** 0.5
                n = n if n > 0 else 1e-12
                cos = dot / (q_norm * n)
                cos = max(-1.0, min(1.0, cos))
                cosines.append((cos + 1.0) * 0.5)
            norm_cosine = {cid: cosines[i] for i, cid in enumerate(chunk_ids)}
        else:
            norm_cosine = {cid: 0.0 for cid in chunk_scores}

        # Optional BM25 (stub: no-op when wb > 0 until implemented)
        norm_bm25: dict[str, float] = {cid: 0.0 for cid in chunk_scores}
        if wb > 0:
            # Placeholder: could add rank_bm25 or simple term overlap
            pass

        out: dict[str, float] = {}
        for cid in chunk_scores:
            s = (wt / total_w) * norm_topic.get(cid, 0.0) + (wc / total_w) * norm_cosine.get(cid, 0.0)
            if wb > 0:
                s += (wb / total_w) * norm_bm25.get(cid, 0.0)
            out[cid] = s
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the current index and configuration to disk."""
        if self._graph is None or self._transition_model is None:
            raise RuntimeError("No index to save. Build or load an index first.")
        save_index(
            path,
            graph=self._graph,
            transition_model=self._transition_model,
            config=self.config,
        )

    @classmethod
    def load(cls, path: str, *, mmap_mode: Optional[str] = None) -> "BetterMem":
        """Load a saved semantic hierarchical index and return a BetterMem instance.

        Parameters
        ----------
        path
            Directory containing graph.joblib, transition.joblib, and optional config.json.
        mmap_mode
            Optional joblib mmap mode (e.g. 'r') for large arrays.
        """
        graph, transition_model, config = load_index(path, mmap_mode=mmap_mode)
        instance = cls(config=config)
        instance._graph = graph
        instance._transition_model = transition_model
        instance._traversal_engine = TraversalEngine(graph=graph, transition_model=transition_model)
        instance._context_aggregator = ContextAggregator(graph=graph)
        return instance

