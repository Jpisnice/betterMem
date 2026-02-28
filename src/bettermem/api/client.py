from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from bettermem.core.graph import Graph
from bettermem.core.navigation_policy import IntentConditionedPolicy
from bettermem.core.transition_model import TransitionModel
from bettermem.core.traversal_engine import TraversalEngine
from bettermem.indexing.chunker import FixedWindowChunker
from bettermem.retrieval.intent import TraversalIntent, classify_intent_heuristic
from bettermem.indexing.corpus_indexer import CorpusIndexer
from bettermem.retrieval.context_aggregator import ContextAggregator
from bettermem.retrieval.query_initializer import QueryInitializer
from bettermem.retrieval.scorer import QueryScorer
from bettermem.storage.persistence import load_index, save_index
from bettermem.topic_modeling.bertopic_adapter import BERTopicAdapter
from .config import BetterMemConfig


class BetterMem:
    """High-level interface for the BetterMem retrieval system.

    This facade orchestrates indexing, traversal, and retrieval over the
    underlying probabilistic topic transition graph. At this stage, only
    method signatures and configuration wiring are defined; concrete
    implementations are added in later stages of the plan.
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
            Optional BetterMemConfig instance. If omitted, a default
            configuration is created.
        topic_model:
            Optional pre-initialized topic model adapter. When None, the
            appropriate backend is constructed during indexing based on
            configuration.
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
        """Build an index over the given corpus.

        This method is responsible for:
        - Chunking documents.
        - Fitting or loading a topic model.
        - Constructing the topic/chunk graph.
        - Estimating transition probabilities.

        Parameters
        ----------
        corpus:
            Documents to index.
        chunker:
            Optional chunker instance. If None, uses FixedWindowChunker().
        """
        # Initialize components
        self._graph = Graph()
        self._transition_model = TransitionModel(
            smoothing_lambda=self.config.smoothing_lambda
        )

        if self._topic_model is None:
            if self.config.topic_model_backend == "bertopic":
                self._topic_model = BERTopicAdapter()
            else:
                raise ValueError(
                    "A topic_model must be provided for backend "
                    f"{self.config.topic_model_backend!r}."
                )

        if chunker is None:
            chunker = FixedWindowChunker()
        self._indexer = CorpusIndexer(
            chunker=chunker,
            topic_model=self._topic_model,
            graph=self._graph,
            transition_model=self._transition_model,
        )
        self._indexer.build_index(corpus)

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
        """Run a retrieval query using intent-conditioned navigation.

        Parameters
        ----------
        text:
            Natural language query.
        steps:
            Optional override for the number of traversal steps.
        top_k:
            Number of chunks or contexts to return.
        diversity:
            Whether to apply diversity-aware selection to the final contexts.
        path_trace:
            If True, explanation will include the traversal path.
        intent:
            Optional override for traversal intent; default is heuristic from query text.

        Returns
        -------
        Sequence[Any]
            A sequence of context objects.
        """
        if self._graph is None or self._transition_model is None or self._traversal_engine is None:
            raise RuntimeError("Index has not been built or loaded.")

        steps = steps or self.config.max_steps

        from bettermem.core.nodes import NodeKind

        prior: dict[str, float] = {}
        semantic_state = None
        start_node: Optional[str] = None

        if self._topic_model is not None:
            initializer = QueryInitializer(topic_model=self._topic_model)
            start_pair, prior_map = initializer.initial_state(text)
            prior = dict(prior_map)
            semantic_state = initializer.semantic_state(text)
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

        intent_val = intent if intent is not None else classify_intent_heuristic(text)
        policy = IntentConditionedPolicy(
            self._graph,
            alpha=self.config.navigation_alpha,
            beta=self.config.navigation_beta,
            gamma=self.config.navigation_gamma,
            delta=self.config.navigation_delta,
        )
        traversal_res = self._traversal_engine.intent_conditioned_navigate(
            start_nodes=[start_node],
            steps=steps,
            intent=intent_val,
            semantic_state=semantic_state,
            policy=policy,
            temperature=self.config.navigation_temperature,
            greedy=self.config.navigation_greedy,
        )
        scorer = QueryScorer()
        scorer.add_visit_counts(traversal_res.visit_counts)
        scores = scorer.scores()
        scores = self._scores_to_chunk_space(scores)
        contexts = self._context_aggregator.select(
            scores,
            top_k=top_k,
            diversity=diversity,
        )
        explanation = {
            "strategy": "intent_conditioned",
            "intent": intent_val.value,
            "prior": prior,
        }
        if path_trace:
            explanation["paths"] = traversal_res.paths
        self._last_explanation = explanation
        return list(contexts)

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def explain(self) -> Any:
        """Return a structured explanation of the last query.

        The explanation is expected to include:
        - Traversal path(s).
        - Transition probabilities.
        - Topic transitions and their associated keywords.
        """
        return self._last_explanation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scores_to_chunk_space(self, scores: dict[str, float]) -> dict[str, float]:
        """Project node-level scores into chunk space using the graph structure.

        Beam search and random walk operate over topic nodes; this helper
        converts their visit counts into scores over chunk nodes by
        distributing mass along topic→chunk edges. If scores already
        contain chunk nodes, those are preserved.
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
    def load(cls, path: str) -> "BetterMem":
        """Load an index and configuration from disk and return a client."""
        graph, transition_model, config = load_index(path)
        instance = cls(config=config)
        instance._graph = graph
        instance._transition_model = transition_model
        instance._traversal_engine = TraversalEngine(graph=graph, transition_model=transition_model)
        instance._context_aggregator = ContextAggregator(graph=graph)
        return instance

