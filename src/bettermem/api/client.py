from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from bettermem.core.graph import Graph
from bettermem.core.transition_model import TransitionModel
from bettermem.core.traversal_engine import TraversalEngine
from bettermem.indexing.chunker import FixedWindowChunker
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
    def build_index(self, corpus: Iterable[str]) -> None:
        """Build an index over the given corpus.

        This method is responsible for:
        - Chunking documents.
        - Fitting or loading a topic model.
        - Constructing the topic/chunk graph.
        - Estimating transition probabilities.

        The concrete implementation is provided in later stages.
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
        strategy: Optional[str] = None,
        diversity: bool = True,
        exploration_factor: Optional[float] = None,
        entropy_penalty: Optional[float] = None,
        path_trace: bool = False,
    ) -> Sequence[Any]:
        """Run a retrieval query against the indexed corpus.

        Parameters
        ----------
        text:
            Natural language query.
        steps:
            Optional override for the number of traversal steps.
        top_k:
            Number of chunks or contexts to return.
        strategy:
            Traversal strategy to use; defaults to the configuration value.
        diversity:
            Whether to apply diversity-aware selection to the final contexts.
        exploration_factor:
            Optional override for exploration factor during traversal.
        entropy_penalty:
            Optional override for entropy penalty in scoring.
        path_trace:
            If True, implementation may retain and return an explanation
            trace describing the traversal paths.

        Returns
        -------
        Sequence[Any]
            A sequence of context objects; the exact type is defined in
            later stages when the indexing pipeline is implemented.
        """
        if self._graph is None or self._transition_model is None or self._traversal_engine is None:
            raise RuntimeError("Index has not been built or loaded.")

        strategy = strategy or self.config.traversal_strategy
        steps = steps or self.config.max_steps
        exploration_factor = (
            exploration_factor
            if exploration_factor is not None
            else self.config.exploration_factor
        )

        from bettermem.core.nodes import NodeKind

        # Compute topic prior
        prior: dict[str, float]
        start_pair: Optional[tuple[str, str]] = None

        if self._topic_model is not None:
            initializer = QueryInitializer(topic_model=self._topic_model)
            start_pair, prior_map = initializer.initial_state(text)
            prior = dict(prior_map)
        else:
            prior = {}

        if not prior:
            # Fallback: uniform prior over topic nodes
            topic_ids = [
                nid
                for nid, node in self._graph.nodes.items()
                if node.kind == NodeKind.TOPIC
            ]
            if not topic_ids:
                return []
            uniform = 1.0 / float(len(topic_ids))
            prior = {nid: uniform for nid in topic_ids}

        # Choose start nodes if not provided
        if start_pair is None:
            sorted_topics = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
            if len(sorted_topics) >= 2:
                start_pair = (sorted_topics[0][0], sorted_topics[1][0])

        results: Sequence[Any] = []
        explanation: dict[str, Any] = {"strategy": strategy, "prior": prior}

        if strategy == "beam":
            if start_pair is None:
                return []
            start_nodes = list(start_pair)
            traversal_res = self._traversal_engine.beam_search(
                start_nodes=start_nodes,
                steps=steps,
                beam_width=self.config.beam_width,
            )
            scorer = QueryScorer()
            scorer.add_visit_counts(traversal_res.visit_counts)
            scores = scorer.scores()
            contexts = self._context_aggregator.select(
                scores,
                top_k=top_k,
                diversity=diversity,
            )
            results = contexts
            if path_trace:
                explanation["paths"] = traversal_res.paths

        elif strategy == "random_walk":
            if start_pair is None:
                return []
            start_nodes = list(start_pair)
            traversal_res = self._traversal_engine.random_walk(
                start_nodes=start_nodes,
                steps=steps,
                exploration_factor=exploration_factor,
            )
            scorer = QueryScorer()
            scorer.add_visit_counts(traversal_res.visit_counts)
            scores = scorer.scores()
            contexts = self._context_aggregator.select(
                scores,
                top_k=top_k,
                diversity=diversity,
            )
            results = contexts
            if path_trace:
                explanation["paths"] = traversal_res.paths

        else:  # personalized_pagerank
            scores = self._traversal_engine.personalized_pagerank(
                prior,
                max_iters=steps,
            )
            contexts = self._context_aggregator.select(
                scores,
                top_k=top_k,
                diversity=diversity,
            )
            results = contexts

        self._last_explanation = explanation
        return results

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

