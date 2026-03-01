"""Semantic hierarchical topic model using embeddings and multi-level recursive clustering."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

from bettermem.topic_modeling.base import BaseTopicModel


def _path_to_topic_id(path: Sequence[int]) -> str:
    """Convert path of indices to string topic ID (e.g. [0, 1, 2] -> 't:0.1.2')."""
    if not path:
        return "t:"
    return "t:" + ".".join(str(p) for p in path)


def _topic_id_to_path(topic_id: str) -> List[int]:
    """Parse topic ID to path (e.g. 't:0.1.2' -> [0, 1, 2])."""
    if not topic_id or not topic_id.startswith("t:"):
        return []
    rest = topic_id[2:].strip()
    if not rest:
        return []
    return [int(x) for x in rest.split(".") if x.isdigit()]


def _depth(topic_id: str) -> int:
    """Depth of topic in tree (root = 0)."""
    return len(_topic_id_to_path(topic_id))


class SemanticHierarchicalTopicModel(BaseTopicModel):
    """Topic model that discovers a multi-level topic hierarchy via recursive clustering.

    - fit(): embeds documents, runs recursive KMeans to build a tree of topics.
    - transform(): returns P(leaf_topic | chunk) over string path IDs.
    - get_hierarchy(): returns parent_id -> [child_ids] (dict[str, list[str]]) for graph building.
    """

    def __init__(
        self,
        n_coarse: int = 5,
        n_fine_per_coarse: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        random_state: int | None = None,
        max_depth: int = 3,
        min_cluster_size: int = 2,
        k_next: Optional[int] = None,
        dag_tau: Optional[float] = None,
    ) -> None:
        self.n_coarse = max(1, n_coarse)
        self.n_fine_per_coarse = max(1, n_fine_per_coarse)
        self.embedding_model_name = embedding_model
        self.random_state = random_state
        self.max_depth = max(1, max_depth)
        self.min_cluster_size = max(1, min_cluster_size)
        self.k_next = n_fine_per_coarse if k_next is None else max(1, k_next)
        self.dag_tau = dag_tau  # if set, add multi-parent edges where cos >= dag_tau (acyclic)

        self._embedding_model = None
        self._hierarchy: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        self._parent_ids: dict[str, List[str]] = {}  # topic_id -> [parent_ids] (for DAG)
        self._centroids: dict[str, List[float]] = {}
        self._keywords_per_topic: dict[str, List[str]] = {}
        self._fitted_documents: List[str] = []
        self._fitted_embeddings = None
        self._leaf_ids: List[str] = []

    def _get_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def fit(self, documents: Iterable[str]) -> None:
        import numpy as np
        from sklearn.cluster import KMeans

        docs = list(documents)
        if not docs:
            self._hierarchy = {}
            self._parent_ids = {}
            self._centroids = {}
            self._keywords_per_topic = {}
            self._leaf_ids = []
            return

        model = self._get_embedding_model()
        embeddings = model.encode(docs, convert_to_numpy=True)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self._fitted_documents = docs
        self._fitted_embeddings = embeddings

        self._hierarchy = {}
        self._parent_ids = {}
        self._centroids = {}
        self._leaf_ids = []

        # Recursive clustering from root
        n_docs = len(docs)
        k0 = min(self.n_coarse, n_docs)
        if k0 < 1:
            return

        kmeans = KMeans(n_clusters=k0, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        root_centroids = kmeans.cluster_centers_

        for c in range(k0):
            path = [c]
            tid = _path_to_topic_id(path)
            self._centroids[tid] = root_centroids[c].tolist()
            mask = labels == c
            sub_emb = embeddings[mask]
            doc_indices = [i for i, m in enumerate(mask) if m]
            self._split_cluster(
                path=path,
                embeddings=embeddings,
                doc_indices=doc_indices,
                sub_emb=sub_emb,
                docs=docs,
            )

        self._compute_leaf_ids()
        if self.dag_tau is not None:
            self._add_dag_edges()
        self._build_keywords(docs)

    def _split_cluster(
        self,
        path: List[int],
        embeddings: "np.ndarray",
        doc_indices: List[int],
        sub_emb: "np.ndarray",
        docs: List[str],
    ) -> None:
        import numpy as np
        from sklearn.cluster import KMeans

        parent_id = _path_to_topic_id(path)
        depth = len(path)
        if depth >= self.max_depth or len(doc_indices) < self.min_cluster_size:
            return

        k = min(self.k_next, len(doc_indices))
        if k < 2:
            return

        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        sub_labels = kmeans.fit_predict(sub_emb)
        child_ids: List[str] = []
        for s in range(k):
            child_path = path + [s]
            child_id = _path_to_topic_id(child_path)
            child_ids.append(child_id)
            self._centroids[child_id] = kmeans.cluster_centers_[s].tolist()
            mask = sub_labels == s
            sub_sub_emb = sub_emb[mask]
            sub_doc_indices = [doc_indices[i] for i in range(len(doc_indices)) if mask[i]]
            self._split_cluster(
                path=child_path,
                embeddings=embeddings,
                doc_indices=sub_doc_indices,
                sub_emb=sub_sub_emb,
                docs=docs,
            )
        self._hierarchy[parent_id] = child_ids
        for cid in child_ids:
            self._parent_ids[cid] = [parent_id]

    def _compute_leaf_ids(self) -> None:
        """Set _leaf_ids to all topic IDs that have no children."""
        all_ids = set(self._centroids.keys())
        parent_ids = set(self._hierarchy.keys())
        self._leaf_ids = sorted(all_ids - parent_ids)

    def _ancestors_of(self, topic_id: str) -> set[str]:
        """Transitive closure of parents (for acyclicity check)."""
        anc: set[str] = set()
        stack = list(self._parent_ids.get(topic_id, []))
        while stack:
            p = stack.pop()
            if p in anc:
                continue
            anc.add(p)
            stack.extend(self._parent_ids.get(p, []))
        return anc

    def _descendants_of(self, topic_id: str) -> set[str]:
        """Transitive closure of children (subtree rooted at topic_id)."""
        result: set[str] = {topic_id}
        stack = list(self._hierarchy.get(topic_id, []))
        while stack:
            c = stack.pop()
            if c in result:
                continue
            result.add(c)
            stack.extend(self._hierarchy.get(c, []))
        return result

    def rollup_leaf_prior_to_ancestors(
        self, leaf_prior: Mapping[str, float]
    ) -> dict[str, float]:
        """Roll leaf prior up to all topics: P(topic|q) = sum(leaf_prior[leaf] for leaf in descendants(topic) and leaf is leaf)."""
        if not leaf_prior:
            return {}
        leaf_set = set(self._leaf_ids)
        all_ids = self.get_all_topic_ids()
        rolled: dict[str, float] = {}
        for topic_id in all_ids:
            desc = self._descendants_of(topic_id)
            leaves_in_subtree = desc & leaf_set
            rolled[topic_id] = sum(leaf_prior.get(leaf, 0.0) for leaf in leaves_in_subtree)
        total = sum(rolled.values())
        if total <= 0:
            return rolled
        inv = 1.0 / total
        return {tid: p * inv for tid, p in rolled.items()}

    def _add_dag_edges(self) -> None:
        """Add multi-parent edges where cos(centroid(t), centroid(u)) >= dag_tau, preserving acyclicity.

        Note: When dag_tau is None (recommended until asymmetric broader/narrower tests exist),
        similarity is kept only as TOPIC_TOPIC edges in the indexer, not in hierarchy/parent_ids.
        """
        import numpy as np

        tau = self.dag_tau
        if tau is None or tau > 1.0:
            return
        all_ids = list(self._centroids.keys())
        n = len(all_ids)
        if n < 2:
            return
        for t in all_ids:
            anc_t = self._ancestors_of(t)
            ct = np.array(self._centroids[t], dtype=np.float32)
            for u in all_ids:
                if u == t:
                    continue
                if u in anc_t:
                    continue
                anc_u = self._ancestors_of(u)
                if t in anc_u:
                    continue
                cu = np.array(self._centroids[u], dtype=np.float32)
                sim = float(np.dot(ct, cu) / (np.linalg.norm(ct) * np.linalg.norm(cu) + 1e-12))
                if sim < tau:
                    continue
                if u not in self._parent_ids.get(t, []):
                    self._parent_ids.setdefault(t, []).append(u)
                if u not in self._hierarchy:
                    self._hierarchy[u] = []
                if t not in self._hierarchy[u]:
                    self._hierarchy[u].append(t)

    def get_parents(self, topic_id: str) -> List[str]:
        """Return list of parent topic IDs (one for tree, multiple for DAG)."""
        return list(self._parent_ids.get(topic_id, []))

    def _build_keywords(self, docs: List[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._keywords_per_topic = {}
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        try:
            X = vectorizer.fit_transform(docs)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
            X = vectorizer.fit_transform(docs)

        feature_names = vectorizer.get_feature_names_out()
        if feature_names is None:
            feature_names = []

        for topic_id, centroid in self._centroids.items():
            # Find doc indices that belong to this topic (nearest centroid)
            import numpy as np
            emb = self._fitted_embeddings
            if emb is None or topic_id not in self._centroids:
                continue
            c = np.array(centroid)
            dists = np.linalg.norm(emb - c, axis=1)
            # Members: within-cluster (closer to this centroid than to any other for same parent)
            path = _topic_id_to_path(topic_id)
            parent_id = _path_to_topic_id(path[:-1]) if len(path) > 1 else None
            if parent_id and parent_id in self._hierarchy:
                sibling_ids = self._hierarchy[parent_id]
                rows = []
                for i in range(len(docs)):
                    best_tid = topic_id
                    best_d = float(dists[i])
                    for sid in sibling_ids:
                        if sid not in self._centroids:
                            continue
                        d = np.linalg.norm(emb[i] - np.array(self._centroids[sid]))
                        if d < best_d:
                            best_d = d
                            best_tid = sid
                    if best_tid == topic_id:
                        rows.append(i)
            else:
                # Root level: use simple threshold or top-k by distance
                k = min(50, max(1, len(docs) // max(1, len(self._centroids))))
                nearest = np.argsort(dists)[:k]
                rows = nearest.tolist()

            if not rows:
                self._keywords_per_topic[topic_id] = []
                continue
            sub_X = X[rows]
            if sub_X.shape[0] == 0:
                self._keywords_per_topic[topic_id] = []
                continue
            scores = sub_X.sum(axis=0).A1
            top = scores.argsort()[-10:][::-1]
            keywords = [str(feature_names[i]) for i in top if i < len(feature_names)]
            self._keywords_per_topic[topic_id] = keywords[:10]

    def get_hierarchy(self) -> dict[str, list[str]]:
        """Return parent_id -> list of child topic IDs."""
        return dict(self._hierarchy)

    def get_all_topic_ids(self) -> List[str]:
        """Return all topic IDs in no particular order (for indexer)."""
        return list(self._centroids.keys())

    def get_leaf_topic_ids(self) -> List[str]:
        """Return topic IDs that have no children (leaf nodes)."""
        return list(self._leaf_ids)

    def get_topic_keywords(self, topic_id: str, top_k: int = 10) -> List[str]:
        keywords = self._keywords_per_topic.get(topic_id, [])
        return keywords[:top_k]

    def transform(
        self, chunks: Iterable[str], *, temperature: float = 0.1
    ) -> Sequence[Mapping[str, float]]:
        """Return P(leaf_topic | chunk) for each chunk via cosine-similarity softmax.

        Uses cosine similarity (not negative L2) with temperature scaling so the
        distribution is peaked enough to produce topic-chunk edges above min_prob,
        even when there are many leaf topics.
        """
        import numpy as np

        chunk_list = list(chunks)
        if not chunk_list or not self._centroids:
            leaf = self.get_leaf_topic_ids()
            if not leaf:
                return [{} for _ in chunk_list]
            u = 1.0 / len(leaf)
            return [{tid: u for tid in leaf} for _ in chunk_list]

        model = self._get_embedding_model()
        emb = model.encode(chunk_list, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        leaf_ids = self.get_leaf_topic_ids()
        if not leaf_ids:
            all_ids = self.get_all_topic_ids()
            u = 1.0 / len(all_ids) if all_ids else 1.0
            return [{tid: u for tid in all_ids} for _ in chunk_list]

        centroid_matrix = np.array(
            [self._centroids[tid] for tid in leaf_ids], dtype=np.float32
        )
        centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        centroid_norms = np.maximum(centroid_norms, 1e-12)
        centroid_normed = centroid_matrix / centroid_norms

        result: List[Mapping[str, float]] = []
        for i in range(emb.shape[0]):
            e = emb[i].astype(np.float32)
            e_norm = np.linalg.norm(e)
            if e_norm < 1e-12:
                u = 1.0 / len(leaf_ids)
                result.append({tid: u for tid in leaf_ids})
                continue
            e_normed = e / e_norm
            cosines = centroid_normed @ e_normed  # shape (n_leaves,)
            probs = _softmax(cosines / max(temperature, 1e-6))
            total = float(probs.sum())
            if total <= 0:
                u = 1.0 / len(leaf_ids)
                result.append({tid: u for tid in leaf_ids})
            else:
                result.append(
                    {tid: float(probs[j]) / total for j, tid in enumerate(leaf_ids)}
                )
        return result

    def get_topic_distribution_for_query(self, text: str) -> Mapping[str, float]:
        return next(iter(self.transform([text])), {})

    def get_centroid(self, topic_id: str) -> Optional[Sequence[float]]:
        """Return the embedding centroid for this topic (path ID)."""
        c = self._centroids.get(topic_id)
        return list(c) if c is not None else None

    def embed_query(self, text: str) -> Optional[Sequence[float]]:
        if not self._centroids:
            return None
        model = self._get_embedding_model()
        emb = model.encode([text], convert_to_numpy=True)
        if emb.ndim == 1:
            return emb.tolist()
        return emb[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> Optional[Sequence[Sequence[float]]]:
        if not texts or not self._centroids:
            return None
        model = self._get_embedding_model()
        emb = model.encode(list(texts), convert_to_numpy=True)
        if emb.ndim == 1:
            return [emb.tolist()]
        return [emb[i].tolist() for i in range(emb.shape[0])]

    def get_coarse_centroid(self, coarse_id: int) -> Optional[Sequence[float]]:
        """Legacy: map root index to centroid. Prefer get_centroid('t:{coarse_id}')."""
        tid = _path_to_topic_id([coarse_id])
        return self.get_centroid(tid)


def _softmax(x) -> "np.ndarray":
    import numpy as np
    exp = np.exp(x - np.max(x))
    s = exp.sum()
    return exp / s if s > 0 else exp
