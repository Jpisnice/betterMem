"""Semantic hierarchical topic model using embeddings and two-level clustering."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

from bettermem.topic_modeling.base import BaseTopicModel


def _encode_topic_id(coarse: int, sub: int) -> int:
    """Encode (coarse, sub) as a single int: coarse_id * 100 + sub_idx."""
    return coarse * 100 + sub


def _decode_topic_id(topic_id: int) -> tuple[int, int]:
    """Decode topic_id into (coarse, sub)."""
    return topic_id // 100, topic_id % 100


class SemanticHierarchicalTopicModel(BaseTopicModel):
    """Topic model that discovers coarse and fine topics via embeddings and clustering.

    - fit(): embeds documents with sentence-transformers, runs coarse KMeans,
      then sub-clusters within each coarse cluster to get a two-level hierarchy.
    - transform(): returns P(subtopic | chunk) over encoded topic IDs.
    - get_hierarchy(): returns coarse_id -> [subtopic_ids] for graph building.
    """

    def __init__(
        self,
        n_coarse: int = 5,
        n_fine_per_coarse: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        random_state: int | None = None,
    ) -> None:
        self.n_coarse = max(1, n_coarse)
        self.n_fine_per_coarse = max(1, n_fine_per_coarse)
        self.embedding_model_name = embedding_model
        self.random_state = random_state

        self._embedding_model = None
        self._coarse_centers = None  # (n_coarse, dim)
        self._fine_centers: List[List[List[float]]] = []  # coarse -> list of fine center vectors
        self._hierarchy: dict[int, list[int]] = {}  # coarse_id -> [encoded subtopic ids]
        self._keywords_per_topic: dict[int, List[str]] = {}  # encoded topic_id -> keywords
        self._fitted_documents: List[str] = []
        self._fitted_embeddings = None  # (n_docs, dim) for transform fallback

    def _get_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def fit(self, documents: Iterable[str]) -> None:
        from sklearn.cluster import KMeans

        docs = list(documents)
        if not docs:
            self._hierarchy = {}
            self._keywords_per_topic = {}
            return

        model = self._get_embedding_model()
        embeddings = model.encode(docs, convert_to_numpy=True)
        self._fitted_documents = docs
        self._fitted_embeddings = embeddings

        n_coarse = min(self.n_coarse, len(docs))
        kmeans_coarse = KMeans(n_clusters=n_coarse, random_state=self.random_state, n_init=10)
        coarse_labels = kmeans_coarse.fit_predict(embeddings)
        self._coarse_centers = kmeans_coarse.cluster_centers_

        self._fine_centers = []
        self._hierarchy = {}
        fine_labels_full = [-1] * len(docs)

        for c in range(n_coarse):
            mask = coarse_labels == c
            sub_emb = embeddings[mask]
            if len(sub_emb) < 2:
                sub_emb = embeddings
            n_fine = min(self.n_fine_per_coarse, len(sub_emb))
            kmeans_fine = KMeans(n_clusters=n_fine, random_state=self.random_state, n_init=10)
            fine_labels = kmeans_fine.fit_predict(sub_emb)
            self._fine_centers.append(kmeans_fine.cluster_centers_.tolist())
            sub_ids = [_encode_topic_id(c, s) for s in range(n_fine)]
            self._hierarchy[c] = sub_ids
            doc_indices = [i for i, m in enumerate(mask) if m]
            for idx, sub_idx in zip(doc_indices, fine_labels):
                fine_labels_full[idx] = int(sub_idx)

        self._build_keywords(docs, coarse_labels, fine_labels_full, n_coarse)

    def _build_keywords(
        self,
        docs: List[str],
        coarse_labels: Sequence[int],
        fine_labels_full: Sequence[int],
        n_coarse: int,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._keywords_per_topic = {}
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
        try:
            X = vectorizer.fit_transform(docs)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
            X = vectorizer.fit_transform(docs)

        feature_names = vectorizer.get_feature_names_out()
        if feature_names is None:
            feature_names = []

        for coarse_id in range(n_coarse):
            sub_emb_list = self._fine_centers[coarse_id] if coarse_id < len(self._fine_centers) else []
            n_fine = len(sub_emb_list)
            for sub_idx in range(n_fine):
                tid = _encode_topic_id(coarse_id, sub_idx)
                rows = [
                    i
                    for i in range(len(docs))
                    if coarse_labels[i] == coarse_id and fine_labels_full[i] == sub_idx
                ]
                if not rows:
                    continue
                sub_X = X[rows]
                if sub_X.shape[0] == 0:
                    self._keywords_per_topic[tid] = []
                    continue
                scores = sub_X.sum(axis=0).A1
                top = scores.argsort()[-10:][::-1]
                keywords = [str(feature_names[i]) for i in top if i < len(feature_names)]
                self._keywords_per_topic[tid] = keywords[:10]

    def get_hierarchy(self) -> dict[int, list[int]]:
        """Return coarse_id -> list of encoded subtopic IDs."""
        return dict(self._hierarchy)

    def transform(self, chunks: Iterable[str]) -> Sequence[Mapping[int, float]]:
        import numpy as np

        chunk_list = list(chunks)
        if not chunk_list or self._coarse_centers is None:
            n = len(self._hierarchy) * self.n_fine_per_coarse
            if n == 0:
                n = 1
            flat = list(self._all_topic_ids())
            if not flat:
                return [{} for _ in chunk_list]
            u = 1.0 / len(flat)
            return [{tid: u for tid in flat} for _ in chunk_list]

        model = self._get_embedding_model()
        emb = model.encode(chunk_list, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        result: List[Mapping[int, float]] = []
        for i in range(emb.shape[0]):
            e = emb[i]
            dists_coarse = np.linalg.norm(self._coarse_centers - e, axis=1)
            probs_coarse = _softmax(-dists_coarse)
            dist_over_sub: List[tuple[int, float]] = []
            for coarse_id, p_c in enumerate(probs_coarse):
                if coarse_id >= len(self._fine_centers):
                    continue
                centers = np.array(self._fine_centers[coarse_id])
                dists_fine = np.linalg.norm(centers - e, axis=1)
                probs_fine = _softmax(-dists_fine)
                for sub_idx, p_s in enumerate(probs_fine):
                    tid = _encode_topic_id(coarse_id, sub_idx)
                    dist_over_sub.append((tid, float(p_c * p_s)))
            total = sum(p for _, p in dist_over_sub)
            if total <= 0:
                ids = self._all_topic_ids()
                u = 1.0 / len(ids) if ids else 1.0
                result.append({tid: u for tid in ids} if ids else {})
            else:
                result.append({tid: p / total for tid, p in dist_over_sub})
        return result

    def _all_topic_ids(self) -> List[int]:
        ids: List[int] = []
        for sub_ids in self._hierarchy.values():
            ids.extend(sub_ids)
        return ids if ids else [0]

    def get_topic_keywords(self, topic_id: int, top_k: int = 10) -> List[str]:
        keywords = self._keywords_per_topic.get(topic_id, [])
        return keywords[:top_k]

    def get_topic_distribution_for_query(self, text: str) -> Mapping[int, float]:
        return next(iter(self.transform([text])), {})

    def get_centroid(self, topic_id: int) -> Optional[Sequence[float]]:
        """Return the embedding centroid for this topic (encoded coarse*100+sub)."""
        if self._coarse_centers is None:
            return None
        coarse_id, sub_idx = _decode_topic_id(topic_id)
        if coarse_id >= len(self._fine_centers):
            return None
        fine_list = self._fine_centers[coarse_id]
        if sub_idx >= len(fine_list):
            return None
        return list(fine_list[sub_idx])

    def embed_query(self, text: str) -> Optional[Sequence[float]]:
        """Return the query embedding using the same encoder as chunks."""
        if self._coarse_centers is None:
            return None
        model = self._get_embedding_model()
        emb = model.encode([text], convert_to_numpy=True)
        if emb.ndim == 1:
            return emb.tolist()
        return emb[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> Optional[Sequence[Sequence[float]]]:
        """Return embedding vectors for a batch of texts (same encoder as chunks)."""
        if not texts or self._coarse_centers is None:
            return None
        model = self._get_embedding_model()
        emb = model.encode(list(texts), convert_to_numpy=True)
        if emb.ndim == 1:
            return [emb.tolist()]
        return [emb[i].tolist() for i in range(emb.shape[0])]

    def get_coarse_centroid(self, coarse_id: int) -> Optional[Sequence[float]]:
        """Return the coarse-level cluster center for hierarchy root nodes."""
        if self._coarse_centers is None or coarse_id >= len(self._coarse_centers):
            return None
        return self._coarse_centers[coarse_id].tolist()


def _softmax(x) -> "np.ndarray":
    import numpy as np
    exp = np.exp(x - np.max(x))
    s = exp.sum()
    return exp / s if s > 0 else exp
