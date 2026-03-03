from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence


class EmbeddingBackend(ABC):
    """Abstract interface for embedding + vector-store backends.

    This is intentionally small and generic so BetterMem can work with
    different providers (Chroma, pgvector, Qdrant, in-process models, etc.).
    """

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return embeddings for a batch of texts.

        Implementations SHOULD:
        - Return one embedding per input text (same length as `texts`).
        - Use a deterministic, fixed-size vector space per backend instance.
        """

    def embed_query(self, text: str) -> Sequence[float]:
        """Return an embedding for a single query string.

        Default implementation calls embed_texts([text]) and returns the first
        vector. Backends can override for efficiency.
        """

        vectors = self.embed_texts([text])
        return list(vectors[0]) if vectors else []

    @abstractmethod
    def upsert_chunks(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        """Upsert chunk documents into the backend's vector store.

        The ID space MUST be stable and shared with BetterMem's graph:
        `ChunkNode.id` should correspond to the backend's document/row ID.
        """

    @abstractmethod
    def query_chunks(
        self,
        *,
        query_text: str,
        top_k: int,
        where: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunk matches for the query.

        The return format is a list of dictionaries with at least:
        - 'id': str (chunk id)
        - 'score': float (higher = more similar; caller may normalize)
        - optionally 'metadata': arbitrary metadata attached at upsert time
        """


class ChromaEmbeddingBackend(EmbeddingBackend):
    """ChromaDB-backed embedding + vector-store implementation.

    This wraps a `chromadb.Collection` (or a Client + collection name)
    and uses its configured embedding function for both:
    - Generating embeddings
    - Storing / querying chunk documents
    """

    def __init__(
        self,
        *,
        collection: Any = None,
        client: Any = None,
        collection_name: Optional[str] = None,
        collection_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._collection = collection
        if self._collection is None:
            if client is None or collection_name is None:
                raise ValueError(
                    "ChromaEmbeddingBackend requires either a `collection` "
                    "or both `client` and `collection_name`."
                )
            try:
                kwargs = dict(collection_kwargs or {})
                if "name" in kwargs:
                    raise ValueError(
                        "Do not pass `name` inside collection_kwargs; use collection_name instead."
                    )
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover - depends on Chroma runtime
                raise RuntimeError(
                    f"Failed to create or access Chroma collection '{collection_name}'. "
                    "Ensure Chroma is running and the client is correctly configured."
                ) from exc

    @property
    def collection(self) -> Any:
        return self._collection

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        # Chroma's embedding functions are primarily exposed via collection.add/upsert/query.
        # To keep this backend generic and avoid relying on private APIs,
        # we first try the configured embedding_function hooks.
        if not texts:
            return []

        # Most Chroma Collection objects store the configured EF on a private
        # attribute `_embedding_function`.
        private_embed_fn = getattr(self._collection, "_embedding_function", None)
        if callable(private_embed_fn):
            vectors = private_embed_fn(list(texts))
            return [list(v) for v in vectors]

        # Fast path: if the underlying collection exposes an embedding function, use it directly.
        embed_fn = getattr(self._collection, "embedding_function", None)
        if callable(embed_fn):
            vectors = embed_fn(list(texts))
            return [list(v) for v in vectors]

        # Fallback: use collection._client's embedding function if available.
        client = getattr(self._collection, "_client", None)
        client_embed_fn = getattr(client, "embedding_function", None) if client is not None else None
        if callable(client_embed_fn):
            vectors = client_embed_fn(list(texts))
            return [list(v) for v in vectors]

        raise RuntimeError(
            "ChromaEmbeddingBackend could not find an embedding function on the "
            "collection or client. Configure the Chroma collection with an "
            "embedding_function (see Chroma docs on embedding functions)."
        )

    def upsert_chunks(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length.")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas, when provided, must match length of ids.")

        kwargs: Dict[str, Any] = {
            "ids": list(ids),
            "documents": list(texts),
        }
        if metadatas is not None:
            kwargs["metadatas"] = [dict(m) for m in metadatas]

        try:
            # Rely on collection's configured embedding function for vectorization + storage.
            self._collection.upsert(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on Chroma runtime
            raise RuntimeError(
                "ChromaEmbeddingBackend failed during upsert_chunks. "
                "Check that the Chroma server is reachable and the collection "
                "accepts the provided ids/documents/metadatas."
            ) from exc

    def query_chunks(
        self,
        *,
        query_text: str,
        top_k: int,
        where: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": max(1, int(top_k)),
        }
        if where:
            params["where"] = dict(where)

        try:
            res = self._collection.query(**params)
        except Exception as exc:  # pragma: no cover - depends on Chroma runtime
            raise RuntimeError(
                "ChromaEmbeddingBackend.query_chunks failed when querying Chroma. "
                "Verify that the collection is available and the query parameters are valid."
            ) from exc

        ids_nested = res.get("ids") or [[]]
        dists_nested = res.get("distances") or [[]]
        metas_nested = res.get("metadatas") or [[]]

        ids = ids_nested[0] if ids_nested else []
        dists = dists_nested[0] if dists_nested else []
        metas = metas_nested[0] if metas_nested else []

        out: List[Dict[str, Any]] = []
        for i, cid in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 0.0
            # Convert distance to a similarity score in [0, 1] (rough heuristic).
            score = 1.0 / (1.0 + dist)
            meta = metas[i] if i < len(metas) else None
            out.append({"id": cid, "score": score, "metadata": meta})
        return out

