from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol


@dataclass
class Chunk:
    """Lightweight representation of a corpus chunk."""

    id: str
    document_id: str
    position: int
    text: str


class BaseChunker(Protocol):
    """Protocol for pluggable chunking strategies."""

    def chunk(self, corpus: Iterable[str]) -> List[Chunk]:
        """Split the corpus into chunks."""
        ...


class FixedWindowChunker:
    """Simple token-window-based chunker.

    This is a baseline implementation used for the initial indexing
    pipeline; more advanced strategies (sentence or semantic) can be
    added later behind the same interface.
    """

    def __init__(self, window_size: int = 200, overlap: int = 50) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, corpus: Iterable[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_idx, document in enumerate(corpus):
            tokens = document.split()
            start = 0
            position = 0
            doc_id = str(doc_idx)
            while start < len(tokens):
                end = min(start + self.window_size, len(tokens))
                text = " ".join(tokens[start:end])
                chunk_id = f"{doc_id}:{position}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc_id,
                        position=position,
                        text=text,
                    )
                )
                if end == len(tokens):
                    break
                start = end - self.overlap
                position += 1
        return chunks

