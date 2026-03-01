from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol


@dataclass
class Chunk:
    """Lightweight representation of a corpus chunk."""

    id: str
    document_id: str
    position: int
    text: str
    structural_group: Optional[int] = None


class BaseChunker(Protocol):
    """Protocol for pluggable chunking strategies."""

    def chunk(self, corpus: Iterable[str]) -> List[Chunk]:
        """Split the corpus into chunks."""
        ...


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by blank lines (one or more newlines)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences; keeps whole sentences only."""
    # Split on sentence boundaries: . ! ? followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


class ParagraphSentenceChunker:
    """Chunk by paragraphs, limiting at entire sentences.

    Splits each document into paragraphs (blank-line separated), then splits
    each paragraph into sentences. Emits chunks that are one or more whole
    sentences and never cross paragraph boundaries. Single sentences that
    exceed max_tokens are emitted as their own chunk.
    """

    def __init__(self, max_tokens: int = 200) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens

    def chunk(self, corpus: Iterable[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_idx, document in enumerate(corpus):
            doc_id = str(doc_idx)
            position = 0
            paragraphs = _split_paragraphs(document)
            for group_id, para in enumerate(paragraphs):
                if not para.strip():
                    continue
                structural_group = doc_idx * 10000 + group_id
                sentences = _split_sentences(para)
                if not sentences:
                    chunk_id = f"{doc_id}:{position}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=doc_id,
                            position=position,
                            text=para,
                            structural_group=structural_group,
                        )
                    )
                    position += 1
                    continue
                acc: List[str] = []
                acc_tokens = 0
                for sent in sentences:
                    sent_tokens = len(sent.split())
                    if acc and acc_tokens + sent_tokens > self.max_tokens:
                        text = " ".join(acc)
                        chunk_id = f"{doc_id}:{position}"
                        chunks.append(
                            Chunk(
                                id=chunk_id,
                                document_id=doc_id,
                                position=position,
                                text=text,
                                structural_group=structural_group,
                            )
                        )
                        position += 1
                        acc = []
                        acc_tokens = 0
                    acc.append(sent)
                    acc_tokens += sent_tokens
                if acc:
                    text = " ".join(acc)
                    chunk_id = f"{doc_id}:{position}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=doc_id,
                            position=position,
                            text=text,
                            structural_group=structural_group,
                        )
                    )
                    position += 1
        return chunks


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

