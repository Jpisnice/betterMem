"""Structure-aware chunker that respects sections and paragraph boundaries."""

from __future__ import annotations

import re
from typing import Iterable, List

from bettermem.indexing.chunker import Chunk


class StructureAwareChunker:
    """Chunker that tags chunks with structural_group by section/paragraph.

    Splits at heading patterns (numbered lines, all-caps short lines, markdown #)
    and blank-line paragraph breaks. Within each section, uses fixed-size token
    windows. Each section gets a unique structural_group (doc_idx * 10000 + section_idx).
    """

    def __init__(self, window_size: int = 200, overlap: int = 50) -> None:
        self._window_size = window_size
        self._overlap = overlap

    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections by headings and paragraph breaks."""
        lines = text.split("\n")
        sections: List[str] = []
        current: List[str] = []

        # Heading: line starting with #, or numbered (e.g. "1. Introduction"), or short all-caps
        markdown_heading = re.compile(r"^\s*#+\s+.+")
        numbered_heading = re.compile(r"^\s*\d+[.)]\s*\S")
        short_all_caps = re.compile(r"^[A-Z][A-Z0-9\s]{1,50}$")

        for line in lines:
            is_heading = (
                markdown_heading.match(line) is not None
                or numbered_heading.match(line.strip()) is not None
                or (len(line.strip()) <= 50 and line.strip() and short_all_caps.match(line.strip()) is not None)
            )
            is_blank = not line.strip()

            if is_heading and current:
                sections.append("\n".join(current))
                current = [line]
            elif is_blank and current:
                sections.append("\n".join(current))
                current = []
            elif is_blank:
                continue
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))
        return sections

    def chunk(self, corpus: Iterable[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_idx, document in enumerate(corpus):
            sections = self._split_into_sections(document)
            doc_id = str(doc_idx)
            position = 0
            for group_id, section_text in enumerate(sections):
                if not section_text.strip():
                    continue
                structural_group = doc_idx * 10000 + group_id
                tokens = section_text.split()
                start = 0
                while start < len(tokens):
                    end = min(start + self._window_size, len(tokens))
                    text = " ".join(tokens[start:end])
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
                    if end == len(tokens):
                        break
                    start = end - self._overlap
                    position += 1
        return chunks
