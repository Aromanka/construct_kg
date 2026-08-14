from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    character_start: int
    character_end: int
    content: str


def chunk_document(
    content: str, *, chunk_size: int | None = None, chunk_overlap: int = 0
) -> list[DocumentChunk]:
    """Split text into deterministic character windows when chunking is requested."""

    if chunk_size is None:
        if chunk_overlap:
            raise ValueError("chunk_overlap requires chunk_size")
        return [DocumentChunk(0, 0, len(content), content)]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    step = chunk_size - chunk_overlap
    for index, start in enumerate(range(0, len(content), step)):
        end = min(start + chunk_size, len(content))
        chunks.append(DocumentChunk(index, start, end, content[start:end]))
        if end == len(content):
            break
    return chunks
