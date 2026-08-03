"""Shared document representations independent of parser libraries."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

SourceType = Literal["PDF", "DOCX"]
EmbeddingVector = tuple[float, ...]


class ChunkingStrategy(str, Enum):
    """Canonical chunking strategies shared by every application boundary."""

    fixed = "fixed"
    sentence = "sentence"
    paragraph = "paragraph"


class IndexingStatus(str, Enum):
    """Canonical outcomes for one document-indexing operation."""

    indexed = "indexed"
    replaced = "replaced"
    skipped = "skipped"


@dataclass(frozen=True, slots=True)
class ExtractedTextUnit:
    """One ordered piece of cleaned text from a source document."""

    text: str
    position: int
    page_number: int | None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Cleaned source content ready for later chunk construction."""

    source_file: str
    source_type: SourceType
    units: tuple[ExtractedTextUnit, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One ordered piece of document content ready for later embedding."""

    content: str
    chunk_index: int
    page_number: int | None


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    """Immutable chunking result with source-level persistence metadata."""

    source_file: str
    source_type: SourceType
    chunking_strategy: ChunkingStrategy
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """One original chunk paired with its normalized embedding vector."""

    chunk: Chunk
    embedding: EmbeddingVector


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    """Immutable document result ready for a later persistence stage."""

    source_file: str
    source_type: SourceType
    chunking_strategy: ChunkingStrategy
    chunks: tuple[EmbeddedChunk, ...]


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Safe application result returned after indexing completes successfully."""

    status: IndexingStatus
    source_file: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    elapsed_seconds: float
