"""Deterministic chunking strategies for parser-independent documents."""

from functools import cache
import re
from typing import TYPE_CHECKING

from rag_app.documents import (
    Chunk,
    ChunkedDocument,
    ChunkingStrategy,
    ExtractedDocument,
    ExtractedTextUnit,
)
from rag_app.exceptions import (
    ChunkGenerationError,
    InvalidChunkingInputError,
    InvalidChunkingStrategyError,
)

if TYPE_CHECKING:
    from spacy.language import Language

MAX_CHUNK_SIZE = 2_000
CHUNK_OVERLAP = 500
CHUNK_STEP = MAX_CHUNK_SIZE - CHUNK_OVERLAP

_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")
_SUPPORTED_STRATEGIES = ", ".join(strategy.value for strategy in ChunkingStrategy)
_RawChunk = tuple[str, int | None]


def validate_chunking_strategy(
    strategy: str | ChunkingStrategy,
) -> ChunkingStrategy:
    """Normalize an external strategy value to its canonical representation."""
    if not isinstance(strategy, str):
        raise InvalidChunkingStrategyError(
            "Chunking strategy must be a string; "
            f"supported values: {_SUPPORTED_STRATEGIES}"
        )

    normalized = strategy.strip().lower()
    try:
        return ChunkingStrategy(normalized)
    except ValueError as error:
        raise InvalidChunkingStrategyError(
            "Unsupported chunking strategy. "
            f"Use one of: {_SUPPORTED_STRATEGIES}."
        ) from error


def chunk_document(
    document: ExtractedDocument,
    strategy: str | ChunkingStrategy,
) -> ChunkedDocument:
    """Chunk one extracted document without reopening or mutating its source."""
    canonical_strategy = validate_chunking_strategy(strategy)
    _validate_document(document)

    if canonical_strategy is ChunkingStrategy.fixed:
        raw_chunks = _chunk_fixed(document)
    elif canonical_strategy is ChunkingStrategy.sentence:
        raw_chunks = _chunk_sentences(document)
    else:
        raw_chunks = _chunk_paragraphs(document)

    meaningful_chunks: list[_RawChunk] = []
    for content, page_number in raw_chunks:
        trimmed_content = content.strip()
        if not trimmed_content:
            continue
        if len(trimmed_content) > MAX_CHUNK_SIZE:
            raise ChunkGenerationError(
                f"Chunking produced content longer than {MAX_CHUNK_SIZE} characters"
            )
        meaningful_chunks.append((trimmed_content, page_number))

    if not meaningful_chunks:
        raise ChunkGenerationError(
            "Chunking did not produce any meaningful document content"
        )

    chunks = tuple(
        Chunk(
            content=content,
            chunk_index=chunk_index,
            page_number=page_number,
        )
        for chunk_index, (content, page_number) in enumerate(meaningful_chunks)
    )
    return ChunkedDocument(
        source_file=document.source_file,
        source_type=document.source_type,
        chunking_strategy=canonical_strategy,
        chunks=chunks,
    )


def _validate_document(document: ExtractedDocument) -> None:
    if not document.units:
        raise InvalidChunkingInputError(
            "Extracted document must contain at least one text unit"
        )
    if document.source_type not in {"PDF", "DOCX"}:
        raise InvalidChunkingInputError(
            f"Unsupported extracted source type '{document.source_type}'"
        )

    for unit in document.units:
        if not isinstance(unit.text, str):
            raise InvalidChunkingInputError("Extracted text units must contain text")
        if document.source_type == "PDF":
            if (
                not isinstance(unit.page_number, int)
                or isinstance(unit.page_number, bool)
                or unit.page_number < 1
            ):
                raise InvalidChunkingInputError(
                    "Every PDF text unit must have a positive physical page number"
                )
        elif unit.page_number is not None:
            raise InvalidChunkingInputError(
                "DOCX text units must not contain page numbers"
            )


def _split_fixed_size(text: str) -> list[str]:
    """Return ordered 2,000-character windows with the canonical overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_SIZE, len(text))
        window = text[start:end].strip()
        if window:
            chunks.append(window)
        if end == len(text):
            break
        start += CHUNK_STEP
    return chunks


def _chunk_fixed(document: ExtractedDocument) -> list[_RawChunk]:
    if document.source_type == "PDF":
        return [
            (content, unit.page_number)
            for unit in document.units
            if unit.text.strip()
            for content in _split_fixed_size(unit.text)
        ]

    document_text = "\n\n".join(
        unit.text for unit in document.units if unit.text.strip()
    )
    return [(content, None) for content in _split_fixed_size(document_text)]


@cache
def _get_sentencizer() -> "Language":
    """Create the minimal English sentence pipeline only when first needed."""
    import spacy

    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    return pipeline


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []

    try:
        pipeline = _get_sentencizer()
        if len(text) >= pipeline.max_length:
            pipeline.max_length = len(text) + 1
        parsed_text = pipeline(text)
    except (RuntimeError, ValueError) as error:
        raise ChunkGenerationError(
            "The English sentence segmenter could not process document content"
        ) from error

    return [sentence for span in parsed_text.sents if (sentence := span.text.strip())]


def _group_sentences(sentences: list[str]) -> list[str]:
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_fixed_size(sentence))
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= MAX_CHUNK_SIZE:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def _chunk_sentences(document: ExtractedDocument) -> list[_RawChunk]:
    if document.source_type == "PDF":
        return [
            (content, unit.page_number)
            for unit in document.units
            for content in _group_sentences(_split_sentences(unit.text))
        ]

    sentences = [
        sentence
        for unit in document.units
        for sentence in _split_sentences(unit.text)
    ]
    return [(content, None) for content in _group_sentences(sentences)]


def _split_pdf_paragraphs(unit: ExtractedTextUnit) -> list[str]:
    return [
        paragraph
        for block in _PARAGRAPH_BOUNDARY.split(unit.text)
        if (paragraph := block.strip())
    ]


def _group_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_group_sentences(_split_sentences(paragraph)))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= MAX_CHUNK_SIZE:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _chunk_paragraphs(document: ExtractedDocument) -> list[_RawChunk]:
    if document.source_type == "PDF":
        return [
            (content, unit.page_number)
            for unit in document.units
            for content in _group_paragraphs(_split_pdf_paragraphs(unit))
        ]

    paragraphs = [unit.text.strip() for unit in document.units if unit.text.strip()]
    return [(content, None) for content in _group_paragraphs(paragraphs)]


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_STEP",
    "MAX_CHUNK_SIZE",
    "chunk_document",
    "validate_chunking_strategy",
]
