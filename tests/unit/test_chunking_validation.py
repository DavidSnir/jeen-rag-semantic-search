"""Tests for shared chunking validation, models, and invariants."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rag_app.cli import ChunkingStrategy as CliChunkingStrategy
from rag_app.documents import ChunkingStrategy, ExtractedDocument, ExtractedTextUnit
from rag_app.exceptions import (
    ChunkGenerationError,
    ChunkingError,
    InvalidChunkingInputError,
    InvalidChunkingStrategyError,
    RagAppError,
)
from rag_app.processing import chunking
from rag_app.processing.chunking import chunk_document, validate_chunking_strategy


def _document(
    text: str = "A short but meaningful heading.",
    *,
    source_type: str = "DOCX",
    page_number: int | None = None,
) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="source.docx" if source_type == "DOCX" else "source.pdf",
        source_type=source_type,  # type: ignore[arg-type]
        units=(ExtractedTextUnit(text=text, position=0, page_number=page_number),),
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("fixed", ChunkingStrategy.fixed),
        ("sentence", ChunkingStrategy.sentence),
        ("paragraph", ChunkingStrategy.paragraph),
        ("  fixed  ", ChunkingStrategy.fixed),
        ("SeNtEnCe", ChunkingStrategy.sentence),
        (ChunkingStrategy.paragraph, ChunkingStrategy.paragraph),
    ],
)
def test_validate_chunking_strategy_returns_canonical_value(
    requested: str | ChunkingStrategy, expected: ChunkingStrategy
) -> None:
    result = validate_chunking_strategy(requested)

    assert result is expected
    assert result.value == result.value.lower()


@pytest.mark.parametrize(
    "requested",
    ["", "   ", "unknown", "fixed-size", "sentences", "paragraphs"],
)
def test_validate_chunking_strategy_rejects_unsupported_values(
    requested: str,
) -> None:
    with pytest.raises(InvalidChunkingStrategyError) as caught:
        validate_chunking_strategy(requested)

    message = str(caught.value)
    assert "fixed" in message
    assert "sentence" in message
    assert "paragraph" in message


def test_validate_chunking_strategy_rejects_non_string_input() -> None:
    with pytest.raises(InvalidChunkingStrategyError, match="must be a string"):
        validate_chunking_strategy(None)  # type: ignore[arg-type]


def test_cli_and_processing_share_one_strategy_definition() -> None:
    assert CliChunkingStrategy is ChunkingStrategy
    assert chunking.ChunkingStrategy is ChunkingStrategy


def test_chunking_errors_use_application_exception_hierarchy() -> None:
    assert issubclass(InvalidChunkingStrategyError, ChunkingError)
    assert issubclass(InvalidChunkingInputError, ChunkingError)
    assert issubclass(ChunkGenerationError, ChunkingError)
    assert issubclass(ChunkingError, RagAppError)


def test_chunk_document_rejects_document_without_units() -> None:
    document = ExtractedDocument(
        source_file="empty.pdf",
        source_type="PDF",
        units=(),
    )

    with pytest.raises(InvalidChunkingInputError, match="at least one text unit"):
        chunk_document(document, "fixed")


@pytest.mark.parametrize("document", [None, object()])
def test_chunk_document_rejects_non_document_input_safely(document: object) -> None:
    with pytest.raises(
        InvalidChunkingInputError, match="must be an extracted document"
    ):
        chunk_document(document, "fixed")  # type: ignore[arg-type]


def test_chunk_document_rejects_invalid_document_metadata_safely() -> None:
    document = ExtractedDocument(
        source_file=None,  # type: ignore[arg-type]
        source_type="DOCX",
        units=(ExtractedTextUnit(text="content", position=0, page_number=None),),
    )

    with pytest.raises(InvalidChunkingInputError, match="source filename"):
        chunk_document(document, "fixed")


def test_chunk_document_rejects_non_tuple_units_safely() -> None:
    document = ExtractedDocument(
        source_file="source.docx",
        source_type="DOCX",
        units=[  # type: ignore[arg-type]
            ExtractedTextUnit(text="content", position=0, page_number=None)
        ],
    )

    with pytest.raises(InvalidChunkingInputError, match="ordered tuple"):
        chunk_document(document, "fixed")


def test_chunk_document_rejects_non_unit_members_safely() -> None:
    document = ExtractedDocument(
        source_file="source.docx",
        source_type="DOCX",
        units=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidChunkingInputError, match="extracted text units"):
        chunk_document(document, "fixed")


def test_chunk_document_rejects_unit_without_string_text_safely() -> None:
    document = ExtractedDocument(
        source_file="source.docx",
        source_type="DOCX",
        units=(
            ExtractedTextUnit(  # type: ignore[arg-type]
                text=None, position=0, page_number=None
            ),
        ),
    )

    with pytest.raises(InvalidChunkingInputError, match="must contain text"):
        chunk_document(document, "fixed")


@pytest.mark.parametrize("position", [-1, 1, True])
def test_chunk_document_rejects_invalid_unit_positions(
    position: int,
) -> None:
    document = ExtractedDocument(
        source_file="source.docx",
        source_type="DOCX",
        units=(ExtractedTextUnit(text="content", position=position, page_number=None),),
    )

    with pytest.raises(InvalidChunkingInputError, match="zero-based and continuous"):
        chunk_document(document, "fixed")


@pytest.mark.parametrize(
    "document",
    [
        _document(source_type="PDF", page_number=None),
        _document(source_type="PDF", page_number=0),
        _document(source_type="DOCX", page_number=1),
    ],
)
def test_chunk_document_rejects_invalid_page_metadata(
    document: ExtractedDocument,
) -> None:
    with pytest.raises(InvalidChunkingInputError):
        chunk_document(document, "fixed")


def test_chunk_document_rejects_invalid_source_type() -> None:
    document = _document(source_type="TXT")

    with pytest.raises(InvalidChunkingInputError, match="source type"):
        chunk_document(document, "fixed")


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunk_document_rejects_result_without_meaningful_content(
    strategy: ChunkingStrategy,
) -> None:
    with pytest.raises(ChunkGenerationError, match="meaningful"):
        chunk_document(_document("   \n\t  "), strategy)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_every_strategy_preserves_metadata_bounds_and_short_content(
    strategy: ChunkingStrategy,
) -> None:
    document = _document("Title.")

    result = chunk_document(document, f"  {strategy.value.upper()}  ")

    assert result.source_file == "source.docx"
    assert result.source_type == "DOCX"
    assert result.chunking_strategy is strategy
    assert [chunk.chunk_index for chunk in result.chunks] == list(
        range(len(result.chunks))
    )
    assert all(chunk.content for chunk in result.chunks)
    assert all(len(chunk.content) <= 2_000 for chunk in result.chunks)
    assert all(chunk.page_number is None for chunk in result.chunks)
    assert "Title." in " ".join(chunk.content for chunk in result.chunks)
    assert str(Path.cwd()) not in repr(result)
    assert document == _document("Title.")


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_every_strategy_preserves_pdf_pages_and_document_level_order(
    strategy: ChunkingStrategy,
) -> None:
    document = ExtractedDocument(
        source_file="pages.pdf",
        source_type="PDF",
        units=(
            ExtractedTextUnit(text="Page two.", position=0, page_number=2),
            ExtractedTextUnit(text="Page four.", position=1, page_number=4),
        ),
    )

    result = chunk_document(document, strategy)

    assert [chunk.page_number for chunk in result.chunks] == [2, 4]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]
    assert [chunk.content for chunk in result.chunks] == ["Page two.", "Page four."]


def test_chunk_models_are_immutable() -> None:
    result = chunk_document(_document(), "fixed")

    with pytest.raises(FrozenInstanceError):
        result.chunks[0].content = "replacement"  # type: ignore[misc]
