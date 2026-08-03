"""Tests for deterministic fixed-size character chunking."""

import pytest

from rag_app.documents import ExtractedDocument, ExtractedTextUnit
from rag_app.exceptions import ChunkGenerationError
from rag_app.processing.chunking import (
    CHUNK_OVERLAP,
    CHUNK_STEP,
    MAX_CHUNK_SIZE,
    chunk_document,
)


def _docx(*texts: str) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="document.docx",
        source_type="DOCX",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=None)
            for index, text in enumerate(texts)
        ),
    )


def _pdf(*pages: tuple[str, int]) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="document.pdf",
        source_type="PDF",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=page_number)
            for index, (text, page_number) in enumerate(pages)
        ),
    )


def _pattern(length: int) -> str:
    return "".join(chr(33 + index % 90) for index in range(length))


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_fixed_rejects_empty_input(text: str) -> None:
    with pytest.raises(ChunkGenerationError):
        chunk_document(_docx(text), "fixed")


@pytest.mark.parametrize("length", [1, 499, 500, 1_500, 1_999, 2_000])
def test_fixed_text_at_or_below_maximum_produces_one_chunk(length: int) -> None:
    text = _pattern(length)

    result = chunk_document(_docx(text), "fixed")

    assert [chunk.content for chunk in result.chunks] == [text]


def test_fixed_text_of_2001_characters_has_one_overlapping_final_window() -> None:
    text = _pattern(2_001)

    result = chunk_document(_docx(text), "fixed")

    assert [len(chunk.content) for chunk in result.chunks] == [2_000, 501]
    assert result.chunks[0].content == text[:2_000]
    assert result.chunks[1].content == text[1_500:]


def test_fixed_multiple_windows_use_exact_step_and_overlap() -> None:
    text = _pattern(4_501)

    result = chunk_document(_docx(text), "fixed")
    contents = [chunk.content for chunk in result.chunks]

    assert MAX_CHUNK_SIZE == 2_000
    assert CHUNK_OVERLAP == 500
    assert CHUNK_STEP == 1_500
    assert contents == [text[0:2_000], text[1_500:3_500], text[3_000:4_501]]
    assert contents[0][-500:] == contents[1][:500]
    assert contents[1][-500:] == contents[2][:500]


def test_fixed_exact_step_multiple_does_not_emit_redundant_final_window() -> None:
    text = _pattern(4_500)

    result = chunk_document(_docx(text), "fixed")

    assert [chunk.content for chunk in result.chunks] == [
        text[0:2_000],
        text[1_500:3_500],
        text[3_000:4_500],
    ]
    assert result.chunks[-1].content.endswith(text[-100:])


def test_fixed_preserves_short_final_content_after_trimming() -> None:
    text = "a" * 2_000 + " " * 500 + "end"

    result = chunk_document(_docx(text), "fixed")

    assert result.chunks[-1].content.endswith("end")
    assert result.chunks[-1].content == "a" * 500 + " " * 500 + "end"


def test_fixed_does_not_emit_whitespace_only_windows() -> None:
    text = " " * 2_000 + "meaningful"

    result = chunk_document(_docx(text), "fixed")

    assert [chunk.content for chunk in result.chunks] == ["meaningful"]


def test_fixed_pdf_pages_are_split_independently_with_page_numbers() -> None:
    first_page = _pattern(2_001)
    second_page = "Second page"

    result = chunk_document(_pdf((first_page, 1), (second_page, 3)), "fixed")

    assert [chunk.page_number for chunk in result.chunks] == [1, 1, 3]
    assert result.chunks[1].content == first_page[1_500:]
    assert result.chunks[2].content == second_page
    assert first_page[-10:] not in result.chunks[2].content


def test_fixed_docx_units_form_one_ordered_stream_with_two_newlines() -> None:
    result = chunk_document(_docx("First unit", "Second | table row", "Third"), "fixed")

    assert len(result.chunks) == 1
    assert [chunk.content for chunk in result.chunks] == [
        "First unit\n\nSecond | table row\n\nThird"
    ]
    assert result.source_file == "document.docx"
    assert result.source_type == "DOCX"
    assert result.chunking_strategy.value == "fixed"
    assert result.chunks[0].page_number is None


def test_fixed_indexes_are_continuous_across_pdf_pages() -> None:
    result = chunk_document(_pdf((_pattern(2_001), 2), (_pattern(2_001), 5)), "fixed")

    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2, 3]


def test_fixed_repeated_execution_is_stable_and_covers_source_ends() -> None:
    text = _pattern(6_137)
    document = _docx(text)

    first = chunk_document(document, "fixed")
    second = chunk_document(document, "fixed")

    assert first == second
    assert first.chunks[0].content.startswith(text[:100])
    assert first.chunks[-1].content.endswith(text[-100:])
    for previous, current in zip(first.chunks, first.chunks[1:], strict=False):
        assert previous.content[-CHUNK_OVERLAP:] == current.content[:CHUNK_OVERLAP]
