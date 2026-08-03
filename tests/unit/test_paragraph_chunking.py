"""Tests for paragraph-preserving semantic chunking."""

import pytest

from rag_app.documents import ExtractedDocument, ExtractedTextUnit
from rag_app.exceptions import ChunkGenerationError
from rag_app.processing.chunking import chunk_document


def _docx(*texts: str) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="paragraphs.docx",
        source_type="DOCX",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=None)
            for index, text in enumerate(texts)
        ),
    )


def _pdf(*pages: tuple[str, int]) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="paragraphs.pdf",
        source_type="PDF",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=page_number)
            for index, (text, page_number) in enumerate(pages)
        ),
    )


def test_paragraph_single_paragraph_is_retained() -> None:
    result = chunk_document(_docx("One meaningful paragraph."), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [
        "One meaningful paragraph."
    ]


def test_paragraph_returns_each_short_unit_as_an_independent_chunk() -> None:
    result = chunk_document(_docx("First paragraph.", "Second paragraph."), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert len(result.chunks) == 2


def test_paragraph_never_combines_short_headings_or_list_items() -> None:
    result = chunk_document(
        _docx("Heading", "Short introduction.", "- First item", "- Second item"),
        "paragraph",
    )

    assert [chunk.content for chunk in result.chunks] == [
        "Heading",
        "Short introduction.",
        "- First item",
        "- Second item",
    ]


def test_paragraph_ignores_empty_blocks_without_discarding_short_blocks() -> None:
    result = chunk_document(_docx("Title", "  ", "x", "\n"), "paragraph")

    assert [chunk.content for chunk in result.chunks] == ["Title", "x"]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]


def test_paragraph_rejects_empty_input() -> None:
    with pytest.raises(ChunkGenerationError, match="meaningful"):
        chunk_document(_docx("  ", "\n\n", "\t"), "paragraph")


def test_paragraph_pdf_uses_blank_lines_not_visual_lines() -> None:
    page = "first visual line\nsecond visual line\n\nNext paragraph."

    result = chunk_document(_pdf((page, 4)), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [
        "first visual line\nsecond visual line",
        "Next paragraph.",
    ]
    assert [chunk.page_number for chunk in result.chunks] == [4, 4]


def test_paragraph_pdf_ignores_repeated_empty_paragraph_blocks() -> None:
    page = "First.\n\n\n\nSecond."

    result = chunk_document(_pdf((page, 1)), "paragraph")

    assert [chunk.content for chunk in result.chunks] == ["First.", "Second."]


def test_paragraph_docx_units_and_table_rows_stay_ordered() -> None:
    result = chunk_document(
        _docx("Introduction", "A | B | C", "D | E | F", "Conclusion"),
        "paragraph",
    )

    assert [chunk.content for chunk in result.chunks] == [
        "Introduction",
        "A | B | C",
        "D | E | F",
        "Conclusion",
    ]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2, 3]
    assert all(chunk.page_number is None for chunk in result.chunks)


def test_paragraph_preserves_multiple_sentences_inside_normal_paragraph() -> None:
    paragraph = "First sentence. Second sentence! Third sentence?"

    result = chunk_document(_docx(f"  {paragraph}  "), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [paragraph]


def test_paragraph_exactly_2000_characters_is_not_split() -> None:
    paragraph = "p" * 2_000

    result = chunk_document(_docx(paragraph), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [paragraph]


def test_paragraph_oversized_paragraph_splits_at_sentence_boundaries() -> None:
    first = "alpha " + "a" * 700 + "."
    second = "beta " + "b" * 700 + "."
    third = "gamma " + "c" * 700 + "."

    result = chunk_document(_docx(f"{first} {second} {third}"), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [first, second, third]
    assert sum(chunk.content.count("alpha") for chunk in result.chunks) == 1
    assert sum(chunk.content.count("beta") for chunk in result.chunks) == 1
    assert sum(chunk.content.count("gamma") for chunk in result.chunks) == 1


def test_paragraph_oversized_sentence_uses_fixed_overlap() -> None:
    sentence = "".join(chr(65 + index % 26) for index in range(2_500))

    result = chunk_document(_docx(sentence), "paragraph")

    assert [len(chunk.content) for chunk in result.chunks] == [2_000, 1_000]
    assert result.chunks[0].content[-500:] == result.chunks[1].content[:500]
    assert result.chunks[1].content.endswith(sentence[-100:])


def test_paragraph_fallback_is_not_recombined_with_adjacent_paragraphs() -> None:
    oversized = "z" * 2_000 + "."

    result = chunk_document(
        _docx("Before.", oversized, "After."), "paragraph"
    )

    assert result.chunks[0].content == "Before."
    assert [len(chunk.content) for chunk in result.chunks[1:3]] == [2_000, 501]
    assert result.chunks[3].content == "After."


def test_paragraph_with_oversized_sentence_preserves_surrounding_sentences() -> None:
    oversized = "z" * 2_000 + "."

    result = chunk_document(
        _docx(f"Before sentence. {oversized} After sentence."), "paragraph"
    )

    assert result.chunks[0].content == "Before sentence."
    assert result.chunks[1].content == oversized[:2_000]
    assert result.chunks[2].content == oversized[1_500:]
    assert result.chunks[3].content == "After sentence."


def test_paragraph_keeps_pdf_pages_separate() -> None:
    result = chunk_document(
        _pdf(("Page one paragraph.", 2), ("Page two paragraph.", 5)),
        "paragraph",
    )

    assert [chunk.content for chunk in result.chunks] == [
        "Page one paragraph.",
        "Page two paragraph.",
    ]
    assert [chunk.page_number for chunk in result.chunks] == [2, 5]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]


def test_paragraph_repeated_execution_is_stable_and_preserves_content() -> None:
    document = _docx("Symbols: π, %, #.", "Numbers: 1, 2, 3.", "Unicode: café.")

    first = chunk_document(document, "paragraph")
    second = chunk_document(document, "paragraph")

    assert first == second
    assert [chunk.content for chunk in first.chunks] == [
        "Symbols: π, %, #.",
        "Numbers: 1, 2, 3.",
        "Unicode: café.",
    ]
