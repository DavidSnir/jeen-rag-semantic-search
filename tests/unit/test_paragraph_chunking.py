"""Tests for paragraph-preserving semantic chunking."""

from rag_app.documents import ExtractedDocument, ExtractedTextUnit
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


def test_paragraph_groups_units_with_two_newlines() -> None:
    result = chunk_document(_docx("First paragraph.", "Second paragraph."), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [
        "First paragraph.\n\nSecond paragraph."
    ]


def test_paragraph_separator_is_included_in_maximum_length() -> None:
    first = "a" * 1_000
    fitting_second = "b" * 998
    overflowing_second = "c" * 999

    fitting = chunk_document(_docx(first, fitting_second), "paragraph")
    overflowing = chunk_document(_docx(first, overflowing_second), "paragraph")

    assert len(fitting.chunks) == 1
    assert len(fitting.chunks[0].content) == 2_000
    assert [chunk.content for chunk in overflowing.chunks] == [
        first,
        overflowing_second,
    ]


def test_paragraph_ignores_empty_blocks_without_discarding_short_blocks() -> None:
    result = chunk_document(_docx("Title", "  ", "x", "\n"), "paragraph")

    assert [chunk.content for chunk in result.chunks] == ["Title\n\nx"]


def test_paragraph_pdf_uses_blank_lines_not_visual_lines() -> None:
    page = "first visual line\nsecond visual line\n\nNext paragraph."

    result = chunk_document(_pdf((page, 4)), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [page]
    assert result.chunks[0].page_number == 4


def test_paragraph_pdf_ignores_repeated_empty_paragraph_blocks() -> None:
    page = "First.\n\n\n\nSecond."

    result = chunk_document(_pdf((page, 1)), "paragraph")

    assert [chunk.content for chunk in result.chunks] == ["First.\n\nSecond."]


def test_paragraph_docx_units_and_table_rows_stay_ordered() -> None:
    result = chunk_document(
        _docx("Introduction", "A | B | C", "D | E | F", "Conclusion"),
        "paragraph",
    )

    content = result.chunks[0].content
    assert content == "Introduction\n\nA | B | C\n\nD | E | F\n\nConclusion"
    assert content.index("A | B | C") < content.index("D | E | F")
    assert result.chunks[0].page_number is None


def test_paragraph_exactly_2000_characters_is_not_split() -> None:
    paragraph = "p" * 2_000

    result = chunk_document(_docx(paragraph), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [paragraph]


def test_paragraph_oversized_paragraph_splits_at_sentence_boundaries() -> None:
    first = "alpha " + "a" * 1_100 + "."
    second = "beta " + "b" * 1_100 + "."

    result = chunk_document(_docx(f"{first} {second}"), "paragraph")

    assert [chunk.content for chunk in result.chunks] == [first, second]
    assert sum(chunk.content.count("alpha") for chunk in result.chunks) == 1
    assert sum(chunk.content.count("beta") for chunk in result.chunks) == 1


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


def test_paragraph_never_groups_across_pdf_pages() -> None:
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
    assert first.chunks[0].content == (
        "Symbols: π, %, #.\n\nNumbers: 1, 2, 3.\n\nUnicode: café."
    )
