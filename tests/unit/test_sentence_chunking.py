"""Tests for spaCy Sentencizer-based document chunking."""

import pytest

from rag_app.documents import ExtractedDocument, ExtractedTextUnit
from rag_app.exceptions import ChunkGenerationError
from rag_app.processing import chunking
from rag_app.processing.chunking import chunk_document


def _docx(*texts: str) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="sentences.docx",
        source_type="DOCX",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=None)
            for index, text in enumerate(texts)
        ),
    )


def _pdf(*pages: tuple[str, int]) -> ExtractedDocument:
    return ExtractedDocument(
        source_file="sentences.pdf",
        source_type="PDF",
        units=tuple(
            ExtractedTextUnit(text=text, position=index, page_number=page_number)
            for index, (text, page_number) in enumerate(pages)
        ),
    )


def test_sentence_single_sentence_is_retained() -> None:
    result = chunk_document(_docx("One concise sentence."), "sentence")

    assert [chunk.content for chunk in result.chunks] == ["One concise sentence."]


def test_sentence_returns_each_short_sentence_as_an_independent_chunk() -> None:
    result = chunk_document(
        _docx("First sentence. Second sentence! Third?"), "sentence"
    )

    assert [chunk.content for chunk in result.chunks] == [
        "First sentence.",
        "Second sentence!",
        "Third?",
    ]
    assert len(result.chunks) == 3


def test_sentence_trims_surrounding_whitespace_without_merging() -> None:
    result = chunk_document(
        _docx("  First sentence.   Second sentence!\n\nThird sentence?  "),
        "sentence",
    )

    assert [chunk.content for chunk in result.chunks] == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]


def test_sentence_normal_chunks_preserve_boundaries_without_overlap() -> None:
    first = "alpha " + "a" * 1_100 + "."
    second = "beta " + "b" * 1_100 + "."

    result = chunk_document(_docx(f"{first} {second}"), "sentence")

    assert [chunk.content for chunk in result.chunks] == [first, second]
    assert sum(chunk.content.count("alpha") for chunk in result.chunks) == 1
    assert sum(chunk.content.count("beta") for chunk in result.chunks) == 1


def test_sentence_preserves_repeated_internal_whitespace() -> None:
    result = chunk_document(_docx("First   sentence. Second sentence."), "sentence")

    assert [chunk.content for chunk in result.chunks] == [
        "First   sentence.",
        "Second sentence.",
    ]


def test_sentence_recognizes_common_punctuation_boundaries() -> None:
    result = chunk_document(_docx("Period. Exclamation! Question?"), "sentence")

    assert [chunk.content for chunk in result.chunks] == [
        "Period.",
        "Exclamation!",
        "Question?",
    ]


def test_sentence_ignores_empty_units_and_preserves_order() -> None:
    result = chunk_document(
        _docx("   ", "First.", "\n\n", "Second.", "Third."), "sentence"
    )

    assert [chunk.content for chunk in result.chunks] == [
        "First.",
        "Second.",
        "Third.",
    ]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]


def test_sentence_rejects_empty_input() -> None:
    with pytest.raises(ChunkGenerationError, match="meaningful"):
        chunk_document(_docx("  \n\t  "), "sentence")


def test_sentence_exactly_2000_characters_is_not_split() -> None:
    sentence = "x" * 1_999 + "."

    result = chunk_document(_docx(sentence), "sentence")

    assert [chunk.content for chunk in result.chunks] == [sentence]


def test_sentence_oversized_sentence_uses_fixed_overlap() -> None:
    sentence = "".join(chr(65 + index % 26) for index in range(2_500))

    result = chunk_document(_docx(sentence), "sentence")

    assert [len(chunk.content) for chunk in result.chunks] == [2_000, 1_000]
    assert result.chunks[0].content[-500:] == result.chunks[1].content[:500]
    assert result.chunks[0].content == sentence[:2_000]
    assert result.chunks[1].content == sentence[1_500:]


def test_sentence_preserves_normal_content_around_oversized_sentence() -> None:
    oversized = "z" * 2_000 + "."

    result = chunk_document(_docx(f"Before. {oversized} After."), "sentence")

    assert result.chunks[0].content == "Before."
    assert [len(chunk.content) for chunk in result.chunks[1:3]] == [2_000, 501]
    assert result.chunks[3].content == "After."


def test_sentence_pdf_sentences_preserve_page_numbers_and_order() -> None:
    result = chunk_document(
        _pdf(
            ("First sentence on page one. Last sentence on page one.", 1),
            ("First sentence on page two.", 2),
        ),
        "sentence",
    )

    assert [chunk.content for chunk in result.chunks] == [
        "First sentence on page one.",
        "Last sentence on page one.",
        "First sentence on page two.",
    ]
    assert [chunk.page_number for chunk in result.chunks] == [1, 1, 2]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert result.source_file == "sentences.pdf"
    assert result.source_type == "PDF"
    assert result.chunking_strategy.value == "sentence"


def test_sentence_docx_keeps_adjacent_units_independent_with_null_pages() -> None:
    result = chunk_document(_docx("First unit.", "Second unit."), "sentence")

    assert [chunk.content for chunk in result.chunks] == [
        "First unit.",
        "Second unit.",
    ]
    assert all(chunk.page_number is None for chunk in result.chunks)


def test_sentence_pipeline_is_minimal_and_reused() -> None:
    chunking._get_sentencizer.cache_clear()

    first = chunking._get_sentencizer()
    second = chunking._get_sentencizer()

    assert first is second
    assert first.lang == "en"
    assert first.pipe_names == ["sentencizer"]


def test_fixed_and_normal_paragraph_chunking_do_not_initialize_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> None:
        raise AssertionError("spaCy should remain lazy")

    monkeypatch.setattr(chunking, "_get_sentencizer", fail_if_called)

    assert chunk_document(_docx("Fixed content."), "fixed").chunks
    assert chunk_document(_docx("Normal paragraph."), "paragraph").chunks


def test_sentence_library_failure_is_wrapped_with_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("synthetic spaCy failure")

    def fail() -> None:
        raise failure

    monkeypatch.setattr(chunking, "_get_sentencizer", fail)

    with pytest.raises(ChunkGenerationError, match="sentence segmenter") as caught:
        chunk_document(_docx("A sentence."), "sentence")

    assert caught.value.__cause__ is failure


def test_unexpected_sentence_programming_error_is_not_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise TypeError("programming bug")

    monkeypatch.setattr(chunking, "_get_sentencizer", fail)

    with pytest.raises(TypeError, match="programming bug"):
        chunk_document(_docx("A sentence."), "sentence")


def test_sentence_repeated_execution_is_stable() -> None:
    document = _docx("First. Second. Third.")

    assert chunk_document(document, "sentence") == chunk_document(document, "sentence")
