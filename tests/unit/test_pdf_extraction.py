"""Tests for page-aware PDF extraction."""

from pathlib import Path

import pytest

from rag_app.exceptions import DocumentExtractionError, EmptyDocumentError
from rag_app.extractors import extract_document
from rag_app.extractors.pdf import _remove_repeated_margins

PDF_FIXTURES = Path(__file__).parents[1] / "fixtures" / "pdf"


def test_pdf_extraction_preserves_page_order_and_physical_page_numbers() -> None:
    document = extract_document(PDF_FIXTURES / "multi-page.pdf")

    assert [unit.text for unit in document.units] == [
        "First page text",
        "Third page text",
    ]
    assert [unit.page_number for unit in document.units] == [1, 3]
    assert [unit.position for unit in document.units] == [0, 1]


def test_pdf_metadata_contains_only_original_filename() -> None:
    document = extract_document(PDF_FIXTURES / "multi-page.pdf")

    assert document.source_file == "multi-page.pdf"
    assert document.source_type == "PDF"
    assert str(PDF_FIXTURES) not in repr(document)


@pytest.mark.parametrize("filename", ["empty.pdf", "non-extractable.pdf"])
def test_pdf_without_embedded_text_is_rejected(filename: str) -> None:
    with pytest.raises(EmptyDocumentError) as caught:
        extract_document(PDF_FIXTURES / filename)

    assert filename not in str(caught.value)
    assert "does not contain extractable text" in str(caught.value)
    assert "OCR is not supported" in str(caught.value)


def test_malformed_pdf_is_reported_without_parser_details() -> None:
    path = PDF_FIXTURES / "malformed.pdf"

    with pytest.raises(DocumentExtractionError) as caught:
        extract_document(path)

    assert "malformed.pdf" in str(caught.value)
    assert str(PDF_FIXTURES) not in str(caught.value)
    assert caught.value.__cause__ is not None


def test_encrypted_pdf_is_reported_as_unsupported() -> None:
    with pytest.raises(DocumentExtractionError, match="password-protected"):
        extract_document(PDF_FIXTURES / "encrypted.pdf")


def test_unsupported_pdf_parser_feature_is_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_reader(*args: object, **kwargs: object) -> object:
        raise NotImplementedError("unsupported synthetic filter details")

    monkeypatch.setattr("rag_app.extractors.pdf.PdfReader", unsupported_reader)

    with pytest.raises(DocumentExtractionError) as caught:
        extract_document(PDF_FIXTURES / "multi-page.pdf")

    assert "unsupported synthetic filter details" not in str(caught.value)
    assert isinstance(caught.value.__cause__, NotImplementedError)


def test_repeated_pdf_headers_and_numbered_footers_are_removed() -> None:
    document = extract_document(PDF_FIXTURES / "repeated-margins.pdf")

    assert all(not unit.text.startswith("Shared Header\n") for unit in document.units)
    assert all("Page 1 of 3" not in unit.text for unit in document.units)
    assert all("Page 2 of 3" not in unit.text for unit in document.units)
    assert all("Page 3 of 3" not in unit.text for unit in document.units)


def test_header_text_in_the_middle_of_a_page_is_preserved() -> None:
    document = extract_document(PDF_FIXTURES / "repeated-margins.pdf")

    assert "Shared Header" in document.units[0].text.splitlines()[1:]


def test_one_page_pdf_does_not_run_recurring_margin_removal() -> None:
    document = extract_document(PDF_FIXTURES / "single-page.pdf")

    assert document.units[0].text == (
        "Only Page Heading\nImportant body\nPage 1"
    )


def test_margin_candidate_must_appear_on_at_least_half_of_all_pages() -> None:
    pages = [
        "Header\nbody 1\nfooter 1",
        "Other\nbody 2\nfooter 2",
        "Other two\nbody 3\nend",
        "Other three\nbody 4\nlast",
        "Other four\nbody 5\nfinish",
    ]

    assert _remove_repeated_margins(pages) == pages


def test_changing_numeric_body_values_are_not_treated_as_page_numbers() -> None:
    pages = [
        "Report\nbody 1\n100",
        "Report\nbody 2\n200",
        "Report\nbody 3\n300",
    ]

    cleaned = _remove_repeated_margins(pages)

    assert [page.splitlines()[-1] for page in cleaned] == ["100", "200", "300"]


def test_fixed_numeric_footer_is_removed_consistently() -> None:
    pages = [
        "First heading\nbody 1\nSection 2",
        "Second heading\nbody 2\nSection 2",
        "Third heading\nbody 3\nSection 2",
    ]

    cleaned = _remove_repeated_margins(pages)

    assert all("Section 2" not in page for page in cleaned)


def test_duplicate_footer_lines_on_one_page_do_not_count_as_two_pages() -> None:
    pages = [
        "First heading\nbody 1\nmore body\nPage 1\nPage 1",
        "Second heading\nbody 2\nmore body\nother\nending",
        "Third heading\nbody 3\nmore body\nanother\nfinish",
    ]

    cleaned = _remove_repeated_margins(pages)

    assert cleaned[0].splitlines()[-2:] == ["Page 1", "Page 1"]
