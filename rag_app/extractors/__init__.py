"""Validation and dispatch boundary for structured document extraction."""

from pathlib import Path

from rag_app.documents import ExtractedDocument, ExtractedTextUnit, SourceType
from rag_app.exceptions import (
    EmptyDocumentError,
    InvalidDocumentPathError,
    UnsupportedDocumentTypeError,
)
from rag_app.processing.cleaning import clean_text

_SOURCE_TYPES: dict[str, SourceType] = {".pdf": "PDF", ".docx": "DOCX"}


def validate_document_path(path: str | Path | None) -> tuple[Path, SourceType]:
    """Validate a document path and return its canonical source type."""
    if path is None or isinstance(path, str) and not path.strip():
        raise InvalidDocumentPathError("A document file path must be supplied")

    candidate = Path(path)
    filename = candidate.name or "the supplied path"

    try:
        if not candidate.exists():
            raise InvalidDocumentPathError(f"Document file '{filename}' does not exist")
        if not candidate.is_file():
            raise InvalidDocumentPathError(
                f"Document path '{filename}' is not a regular file"
            )
        with candidate.open("rb"):
            pass
    except InvalidDocumentPathError:
        raise
    except OSError as error:
        raise InvalidDocumentPathError(
            f"Document file '{filename}' cannot be read"
        ) from error

    extension = candidate.suffix.lower()
    if not extension:
        raise UnsupportedDocumentTypeError(
            f"Document file '{filename}' has no supported file extension"
        )
    if extension not in _SOURCE_TYPES:
        raise UnsupportedDocumentTypeError(
            f"Document file '{filename}' has an unsupported type; use PDF or DOCX"
        )

    return candidate, _SOURCE_TYPES[extension]


def extract_document(path: str | Path | None) -> ExtractedDocument:
    """Validate, extract, and clean a supported document."""
    source_path, source_type = validate_document_path(path)

    if source_type == "PDF":
        from rag_app.extractors.pdf import extract_pdf

        raw_units = extract_pdf(source_path)
    else:
        from rag_app.extractors.docx import extract_docx

        raw_units = extract_docx(source_path)

    meaningful_units = (
        (cleaned_text, page_number)
        for text, page_number in raw_units
        if (cleaned_text := clean_text(text))
    )
    units = tuple(
        ExtractedTextUnit(
            text=cleaned_text,
            position=position,
            page_number=page_number,
        )
        for position, (cleaned_text, page_number) in enumerate(meaningful_units)
    )

    if not units:
        message = f"No extractable text was found in '{source_path.name}'"
        if source_type == "PDF":
            message += "; OCR is not supported for scanned PDFs"
        raise EmptyDocumentError(message)

    return ExtractedDocument(
        source_file=source_path.name,
        source_type=source_type,
        units=units,
    )


__all__ = [
    "ExtractedDocument",
    "ExtractedTextUnit",
    "extract_document",
    "validate_document_path",
]
