"""Tests for shared document path validation and dispatch metadata."""

from pathlib import Path
import shutil

import pytest

from rag_app.exceptions import InvalidDocumentPathError, UnsupportedDocumentTypeError
from rag_app.extractors import extract_document, validate_document_path

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize(
    ("relative_path", "source_type"),
    [
        ("pdf/multi-page.pdf", "PDF"),
        ("docx/ordered-content.docx", "DOCX"),
    ],
)
def test_validate_document_path_accepts_supported_files(
    relative_path: str, source_type: str
) -> None:
    path = FIXTURES / relative_path

    assert validate_document_path(path) == (path, source_type)


@pytest.mark.parametrize("missing_path", [None, "", "   "])
def test_validate_document_path_rejects_unsupplied_path(
    missing_path: str | None,
) -> None:
    with pytest.raises(InvalidDocumentPathError, match="must be supplied"):
        validate_document_path(missing_path)


def test_validate_document_path_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.pdf"

    with pytest.raises(InvalidDocumentPathError, match="missing.pdf.*does not exist"):
        validate_document_path(path)


def test_validate_document_path_rejects_directory(tmp_path: Path) -> None:
    directory = tmp_path / "document.pdf"
    directory.mkdir()

    with pytest.raises(InvalidDocumentPathError, match="not a regular file"):
        validate_document_path(directory)


def test_validate_document_path_rejects_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.pdf"
    path.write_bytes(b"content")
    original_open = Path.open

    def deny_target(candidate: Path, *args: object, **kwargs: object) -> object:
        if candidate == path:
            raise PermissionError("synthetic permission failure")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_target)

    with pytest.raises(InvalidDocumentPathError, match="cannot be read") as caught:
        validate_document_path(path)

    assert isinstance(caught.value.__cause__, PermissionError)
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("filename", ["notes.txt", "legacy.doc"])
def test_validate_document_path_rejects_unsupported_extension(
    tmp_path: Path, filename: str
) -> None:
    path = tmp_path / filename
    path.write_text("synthetic", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="unsupported type"):
        validate_document_path(path)


def test_validate_document_path_rejects_missing_extension(tmp_path: Path) -> None:
    path = tmp_path / "document"
    path.write_text("synthetic", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="no supported file extension"):
        validate_document_path(path)


@pytest.mark.parametrize(
    ("filename", "source_fixture", "source_type"),
    [
        ("UPPER.PDF", "pdf/multi-page.pdf", "PDF"),
        ("Mixed.PdF", "pdf/multi-page.pdf", "PDF"),
        ("UPPER.DOCX", "docx/ordered-content.docx", "DOCX"),
        ("Mixed.DoCx", "docx/ordered-content.docx", "DOCX"),
    ],
)
def test_dispatch_accepts_case_insensitive_extensions(
    tmp_path: Path, filename: str, source_fixture: str, source_type: str
) -> None:
    path = tmp_path / filename
    shutil.copyfile(FIXTURES / source_fixture, path)

    document = extract_document(path)

    assert document.source_file == filename
    assert document.source_type == source_type
    assert str(tmp_path) not in repr(document)
