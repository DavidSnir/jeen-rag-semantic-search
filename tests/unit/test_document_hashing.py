"""Unit tests for exact binary SHA-256 document identity."""

import hashlib
import re
from pathlib import Path

import pytest

from rag_app.exceptions import DocumentHashingError
from rag_app.processing.hashing import calculate_document_hash


@pytest.mark.parametrize(
    "content",
    [b"", b"small document", b"binary\x00content\xff"],
)
def test_known_bytes_produce_standard_sha256(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(content)

    result = calculate_document_hash(path)

    assert result == hashlib.sha256(content).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", result)


def test_file_larger_than_one_read_block_is_hashed_completely(tmp_path: Path) -> None:
    content = b"a" * (1024 * 1024 + 37)
    path = tmp_path / "large.docx"
    path.write_bytes(content)

    assert calculate_document_hash(path) == hashlib.sha256(content).hexdigest()


def test_hashing_is_deterministic_and_changes_with_one_byte(tmp_path: Path) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"version-a")
    first = calculate_document_hash(path)
    second = calculate_document_hash(path)

    path.write_bytes(b"version-b")

    assert first == second
    assert calculate_document_hash(path) != first


def test_filename_does_not_contribute_to_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "Second.DOCX"
    first.write_bytes(b"identical bytes")
    second.write_bytes(b"identical bytes")

    assert calculate_document_hash(first) == calculate_document_hash(second)


def test_read_failure_uses_only_safe_filename(tmp_path: Path) -> None:
    path = tmp_path / "private-parent" / "missing.pdf"

    with pytest.raises(DocumentHashingError) as raised:
        calculate_document_hash(path)

    assert raised.value.__cause__ is not None
    assert "missing.pdf" in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
