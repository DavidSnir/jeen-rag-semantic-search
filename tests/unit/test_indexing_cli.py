"""CLI tests for indexing summaries and exit semantics."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import rag_app.cli as cli_module
from rag_app.documents import ChunkingStrategy, IndexingResult, IndexingStatus
from rag_app.exceptions import PersistenceError


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (IndexingStatus.indexed, "Indexed document"),
        (IndexingStatus.replaced, "Replaced existing document"),
        (IndexingStatus.skipped, "Skipped unchanged document"),
    ],
)
def test_index_success_summary_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: IndexingStatus,
    label: str,
) -> None:
    path = tmp_path / "Report.PDF"
    path.write_bytes(b"test")
    received: list[tuple[Path, str]] = []

    def index(file: Path, strategy: str) -> IndexingResult:
        received.append((file, strategy))
        return IndexingResult(
            status=status,
            source_file="Report.PDF",
            chunking_strategy=ChunkingStrategy.fixed,
            chunk_count=3,
            elapsed_seconds=0.126,
        )

    monkeypatch.setattr(cli_module, "index_document", index)
    result = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(path), "--strategy", "fixed"],
    )

    assert result.exit_code == 0
    assert received == [(path, "fixed")]
    assert result.output == (
        f"{label}: Report.PDF | strategy=fixed | chunks=3 | elapsed=0.13s\n"
    )
    assert str(tmp_path) not in result.output


def test_expected_index_failure_is_safe_and_uses_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "document.docx"
    path.write_bytes(b"test")

    def fail(file: Path, strategy: str) -> None:
        raise PersistenceError("Document persistence failed.")

    monkeypatch.setattr(cli_module, "index_document", fail)
    result = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(path), "--strategy", "sentence"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: Document persistence failed.\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["index", "--strategy", "fixed"],
        ["index", "--file", __file__, "--strategy", "invalid"],
    ],
)
def test_usage_failures_keep_exit_two(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli_module.app, arguments)

    assert result.exit_code == 2


def test_missing_file_is_application_failure_with_exit_one(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    result = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(missing), "--strategy", "fixed"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: Document file was not found: missing.pdf\n"
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_unsupported_file_is_application_failure_with_exit_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("sensitive content", encoding="utf-8")

    result = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(path), "--strategy", "fixed"],
    )

    assert result.exit_code == 1
    assert result.output == (
        "Error: Unsupported document type. Use a PDF or DOCX file.\n"
    )
    assert "sensitive content" not in result.output
    assert "Traceback" not in result.output


def test_strategy_is_normalized_and_invalid_value_is_not_reflected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"test")
    received: list[str] = []

    def index(file: Path, strategy: str) -> IndexingResult:
        received.append(strategy)
        return IndexingResult(
            IndexingStatus.indexed,
            "document.pdf",
            ChunkingStrategy.fixed,
            1,
            0.01,
        )

    monkeypatch.setattr(cli_module, "index_document", index)
    normalized = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(path), "--strategy", " FIXED "],
    )
    secret = "stage7-secret-strategy"
    rejected = CliRunner().invoke(
        cli_module.app,
        ["index", "--file", str(path), "--strategy", secret],
    )

    assert normalized.exit_code == 0
    assert received == ["fixed"]
    assert rejected.exit_code == 2
    assert secret not in rejected.output
    assert "fixed, sentence, paragraph" in rejected.output
