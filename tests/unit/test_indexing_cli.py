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
        ["index", "--file", "missing.pdf", "--strategy", "fixed"],
        ["index", "--file", __file__, "--strategy", "invalid"],
    ],
)
def test_usage_failures_keep_exit_two(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli_module.app, arguments)

    assert result.exit_code == 2
