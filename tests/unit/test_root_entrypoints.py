"""Behavioral parity tests for the assignment-compatible root wrappers."""

import sys
from pathlib import Path

import pytest

import rag_app.cli as cli_module
from rag_app.documents import (
    ChunkingStrategy,
    IndexingResult,
    IndexingStatus,
    SearchResponse,
)
from rag_app.exceptions import GeminiRequestError


def _run_wrapper(
    wrapper: object,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> int:
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as raised:
        wrapper()  # type: ignore[operator]
    return int(raised.value.code)


def test_index_wrapper_uses_shared_success_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"test")
    received: list[tuple[Path, str]] = []

    def index(file: Path, strategy: str) -> IndexingResult:
        received.append((file, strategy))
        return IndexingResult(
            IndexingStatus.indexed,
            "report.pdf",
            ChunkingStrategy.fixed,
            2,
            0.1,
        )

    monkeypatch.setattr(cli_module, "index_document", index)
    code = _run_wrapper(
        cli_module.run_index_wrapper,
        monkeypatch,
        ["index_documents.py", "--file", str(path), "--strategy", "fixed"],
    )

    captured = capsys.readouterr()
    assert code == 0
    assert received == [(path, "fixed")]
    assert captured.out == (
        "Indexed document: report.pdf | strategy=fixed | chunks=2 | elapsed=0.10s\n"
    )
    assert captured.err == ""


def test_index_wrapper_missing_file_is_safe_application_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run_wrapper(
        cli_module.run_index_wrapper,
        monkeypatch,
        [
            "index_documents.py",
            "--file",
            str(tmp_path / "missing.pdf"),
            "--strategy",
            "fixed",
        ],
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err == "Error: Document file was not found: missing.pdf\n"
    assert str(tmp_path) not in captured.err
    assert "Traceback" not in captured.err


def test_search_wrapper_empty_result_is_successful(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "search_documents",
        lambda query, strategy, top_k: SearchResponse(
            query, ChunkingStrategy.paragraph, top_k, ()
        ),
    )
    code = _run_wrapper(
        cli_module.run_search_wrapper,
        monkeypatch,
        ["search.py", "--query", "query", "--strategy", "paragraph"],
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "No indexed results found for strategy=paragraph.\n"
    assert captured.err == ""


def test_search_wrapper_expected_failure_is_safe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "stage7-provider-secret"

    def fail(query: str, strategy: str, top_k: int) -> None:
        raise GeminiRequestError("Gemini quota or rate limit was exceeded.") from (
            RuntimeError(secret)
        )

    monkeypatch.setattr(cli_module, "search_documents", fail)
    code = _run_wrapper(
        cli_module.run_search_wrapper,
        monkeypatch,
        ["search.py", "--query", "query", "--strategy", "fixed"],
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err == "Error: Gemini quota or rate limit was exceeded.\n"
    assert secret not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("wrapper", "arguments"),
    [
        (cli_module.run_index_wrapper, ["index_documents.py", "--strategy", "fixed"]),
        (cli_module.run_search_wrapper, ["search.py", "--strategy", "fixed"]),
    ],
)
def test_root_wrapper_usage_failures_exit_two(
    wrapper: object,
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run_wrapper(wrapper, monkeypatch, arguments)

    captured = capsys.readouterr()
    assert code == 2
    assert "Missing option" in captured.err
    assert "Traceback" not in captured.err
