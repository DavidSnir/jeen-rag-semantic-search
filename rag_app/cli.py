"""Typer command-line interface for application use cases."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from rag_app.config import DEFAULT_TOP_K
from rag_app.database.repository import (
    check_database_readiness,
    initialize_schema,
)
from rag_app.documents import ChunkingStrategy, IndexingStatus, SearchResponse
from rag_app.exceptions import RagAppError
from rag_app.services.indexing import index_document, reset_index
from rag_app.services.search import search_documents

app = typer.Typer(
    help="Index documents and search indexed content semantically.",
    no_args_is_help=True,
    add_completion=False,
)

Result = TypeVar("Result")


@app.command("database-init")
def database_init() -> None:
    """Create and validate the PostgreSQL/pgvector schema."""
    status = _run(initialize_schema)
    typer.echo(
        "Database schema initialized "
        f"(PostgreSQL {status.postgresql_version}, "
        f"pgvector {status.pgvector_version}, {status.embedding_type})."
    )


@app.command("database-check")
def database_check() -> None:
    """Verify PostgreSQL connectivity and the complete expected schema."""
    status = _run(check_database_readiness)
    typer.echo(
        "Database is ready "
        f"(PostgreSQL {status.postgresql_version}, "
        f"pgvector {status.pgvector_version}, {status.embedding_type})."
    )


def _validate_document_file(path: Path) -> Path:
    if path.suffix.lower() not in {".pdf", ".docx"}:
        raise typer.BadParameter("File must have a .pdf or .docx extension")
    return path


def _validate_query(query: str) -> str:
    canonical_query = query.strip()
    if not canonical_query:
        raise typer.BadParameter("Query must not be empty")
    return canonical_query


@app.command()
def index(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=_validate_document_file,
            help="PDF or DOCX document to index.",
        ),
    ],
    strategy: Annotated[
        ChunkingStrategy,
        typer.Option("--strategy", help="Chunking strategy to use."),
    ],
) -> None:
    """Index one document with the selected chunking strategy."""
    result = _run(index_document, file, strategy.value)
    status_label = {
        IndexingStatus.indexed: "Indexed document",
        IndexingStatus.replaced: "Replaced existing document",
        IndexingStatus.skipped: "Skipped unchanged document",
    }[result.status]
    typer.echo(
        f"{status_label}: {result.source_file} | "
        f"strategy={result.chunking_strategy.value} | "
        f"chunks={result.chunk_count} | elapsed={result.elapsed_seconds:.2f}s"
    )


@app.command()
def search(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            callback=_validate_query,
            help="Question or text to search for.",
        ),
    ],
    strategy: Annotated[
        ChunkingStrategy,
        typer.Option("--strategy", help="Chunking strategy to search."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Maximum number of results."),
    ] = DEFAULT_TOP_K,
) -> None:
    """Search indexed content using semantic similarity."""
    response = _run(search_documents, query, strategy.value, top_k)
    _display_search_response(response)


def _display_search_response(response: SearchResponse) -> None:
    if not response.matches:
        typer.echo(
            "No indexed results found for "
            f"strategy={response.chunking_strategy.value}."
        )
        return

    for match in response.matches:
        page = str(match.page_number) if match.page_number is not None else "n/a"
        typer.echo(f"Result {match.rank} | score={match.score:.4f}")
        typer.echo(
            f"source={match.source_file} | type={match.source_type} | "
            f"strategy={match.chunking_strategy.value} | "
            f"chunk={match.chunk_index} | page={page}"
        )
        typer.echo(match.content)
        if match.rank != len(response.matches):
            typer.echo("-" * 72)


@app.command()
def reset(
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm removal of all indexed content."),
    ] = False,
) -> None:
    """Remove all indexed content after explicit confirmation."""
    if not yes:
        raise typer.BadParameter("Pass --yes to confirm reset", param_hint="--yes")
    _run(reset_index)


def _run(operation: Callable[..., Result], *args: object) -> Result:
    """Present expected application failures without a stack trace."""
    try:
        return operation(*args)
    except RagAppError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def main(args: Sequence[str] | None = None, prog_name: str | None = None) -> None:
    """Run the unified CLI, optionally with arguments supplied by a wrapper."""
    app(args=list(args) if args is not None else None, prog_name=prog_name)


def run_index_wrapper() -> None:
    """Run the shared index command from the assignment-compatible script."""
    typer.run(index)


def run_search_wrapper() -> None:
    """Run the shared search command from the assignment-compatible script."""
    typer.run(search)


if __name__ == "__main__":
    main()
