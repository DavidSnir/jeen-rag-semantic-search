# Jeen RAG Semantic Search

A Python application for indexing PDF and DOCX documents and retrieving relevant
content with Gemini embeddings and PostgreSQL/pgvector semantic search.

## Status

Stage 1 provides a reproducible PostgreSQL 17 and pgvector 0.8.2 environment,
an idempotent database schema, and application-level readiness verification.
Document extraction, chunking, Gemini embeddings, production persistence,
indexing, and semantic search are not implemented yet.

## Runtime

The supported runtime is Python `>=3.12,<3.13`. The repository's
`.python-version` selects Python 3.12.

Install an available Python 3.12 release if needed. For example, with pyenv:

```bash
pyenv install 3.12
```

Verify that Python 3.12 is selected before installing dependencies:

```bash
python --version
python -c "import sys; assert sys.version_info[:2] == (3, 12)"
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## PostgreSQL Prerequisites

Install Docker Desktop or another Docker environment with Docker Compose support.
The Compose configuration uses the pinned image
`pgvector/pgvector:0.8.2-pg17-bookworm`; no local PostgreSQL installation is
required.

## Configuration

Create a local environment file and provide real values at runtime:

```bash
cp .env.example .env
```

Set `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`, and
`POSTGRES_URL` in `.env`. The individual values are consumed by Docker Compose,
while the Python application uses only `POSTGRES_URL`.

Keep these values consistent. For example, changing `POSTGRES_PASSWORD` also
requires changing the password component of `POSTGRES_URL`. URL-encode any
URI-special characters used in that component. Replace the example password
before starting PostgreSQL. Never commit `.env` or real credentials.

`GEMINI_API_KEY` may remain empty for Stage 1 database setup and tests. No Gemini
request is made by this stage.

## Database Setup

Start PostgreSQL and wait until its configured health check passes:

```bash
docker compose up --detach --wait postgres
```

Inspect health or startup logs when needed:

```bash
docker compose ps postgres
docker inspect --format='{{.State.Health.Status}}' jeen-rag-postgres
docker compose logs postgres
```

On the first creation of the named volume, PostgreSQL executes
`rag_app/database/schema.sql` automatically. This file is the only canonical
schema definition. Docker entrypoint initialization does not rerun for an
existing volume, so the application also provides an idempotent initializer:

```bash
python -m rag_app.cli database-init
```

Verify connectivity, pgvector registration, the complete table definition,
constraints, relational indexes, and HNSW cosine index independently:

```bash
python -m rag_app.cli database-check
```

Stop the container without deleting indexed data:

```bash
docker compose down
```

For a complete development reset, remove the container and named volume:

```bash
docker compose down --volumes
```

Warning: removing `jeen-rag-postgres-data` permanently deletes all locally
indexed data stored in that volume.

## Tests

Unit tests do not require a running database:

```bash
python -m pytest tests/unit
```

Database integration tests require the healthy Compose service and a matching
`POSTGRES_URL` in `.env`:

```bash
docker compose up --detach --wait postgres
python -m rag_app.cli database-init
python -m pytest tests/integration
```

## Planned Commands

Assignment-compatible entry points:

```bash
python index_documents.py --file data/input/document.pdf --strategy fixed
python search.py --query "What is proof of work?" --strategy fixed --top-k 5
```

Unified CLI:

```bash
python -m rag_app.cli index --file data/input/document.pdf --strategy fixed
python -m rag_app.cli search --query "What is proof of work?" --strategy paragraph --top-k 5
python -m rag_app.cli reset --yes
```

The supported chunking strategies will be `fixed`, `sentence`, and `paragraph`.
These commands currently expose help and validate their arguments, but report
that application functionality is unavailable. Only `database-init` and
`database-check` are functional in Stage 1.

## Architecture

- `rag_app/extractors`: structured PDF and DOCX text extraction.
- `rag_app/processing`: cleaning and the three chunking strategies.
- `rag_app/embeddings`: Gemini embedding requests and dimension validation.
- `rag_app/database`: implemented PostgreSQL connection, schema, and readiness
  boundaries; later persistence and vector search remain planned.
- `rag_app/services`: indexing and search use-case orchestration.
- `rag_app/cli.py`: argument handling and user-facing command output only.

See [`docs/architecture-decisions.md`](docs/architecture-decisions.md) for the
approved implementation decisions.

## Example Document

Future demonstrations will use the English Bitcoin Whitepaper. Download it from
the [Bitcoin project website](https://bitcoin.org/bitcoin.pdf) and place it in
`data/input/`; the PDF is not committed to this repository.
