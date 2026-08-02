# Jeen RAG Semantic Search

A Python application for indexing PDF and DOCX documents and retrieving relevant
content with Gemini embeddings and PostgreSQL/pgvector semantic search.

## Status

Repository bootstrap only. The command-line interfaces and architectural
boundaries are defined, but document extraction, chunking, embeddings, database
persistence, and semantic search are not implemented yet.

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

## Configuration

Create a local environment file and provide real values at runtime:

```bash
cp .env.example .env
```

`GEMINI_API_KEY` and `POSTGRES_URL` are required when the corresponding
application features are implemented. Never commit `.env` or real credentials.

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
that application functionality is unavailable during Stage 0.

## Planned Architecture

- `rag_app/extractors`: structured PDF and DOCX text extraction.
- `rag_app/processing`: cleaning and the three chunking strategies.
- `rag_app/embeddings`: Gemini embedding requests and dimension validation.
- `rag_app/database`: PostgreSQL connections, schema, persistence, and vector search.
- `rag_app/services`: indexing and search use-case orchestration.
- `rag_app/cli.py`: argument handling and user-facing command output only.

See [`docs/architecture-decisions.md`](docs/architecture-decisions.md) for the
approved implementation decisions.

## Example Document

Future demonstrations will use the English Bitcoin Whitepaper. Download it from
the [Bitcoin project website](https://bitcoin.org/bitcoin.pdf) and place it in
`data/input/`; the PDF is not committed to this repository.
