# Jeen RAG Semantic Search

A Python application for indexing PDF and DOCX documents and retrieving relevant
content with Gemini embeddings and PostgreSQL/pgvector semantic search.

## Status

Stage 4 adds validated Gemini document embeddings over the ordered Stage 3
chunk model. One synchronous multi-content request produces a positional
768-dimensional vector for every chunk, and every accepted vector is manually
L2-normalized. Stage 1's PostgreSQL/pgvector environment, Stage 2 extraction,
and Stage 3 chunking remain available. Chunk persistence, duplicate-document
handling, complete indexing, and semantic search are not implemented yet.

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

Gemini embedding access requires a user-supplied `GEMINI_API_KEY`. Keep the key
only in the local environment or uncommitted `.env` file; never commit it. The
embedding layer requires `EMBEDDING_MODEL=gemini-embedding-001` and
`EMBEDDING_DIMENSION=768`, and rejects other non-empty model values or other
dimensions. Database-only operations do not require the Gemini key.

## Document Extraction

The application extraction boundary validates that a supplied path exists,
points to a readable regular file, and has a supported extension. `.pdf` and
`.docx` extensions are accepted case-insensitively.

PDF text is extracted page by page with one-based physical page numbers. Blank
pages are omitted without renumbering later pages. Recurring text is removed
only from conservative page-edge regions. DOCX paragraphs and table rows are
extracted from the main body in their original relative order; table cells use
` | ` as their visible separator. DOCX units do not have page numbers.

OCR is not supported. Scanned or image-only PDFs must contain embedded text to
be usable. Password-protected PDFs are not supported. Malformed files and
documents with no extractable text are rejected with safe application errors.
Extraction is currently an application API rather than a separate CLI command.

## Document Chunking

The application-level chunking boundary accepts an already extracted document
and one of three canonical strategies:

- `fixed` constructs 2,000-character windows with 500 characters of overlap and
  a 1,500-character step. Size is measured in Python characters, not tokens,
  words, bytes, or embedding-model units.
- `sentence` uses a lazily initialized `spacy.blank("en")` pipeline containing
  only the Sentencizer. Consecutive sentences are joined with one space up to
  the shared 2,000-character maximum.
- `paragraph` preserves DOCX extracted-unit boundaries and detects PDF
  paragraphs at blank lines. Consecutive paragraphs are joined with two
  newlines up to the shared maximum.

Normal sentence and paragraph chunks do not overlap. Paragraphs over 2,000
characters are grouped by sentence, and any individual sentence over 2,000
characters uses the fixed-size 500-character overlap as a deterministic
fallback. No downloaded spaCy language model is required.

Every PDF page is processed independently, so no chunk spans physical pages and
each PDF chunk retains its one-based source page number. DOCX chunks keep a null
page number and may group adjacent ordered units where the strategy allows it.
Chunk indexes are zero-based and continuous across the complete document.
Empty results are removed, but meaningful short titles, sentences, paragraphs,
table rows, and final windows are retained.

Chunking is an application API rather than a separate command. It does not read
files, request Gemini embeddings, or connect to PostgreSQL.

## Gemini Document Embeddings

The embedding boundary accepts one validated `ChunkedDocument` and submits all
of its chunk contents in their existing zero-based order through one synchronous
multi-content request. One-chunk documents use the same request path. Requests
use exactly `gemini-embedding-001`, request 768 output dimensions, set the task
type to retrieval document, and use only the source filename as the shared
retrieval title. Chunk metadata, source paths, credentials, and page labels are
not added to embedded text.

Gemini must return exactly one vector for every input chunk. Missing vectors,
count mismatches, dimensions other than 768, booleans, nulls, non-numeric or
non-finite values, and zero vectors reject the complete operation. Valid vectors
are converted to immutable tuples and manually L2-normalized to unit length.
Chunks and all source metadata remain unchanged and positionally paired with
their vectors. No partial embedded document is returned after any invalid
vector.

Provider failures are converted into safe application exceptions while the
original SDK exception remains available as the internal cause. Public errors
and metadata-only logs omit API keys, request bodies, chunk content, vectors,
raw responses, and local source paths. The application does not implement
custom retry or backoff and relies on the pinned Google Gen AI SDK behavior.
The embedding boundary does not read files, chunk documents, hash content,
connect to PostgreSQL, or insert rows.

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

Run only the Stage 2 document tests with:

```bash
python -m pytest tests/unit/test_document_validation.py \
  tests/unit/test_pdf_extraction.py \
  tests/unit/test_docx_extraction.py \
  tests/unit/test_text_cleaning.py
```

Run only the Stage 3 chunking tests with:

```bash
python -m pytest tests/unit/test_chunking_validation.py \
  tests/unit/test_fixed_chunking.py \
  tests/unit/test_sentence_chunking.py \
  tests/unit/test_paragraph_chunking.py
```

Run only the Stage 4 embedding tests with:

```bash
python -m pytest tests/unit/test_embedding_config.py \
  tests/unit/test_gemini_embeddings.py
```

These tests use constructed documents, small synthetic fixtures, and injected
Gemini fakes. Automated tests do not download documents or spaCy models, require
Gemini credentials, or contact Gemini. CI uses no Gemini key and all Gemini
responses and failures are mocked.

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

The supported chunking strategies are `fixed`, `sentence`, and `paragraph`.
The public indexing and search workflows remain incomplete because extraction,
chunking, embeddings, and persistence are not connected into a complete flow.
Database insertion, duplicate-document handling, query embedding, and semantic
search are not implemented. The commands expose help and validate their
arguments, but report that later-stage application functionality is
unavailable. Only `database-init` and `database-check` are currently functional
commands.

## Architecture

- `rag_app/extractors`: implemented validation and structured PDF/DOCX text
  extraction.
- `rag_app/processing`: implemented shared cleaning and all three bounded,
  page-aware chunking strategies.
- `rag_app/embeddings`: implemented Gemini document requests, strict response
  validation, safe provider errors, and shared L2 normalization.
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
