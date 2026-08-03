# Jeen RAG Semantic Search

A Python application for indexing PDF and DOCX documents and retrieving relevant
content with Gemini embeddings and PostgreSQL/pgvector semantic search.

## Status

Stage 7 provides hardened, synchronous document indexing and semantic search for
PDF and DOCX content. Expected validation, Gemini, and PostgreSQL failures use
safe application exceptions, concise CLI messages, and predictable exit codes
without tracebacks. Duplicate indexing is skipped and changed content is
atomically replaced within the documented filename and strategy scope.
Generated answers and index reset are not implemented.

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

Indexing and semantic search require both a reachable PostgreSQL database with
the canonical schema and a valid Gemini key. Automated tests inject Gemini
fakes and neither need nor use a real key.

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

## Document Indexing

Before indexing, configure `.env`, start PostgreSQL, and initialize and verify
the canonical schema as described above. Set a valid `GEMINI_API_KEY`, then use
either the assignment-compatible entry point:

```bash
python index_documents.py --file data/input/document.pdf --strategy fixed
```

or the equivalent unified CLI command:

```bash
python -m rag_app.cli index --file data/input/document.docx --strategy paragraph
```

Both `.pdf` and `.docx` files are supported, and `fixed`, `sentence`, and
`paragraph` are the available strategies. Indexing calculates SHA-256 over the
exact binary bytes of the source file before extraction. This means even a
binary-only change, such as rewritten PDF or DOCX container metadata, produces
a different hash whether or not the extracted text changed.

Replacement scope is the source filename basename plus chunking strategy; the
source directory is intentionally not stored. The resulting behavior is:

- The same basename, exact bytes, and strategy is skipped without extraction,
  a Gemini request, or new inserts.
- The same basename and strategy with different bytes replaces all chunks in
  that scope.
- The same basename indexed with a different strategy coexists as a separate
  indexed representation.
- A different basename coexists even when its binary content is identical.
- Renaming a file therefore creates a separate indexed document rather than
  replacing chunks stored under the old basename.

This basename-only identity has an important limitation: files in different
directories with the same basename and strategy are treated as one logical
document. Depending on their exact bytes, the later operation will skip or
replace the earlier one.

Persistence serializes concurrent updates for one basename and strategy, then
rechecks duplicate state. Deletion of an old version, insertion of every new
chunk, and post-insert verification occur in one transaction. A failure rolls
back the complete replacement, so old chunks are preserved and no partial new
version remains. Insertion prefers PostgreSQL `COPY FROM STDIN` and falls back
to `executemany()` only when the cursor does not provide COPY.

Each row stores chunk content and its normalized embedding together with
`source_file`, `document_hash`, `source_type`, `chunk_index`,
`chunking_strategy`, `page_number`, and the database-generated `created_at`.
PDF page numbers remain one-based; DOCX page numbers are `NULL`. A successful
command prints whether the document was indexed, replaced, or skipped, followed
by its basename, strategy, chunk count, and elapsed time.

Extraction, chunking, and Gemini embedding complete before persistence begins.
If Gemini fails or returns any invalid embedding, a new document writes no
rows, and an existing version awaiting replacement remains unchanged. Gemini
never produces partial persisted document data.

## Semantic Search

Before searching, configure `.env`, start PostgreSQL, initialize and verify the
canonical schema, and index at least one document with the strategy to search.
Set a valid `GEMINI_API_KEY`, then use either the assignment-compatible entry
point:

```bash
python search.py --query "What is proof of work?" --strategy fixed
```

or the equivalent unified CLI command:

```bash
python -m rag_app.cli search \
  --query "What is proof of work?" \
  --strategy paragraph \
  --top-k 10
```

Search trims and validates the query, then embeds it with exactly
`gemini-embedding-001`, the `RETRIEVAL_QUERY` task type, and 768 output
dimensions. Query requests contain no retrieval title. The returned vector is
strictly validated and L2-normalized using the same numerical boundary as
document embeddings.

Retrieval executes a read-only SQL `SELECT` against `public.chunks`, filters by
the selected chunking strategy, orders by pgvector cosine distance with `<=>`,
and applies the requested result limit. The default `top_k` is `5`; pass a
positive `--top-k` value to override it. Results are ordered nearest first and
show chunk content, rank, `source_file`, `source_type`, `chunk_index`, strategy,
PDF page number when available, and a similarity score calculated as
`1 - cosine_distance`.

Higher scores indicate greater cosine similarity. Search is limited to the
selected strategy and returns stored chunks, not generated answers. It never
loads stored vectors or document hashes into the CLI result.

No relevance threshold is applied. If indexed chunks exist for the selected
strategy, search returns the nearest neighbors even when they are irrelevant to
the query; callers must interpret the score accordingly. If no rows exist for
the strategy, the command prints an empty-result message and exits successfully
with status code `0`. Search does not insert, update, or delete indexed data and
does not generate a synthesized answer.

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

Run the Stage 5 unit tests with:

```bash
python -m pytest tests/unit/test_document_hashing.py \
  tests/unit/test_indexing_service.py \
  tests/unit/test_indexing_cli.py \
  tests/unit/test_chunk_persistence.py
```

Run the Stage 6 semantic-search unit tests with:

```bash
python -m pytest tests/unit/test_query_embeddings.py \
  tests/unit/test_search_validation.py \
  tests/unit/test_search_service.py \
  tests/unit/test_semantic_search_repository.py \
  tests/unit/test_search_cli.py
```

These tests use constructed documents, small synthetic fixtures, and injected
Gemini fakes. Automated document and query embedding tests do not download
documents or spaCy models, require Gemini credentials, or contact Gemini. CI
uses no Gemini key and all Gemini responses and failures are mocked. CI uses
pytest discovery for the complete unit and integration test directories, so
new tests do not require a manually maintained test-file list.

Run the Stage 7 error-handling tests with:

```bash
python -m pytest tests/unit/test_cli_error_handling.py \
  tests/unit/test_root_entrypoints.py \
  tests/unit/test_database_connection.py \
  tests/unit/test_database_repository_errors.py
```

Database integration tests require the healthy Compose service and a matching
`POSTGRES_URL` in `.env`:

```bash
docker compose up --detach --wait postgres
python -m rag_app.cli database-init
python -m pytest tests/integration/test_database_setup.py \
  tests/integration/test_indexing_persistence.py \
  tests/integration/test_semantic_search.py
```

## Commands

The assignment-compatible indexing and search entry points are functional:

```bash
python index_documents.py --file data/input/document.pdf --strategy fixed
python search.py --query "What is proof of work?" --strategy fixed --top-k 5
```

The corresponding unified commands are also functional:

```bash
python -m rag_app.cli index --file data/input/document.pdf --strategy fixed
python -m rag_app.cli search --query "What is proof of work?" --strategy paragraph --top-k 5
```

The reset entry point exposes help and validates explicit confirmation, but its
application functionality remains unavailable:

```bash
python -m rag_app.cli reset --yes
```

Generated answers and destructive index reset are not implemented.
`database-init` and `database-check` remain functional database commands.

## Troubleshooting

Expected application failures are written to standard error without a Python
traceback. The root scripts and unified commands use the same application
handlers and exit-code contract.

| Exit code | Meaning |
| ---: | --- |
| `0` | Successful indexing, duplicate skip, replacement, populated search, or empty search |
| `1` | Expected application, configuration, Gemini, PostgreSQL, or unavailable-feature failure |
| `2` | Invalid command syntax, missing option, unsupported strategy, invalid query, invalid `top_k`, or missing reset confirmation |

| Symptom | Action |
| --- | --- |
| `PostgreSQL is unavailable` | Start the database service and verify that `POSTGRES_URL` targets it. |
| `The database schema is not ready` | Run `python -m rag_app.cli database-init`, then rerun `database-check`. Verify the pinned PostgreSQL and pgvector versions if initialization fails. |
| Missing `GEMINI_API_KEY` | Set the variable in the environment or uncommitted `.env` file. |
| Gemini authentication or permission failure | Verify `GEMINI_API_KEY` and that the associated project can use the configured embedding model. |
| Gemini quota or rate-limit failure | Check the Gemini project's quota and billing status before retrying manually. The application does not retry automatically. |
| Unsupported document | Supply a readable PDF or DOCX file. Other extensions are rejected before Gemini or database writes. |
| Document has no extractable text | Use a document containing embedded text. OCR and image-only scanned PDFs are not supported. |
| No indexed results found | This is a successful empty search with exit code `0`. Initialize the schema and index content with the selected strategy if results were expected. |
| Reset is unavailable | `reset --yes` exits with code `1` and does not change the database. |

Run the same coverage gate used by CI with:

```bash
python -m coverage erase
python -m coverage run --source=rag_app -m pytest tests/unit
python -m coverage run --append --source=rag_app -m pytest tests/integration
python -m coverage report --fail-under=80
```

## Architecture

- `rag_app/extractors`: implemented validation and structured PDF/DOCX text
  extraction.
- `rag_app/processing`: implemented shared cleaning and all three bounded,
  page-aware chunking strategies.
- `rag_app/embeddings`: implemented Gemini document and retrieval-query
  requests, strict response validation, safe provider errors, and shared L2
  normalization.
- `rag_app/database`: implemented PostgreSQL connection, schema, readiness,
  atomic document persistence, and read-only pgvector cosine search.
- `rag_app/services`: complete indexing and semantic-search orchestration.
- `rag_app/cli.py`: argument handling and user-facing command output only.

See [`docs/architecture-decisions.md`](docs/architecture-decisions.md) for the
approved implementation decisions.

## Example Document

Future demonstrations will use the English Bitcoin Whitepaper. Download it from
the [Bitcoin project website](https://bitcoin.org/bitcoin.pdf) and place it in
`data/input/`; the PDF is not committed to this repository.
