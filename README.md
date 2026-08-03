# Jeen RAG Semantic Search

## Project Overview

Jeen RAG Semantic Search is a synchronous Python application that indexes PDF and
DOCX documents and retrieves semantically related chunks. It extracts and cleans
text, applies a selected chunking strategy, creates Gemini embeddings, stores the
chunks and metadata in PostgreSQL with pgvector, and searches them by cosine
distance.

The application returns matching source chunks and metadata. It does **not**
generate or synthesize a final natural-language answer.

## Current Implementation Status

The current implementation provides:

- PDF and DOCX extraction and cleaning.
- Fixed-size, sentence-based, and paragraph-based chunking.
- Gemini document and query embeddings using `gemini-embedding-001` with 768
  dimensions.
- Transactional PostgreSQL persistence with duplicate detection and atomic
  replacement.
- Strategy-scoped semantic search through pgvector's cosine-distance operator.
- Root-level indexing and search wrappers plus a unified Typer CLI.
- Unit and PostgreSQL integration tests that use deterministic Gemini fakes.

Expected validation, Gemini, and PostgreSQL failures produce concise CLI errors
without exposing credentials or local source paths. Destructive application-level
index reset and generated answers are not implemented.

## Supported Documents and Chunking Strategies

Document extensions are matched case-insensitively.

| Type | Extraction behavior |
| --- | --- |
| PDF (`.pdf`) | Extracted page by page with `pypdf`; chunks retain one-based physical page numbers. |
| DOCX (`.docx`) | Main-body paragraphs and table rows are extracted in document order with `python-docx`; page numbers are `NULL`. |

The `--strategy` option is required and accepts exactly `fixed`, `sentence`, or
`paragraph`:

- `fixed`: creates windows of at most 2,000 Python characters with 500
  characters of overlap and a 1,500-character step. The limit is measured in
  characters, not tokens, words, bytes, or model units.
- `sentence`: uses the English spaCy Sentencizer and normally emits one detected
  non-empty sentence per chunk without overlap. An individual sentence longer
  than 2,000 characters falls back to fixed-size windows with 500-character
  overlap. No downloaded spaCy language model is required.
- `paragraph`: normally emits one detected non-empty paragraph per chunk, which
  may contain several sentences. PDF paragraphs depend on blank-line separators
  in extracted text; DOCX extracted units provide paragraph boundaries. An
  oversized paragraph is split into independent sentence chunks, and an
  oversized individual sentence then uses fixed-size overlap.

Short semantic units are not combined. PDF pages are processed independently, so
chunks never span physical pages. Chunk indexes are zero-based and continuous
across the complete document.

## Architecture and Indexing Flow

Indexing performs these operations in order:

1. Validate that the path is a readable PDF or DOCX file and normalize the
   selected strategy.
2. Verify PostgreSQL and the expected pgvector schema.
3. Hash the exact file bytes with SHA-256 and check the stored document state.
4. Extract and clean text while retaining source order and available page
   metadata.
5. Produce chunks with the selected strategy.
6. Generate Gemini retrieval-document embeddings in ordered synchronous batches
   of at most 100 chunks.
7. Validate every vector, require exactly 768 finite dimensions, reject zero
   vectors, and L2-normalize valid vectors.
8. Persist all chunks and metadata in one PostgreSQL transaction.

Search validates the query and strategy, checks schema readiness, creates one
768-dimensional Gemini retrieval-query embedding, validates and normalizes it,
and retrieves only rows stored with the selected strategy. PostgreSQL orders the
retrieved candidates by the pgvector cosine-distance operator `<=>`.

## Prerequisites

Only Python 3.12 and the CI environment are tested; this README does not claim
support for other Python versions or operating systems.

| Prerequisite | Purpose |
| --- | --- |
| Python 3.12 | Runs the application, CLI, and tests. The repository's `.python-version` selects 3.12. |
| Git | Clones the repository and supports history secret scanning. |
| Docker with Docker Compose support | Runs the pinned PostgreSQL/pgvector service. Compose must support `docker compose up --wait`. |
| Gemini API key | Required for new or changed document embeddings and semantic-search query embeddings. |
| Internet access | Installs Python dependencies, downloads the Compose image and example PDF, and reaches Gemini for real requests. |
| Suitable shell | Sets environment variables and runs the documented commands. POSIX-compatible examples are used except where PowerShell is shown explicitly. |

PostgreSQL and pgvector do not need to be installed on the host. They run in
Docker Compose from the pinned `pgvector/pgvector:0.8.2-pg17-bookworm` image.

`curl` is optional because the Bitcoin PDF can be downloaded in a browser or
with PowerShell. Gitleaks 8.30.1 is optional for normal application use but is
required to reproduce the repository's complete secret-scanning quality gate.

## Installation

Start from a fresh clone:

```bash
git clone https://github.com/DavidSnir/jeen-rag-semantic-search.git
cd jeen-rag-semantic-search
```

Create a Python 3.12 virtual environment.

POSIX-compatible shell:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade the installer, install the pinned application dependencies, and create
the local environment file:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

In PowerShell, replace the final command with:

```powershell
Copy-Item .env.example .env
```

These help commands load configuration-free and do not contact PostgreSQL or
Gemini:

```bash
python -m rag_app.cli --help
python index_documents.py --help
python search.py --help
```

## Environment Configuration

Edit the uncommitted `.env` file before starting PostgreSQL. Values already
exported by the shell take precedence over values loaded from `.env`.

| Variable | Consumer | Required and expected value |
| --- | --- | --- |
| `GEMINI_API_KEY` | Application | Required when a Gemini embedding request is needed. Use a valid Gemini Developer API key; an empty placeholder is acceptable only for setup, database commands, and automated tests. |
| `POSTGRES_URL` | Application | Required for every database command, indexing operation, and search. Use a PostgreSQL URI such as `postgresql://USER:PASSWORD@HOST:PORT/DATABASE`. Its components must match the Compose values. |
| `POSTGRES_DB` | Docker Compose | Required. Names the database created when the Compose volume is first initialized. A synthetic local name is acceptable. |
| `POSTGRES_USER` | Docker Compose | Required. Names the PostgreSQL role created on the first empty volume. A synthetic local role is acceptable. |
| `POSTGRES_PASSWORD` | Docker Compose | Required. Initializes the local role password on the first empty volume. Use a non-reused synthetic local password and URL-encode URI-special characters in `POSTGRES_URL`. |
| `POSTGRES_PORT` | Docker Compose | Required. An available host port mapped to PostgreSQL's container port `5432`, for example `5432`. |
| `EMBEDDING_MODEL` | Application | Optional because it defaults to `gemini-embedding-001`. Any other non-empty model is rejected. |
| `EMBEDDING_DIMENSION` | Application | Optional because it defaults to `768`. If set, it must be the integer `768`. |
| `LOG_LEVEL` | Aggregate settings only | Optional and defaults to `INFO`; the current CLI keeps package logging quiet and does not configure output from this value. |

The individual `POSTGRES_*` values configure Compose, while the Python
application connects only through `POSTGRES_URL`. Changing database credentials
in `.env` after a volume has been initialized does not update the existing
PostgreSQL database, role, or password. Use the original values or deliberately
create a fresh volume.

Never commit `.env`. `.env.example` must contain placeholders only. Do not place
API keys, reusable passwords, private connection strings, or credentials in
README output, logs, screenshots, issue reports, or Git history.

Unit tests and normal integration tests use Gemini fakes and do not need a real
Gemini key. Integration tests require an explicitly exported, disposable
database whose name ends in `_test`; they erase all rows from its `chunks` table.

## PostgreSQL and pgvector Setup

Validate the Compose configuration, start PostgreSQL, and wait for its configured
health check:

```bash
docker compose config --quiet
docker compose up --detach --wait --wait-timeout 90 postgres
```

The image already includes pgvector. The service, network, and `postgres_data`
volume are scoped to the current Compose project; the configuration does not use
a fixed container name or checkout-global volume name.

Inspect service state, readiness, and logs:

```bash
docker compose ps postgres
docker compose exec -T postgres sh -c \
  'pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
docker compose logs postgres
```

The `sh -c` form expands the variables inside the container. Merely copying
`.env` does not export those values into the host shell.

Stop PostgreSQL while preserving the named volume and indexed data:

```bash
docker compose down
```

Start it again with the same `docker compose up` command. The named volume
preserves PostgreSQL state across ordinary container restarts.

For a deliberate complete local reset:

```bash
docker compose down --volumes --remove-orphans
```

**Warning:** `--volumes` permanently deletes this Compose project's complete
PostgreSQL volume, including every locally indexed chunk. It is not normal
shutdown or cleanup.

## Schema Initialization and Verification

The canonical schema is
[`rag_app/database/schema.sql`](rag_app/database/schema.sql). Docker mounts it at
`/docker-entrypoint-initdb.d/001-schema.sql`, so PostgreSQL applies it
automatically when it creates a new empty Compose volume. Entrypoint scripts do
not rerun for an existing volume.

Explicitly apply the canonical schema and validate the result:

```bash
python -m rag_app.cli database-init
```

Independently verify connectivity and the complete expected schema:

```bash
python -m rag_app.cli database-check
```

`database-init` is idempotent and can be run more than once. It is a bootstrap
operation, not a general migration framework, and it does not upgrade or repair
an arbitrary incompatible existing table. The project currently has no general
migration command. `database-check` is the strict readiness check used before
indexing and search; those operations do not automatically run
`database-init`.

The schema provides:

- The pgvector `vector` extension.
- A `public.chunks` table with chunk content and a `vector(768)` embedding.
- Source filename, source type, SHA-256 document hash, zero-based chunk index,
  chunking strategy, optional PDF page number, and creation timestamp.
- Constraints for non-empty content, supported source types and strategies,
  valid hashes, indexes, and page metadata.
- A uniqueness constraint over source filename, document hash, strategy, and
  chunk index to prevent duplicate version rows.
- B-tree indexes for document and hash lookup plus an HNSW
  `vector_cosine_ops` index for cosine search.

## Document Indexing

Before indexing, configure `.env`, start PostgreSQL, initialize and verify its
schema, and set a valid `GEMINI_API_KEY`.

Use the root wrapper:

```bash
python index_documents.py --file data/input/document.pdf --strategy fixed
```

Replace the example path with an existing supported document. The Bitcoin
demonstration below provides a directly reproducible input file.

Or use the equivalent unified command:

```bash
python -m rag_app.cli index \
  --file data/input/document.docx \
  --strategy paragraph
```

There is no default indexing strategy. New and changed documents require a live
Gemini request. An exact duplicate is detected after database readiness, file
validation, hashing, and stored-state lookup, but before extraction or Gemini
configuration is loaded.

Index identity and replacement behavior are:

- The same filename basename, exact file bytes, and strategy is skipped.
- The same filename and strategy with changed bytes atomically replaces the old
  version. Failed extraction, embedding, validation, or persistence preserves the
  existing version.
- Different strategies coexist for the same source file.
- Different basenames coexist even when their bytes are identical.
- Source directories are not stored, so files in different directories with the
  same basename and strategy represent one logical document and can skip or
  replace one another.

The document hash covers exact binary bytes. Rewriting PDF or DOCX container
metadata can therefore produce a new version even when extracted text appears
unchanged.

## Semantic Search

Search requires a query and one explicit strategy:

```bash
python search.py \
  --query "How does proof-of-work prevent double-spending?" \
  --strategy paragraph
```

The default result limit is `top_k=5`. Request a different positive limit with
`--top-k`:

```bash
python -m rag_app.cli search \
  --query "How does proof-of-work prevent double-spending?" \
  --strategy paragraph \
  --top-k 2
```

Search examines only rows indexed with the selected strategy; it never searches
across strategies. Results are ordered by pgvector cosine distance, nearest
retrieved candidate first. The CLI displays `score = 1 - cosine_distance` rounded
to four decimal places rather than displaying raw distance. Higher scores mean
greater cosine similarity, and scores may be negative.

The query is compatible with the HNSW index, but PostgreSQL decides whether to
use that index. When selected, HNSW retrieval is approximate rather than an
exhaustive exact scan. No relevance threshold is applied: if rows exist for the
strategy, the command can return weak matches. If no rows exist, it prints an
empty-result message and exits successfully after the required schema and Gemini
query embedding operations.

Results contain stored chunk text and metadata, not an AI-generated answer.
Search is read-only and does not expose stored vectors, hashes, row IDs, or
timestamps.

## Reproducible Bitcoin Whitepaper Demonstration

The demonstration uses Satoshi Nakamoto's English Bitcoin Whitepaper from the
[official Bitcoin project source](https://bitcoin.org/bitcoin.pdf). Keep local
input documents in `data/input/`, whose contents are already ignored by Git. Do
not commit the PDF or add another filename-specific ignore rule.

Download it in a POSIX-compatible shell:

```bash
curl --fail --location \
  --output data/input/bitcoin.pdf \
  https://bitcoin.org/bitcoin.pdf
```

Or in Windows PowerShell:

```powershell
Invoke-WebRequest `
  -Uri https://bitcoin.org/bitcoin.pdf `
  -OutFile data/input/bitcoin.pdf
```

The PDF used for the captured demonstration is 184,292 bytes with SHA-256:

```text
b1674191a88ec5cdd733e4240a81803105dc412d6c6708d53ab94fc248f4f553
```

Verify it in a POSIX-compatible shell:

```bash
shasum -a 256 data/input/bitcoin.pdf
```

Or in Windows PowerShell:

```powershell
(Get-FileHash data/input/bitcoin.pdf -Algorithm SHA256).Hash.ToLower()
```

If the checksum differs, do not expect the documented chunk counts or search
evidence to match; obtain the verified English PDF before continuing.

The following commands were run against an empty PostgreSQL volume using the
official PDF. Elapsed time varies by machine and provider response time.

```bash
python index_documents.py --file data/input/bitcoin.pdf --strategy fixed
python index_documents.py --file data/input/bitcoin.pdf --strategy sentence
python index_documents.py --file data/input/bitcoin.pdf --strategy paragraph
python index_documents.py --file data/input/bitcoin.pdf --strategy fixed
```

Actual indexing and duplicate-skip output:

```text
Indexed document: bitcoin.pdf | strategy=fixed | chunks=16 | elapsed=2.22s
Indexed document: bitcoin.pdf | strategy=sentence | chunks=161 | elapsed=7.07s
Indexed document: bitcoin.pdf | strategy=paragraph | chunks=142 | elapsed=6.57s
Skipped unchanged document: bitcoin.pdf | strategy=fixed | chunks=16 | elapsed=0.05s
```

Actual chunk-count comparison for the same file:

| Strategy | Chunks |
| --- | ---: |
| `fixed` | 16 |
| `sentence` | 161 |
| `paragraph` | 142 |

The reproducible search query is:

```text
How does proof-of-work prevent double-spending?
```

Run it against the paragraph representation and request two results:

```bash
python search.py \
  --query "How does proof-of-work prevent double-spending?" \
  --strategy paragraph \
  --top-k 2
```

Actual output from the indexed data:

```text
Result 1 | score=0.7263
source=bitcoin.pdf | type=PDF | strategy=paragraph | chunk=48 | page=3
Once the CPU
effort has been expended to make it satisfy the proof-of-work, the block cannot be changed
without redoing the work.
------------------------------------------------------------------------
Result 2 | score=0.7079
source=bitcoin.pdf | type=PDF | strategy=paragraph | chunk=55 | page=3
To modify a past block, an attacker would have to
redo the proof-of-work of the block and all blocks after it and then catch up with and surpass the
work of the honest nodes.
```

These scores and ranks came from a real Gemini request and may change if the
provider's embedding behavior changes. The metadata and stored chunks correspond
to the database evidence below.

## Database Inspection

The Compose image includes `psql`. This query uses the database name and role
inside the container, reports all three strategy counts, and inspects the two
chunks returned by the demonstration search. It shows dimensions only, never
full vectors.

```bash
docker compose exec -T postgres sh -c \
  'PGUSER="$POSTGRES_USER" PGDATABASE="$POSTGRES_DB" psql --no-psqlrc --pset pager=off' <<'SQL'
SELECT source_file,
       chunking_strategy AS strategy,
       count(*) AS chunks,
       min(vector_dims(embedding)) AS dimensions
FROM public.chunks
GROUP BY source_file, chunking_strategy
ORDER BY source_file, chunking_strategy;

SELECT source_file,
       chunking_strategy AS strategy,
       chunk_index,
       page_number,
       left(replace(content, chr(10), chr(32)), 88) AS content_preview,
       vector_dims(embedding) AS dimensions
FROM public.chunks
WHERE source_file = 'bitcoin.pdf'
  AND chunking_strategy = 'paragraph'
  AND chunk_index IN (48, 55)
ORDER BY chunk_index;
SQL
```

Actual output; content is intentionally limited by the query to an 88-character
preview:

```text
 source_file | strategy  | chunks | dimensions
-------------+-----------+--------+------------
 bitcoin.pdf | fixed     |     16 |        768
 bitcoin.pdf | paragraph |    142 |        768
 bitcoin.pdf | sentence  |    161 |        768
(3 rows)

 source_file | strategy  | chunk_index | page_number |                                     content_preview                                      | dimensions
-------------+-----------+-------------+-------------+------------------------------------------------------------------------------------------+------------
 bitcoin.pdf | paragraph |          48 |           3 | Once the CPU effort has been expended to make it satisfy the proof-of-work, the block ca |        768
 bitcoin.pdf | paragraph |          55 |           3 | To modify a past block, an attacker would have to redo the proof-of-work of the block an |        768
(2 rows)
```

## Testing and Quality Checks

Validate dependencies, linting, formatting, CLI loading, deterministic fixtures,
and Compose configuration:

```bash
python -m pip check
python -c "from google import genai; import coverage, docx, dotenv, pgvector, psycopg, pypdf, pytest, pytest_cov, ruff, spacy, typer"
python -m ruff check .
python -m ruff format --check .
python -m rag_app.cli --help
python index_documents.py --help
python search.py --help
python tests/fixtures/generate_fixtures.py
git diff --exit-code -- tests/fixtures
docker compose config --quiet
```

Run the unit suite directly when a database is not needed:

```bash
python -m pytest tests/unit
```

The complete coverage gate below runs unit tests and integration tests. The
subshell uses a separate Compose project, explicitly exported synthetic
credentials, and a database ending in `_test`. Its trap removes the disposable
volume even when a command fails.

**Warning:** integration tests run `TRUNCATE TABLE public.chunks RESTART IDENTITY`
before and after every test. Never point them at a database containing data you
want to keep, even if its name happens to end in `_test`.

```bash
(
  set -e
  python -m coverage erase
  python -m coverage run --source=rag_app -m pytest tests/unit
  export POSTGRES_DB=rag_app_test
  export POSTGRES_USER=rag_app_test
  export POSTGRES_PASSWORD=local-test-only-password
  export POSTGRES_PORT=55432
  export POSTGRES_URL=postgresql://rag_app_test:local-test-only-password@localhost:55432/rag_app_test
  trap 'docker compose -p rag-integration-test down --volumes --remove-orphans' EXIT
  docker compose -p rag-integration-test up --detach --wait --wait-timeout 90 postgres
  python -m rag_app.cli database-init
  python -m coverage run --append --source=rag_app -m pytest tests/integration
  python -m coverage report --fail-under=80
)
```

Automated tests remove ambient Gemini credentials and block construction of a
real Gemini client. They do not contact Gemini or need a valid key.

With Gitleaks 8.30.1 installed, run the same redacted current-tree and complete
Git-history scans as CI:

The directory scan examines ignored and untracked files, including a populated
`.env`. Run it from a clean verification checkout without real credentials, or
temporarily keep `.env` outside the repository while scanning. Do not expose or
allowlist a real key merely to make the local scan pass.

```bash
gitleaks version
gitleaks dir --redact --no-banner .
gitleaks git --redact --no-banner --log-opts="--all -m" .
```

## Troubleshooting

Expected application failures are written to standard error without a traceback.
Exit code `0` means success, duplicate skip, or an empty search; code `1` means an
expected runtime/configuration failure; code `2` means invalid CLI usage.

| Symptom | Likely cause and safe corrective action | Existing database data |
| --- | --- | --- |
| Missing `GEMINI_API_KEY` | Set a valid key in the uncommitted `.env` file or process environment. Database-only commands and automated tests do not need it. | Unchanged. A new document writes nothing; a pending replacement preserves its old version. |
| Gemini authentication or permission failure | Verify the key and that its project can use `gemini-embedding-001`. Revoke and replace a key if it may have been exposed; never print it for diagnosis. | Unchanged. |
| Gemini quota or rate-limit failure | Check project quota and billing, wait when appropriate, and retry manually. The application does not add custom retry or backoff. | Unchanged. |
| Gemini transport or temporary provider failure | Verify internet, DNS, proxy, and provider status before retrying. | Unchanged. |
| Missing or invalid `POSTGRES_URL` | Set a valid PostgreSQL connection URI whose credentials, port, and database match the running service. Do not publish the URI. | Unchanged. |
| `PostgreSQL is unavailable` | Start the `postgres` service, inspect `docker compose ps postgres`, and verify the URL and host-port mapping. | Unchanged. |
| Database health check fails | Inspect `docker compose logs postgres`, check required Compose values, port conflicts, and whether credentials were changed after volume initialization. Do not delete the volume unless all local data is disposable. | Diagnostics are read-only; `down --volumes` would delete everything. |
| Schema is missing or incompatible | Run `database-init`, then `database-check`. Initialization is idempotent but not a migration framework; investigate incompatible existing objects instead of bypassing validation. | Initialization preserves compatible rows. Deleting the volume to repair a disposable database removes all rows. |
| Unsupported file extension | Supply a readable `.pdf` or `.docx`; other extensions and extensionless files are not supported. | Unchanged. |
| File is missing, not regular, or unreadable | Correct the path or file permissions and retry. The CLI reports only the basename for path errors. | Unchanged. |
| Document is empty after cleaning | Use a document containing meaningful extractable text. | Unchanged. |
| Scanned or image-only PDF | OCR is not implemented. Obtain a PDF with embedded text or perform OCR outside this application before indexing a supported file. | Unchanged. |
| Password-protected or malformed document | Supply an unencrypted, valid PDF or DOCX. Password handling is not implemented. | Unchanged. |
| Embedding dimension mismatch or invalid vector | Keep `EMBEDDING_MODEL=gemini-embedding-001` and `EMBEDDING_DIMENSION=768`; do not pad, truncate, or manually insert provider vectors. Retry only after resolving provider/configuration issues. | Unchanged; no partial document or replacement is persisted. |
| No indexed results for the selected strategy | Verify the requested strategy and index the document with that same strategy. An empty search is successful and cross-strategy fallback does not occur. | Search is read-only. |
| `Skipped unchanged document` | This is expected when filename, strategy, and exact-byte hash match. Choose another strategy or index genuinely changed content if a new representation is intended. | Unchanged. |
| Integration tests reject the database | Export `POSTGRES_URL` explicitly and target a separate disposable database whose name ends in `_test`. Do not weaken the guard. | The guard fails before test truncation; a valid test target is deliberately erased during tests. |
| Ruff lint failure | Fix the reported code issue and rerun `python -m ruff check .`; do not suppress rules merely to pass the gate. | Application data is unaffected. |
| Ruff formatting failure | Run `python -m ruff format .`, review the diff, and rerun the check. | Application data is unaffected. |
| Coverage is below 80% | Add or correct meaningful tests, rerun both suites with coverage, and inspect uncovered behavior. | Only the explicitly disposable integration database is truncated. |
| Gitleaks failure | Review the redacted finding. Remove and revoke real secrets, then clean affected Git history through an approved process; do not add broad ignores for unexplained findings. | Application data is unaffected. |
| `reset --yes` reports unavailable | This is the implemented behavior. Use Compose volume deletion only when a complete destructive local reset is intended. | The command changes nothing. |

## Known Limitations

- OCR is not supported. Scanned or image-only PDFs require embedded text before
  this application can index them.
- PDF extraction uses `pypdf`; visual layout is not fully preserved.
- PDF paragraph identification depends on paragraph separators preserved in the
  extracted text. Visually separate paragraphs may be merged when a PDF does not
  expose a reliable blank-line boundary.
- Password-protected PDFs are not supported.
- Sentence detection uses an English spaCy Sentencizer and is configured for
  English documents.
- Fixed-size limits and overlap are measured in Python characters, not tokens.
- Semantic search is scoped to exactly one chunking strategy at a time.
- PostgreSQL may use the approximate HNSW index; retrieval has no relevance
  threshold.
- The application retrieves matching chunks but does not generate a final
  answer.
- Source identity uses only filename basename and strategy, not source directory.
- Destructive index reset is not implemented through the application.
- Removing the Compose volume deletes all locally indexed rows and all other
  PostgreSQL state in that volume.
- Integration tests truncate the complete `public.chunks` table and must run only
  against an explicitly disposable test database.

## Data Cleanup and Reset Behavior

Normal operations have limited effects:

- `docker compose down` removes the service container and network but preserves
  the named PostgreSQL volume and indexed rows.
- Search is read-only.
- Duplicate indexing performs no write.
- Changed-content indexing atomically replaces only rows with the same source
  filename basename and strategy; failure preserves the previous version.
- `python -m rag_app.cli reset --yes` exits with an unavailable-feature error and
  changes no data.

Destructive operations are separate:

- `docker compose down --volumes --remove-orphans` permanently deletes the
  current Compose project's complete local PostgreSQL volume.
- Integration tests truncate every row in `public.chunks` on their configured
  test database before and after each test.

Always verify the Compose project and database target before either destructive
operation. There is no application command that selectively or globally resets
the index.
