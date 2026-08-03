# Architecture Decisions

This log records the approved implementation constraints for the document-indexing
and semantic-search application. Stage 0 established the initial decisions;
later stages add resolved implementation details without removing them.

## Document Extraction

1. PDF extraction uses `pypdf`.
2. DOCX extraction uses `python-docx`.
3. PDF content is extracted page by page.
4. DOCX paragraphs and table content should be extracted in document order as closely as possible.
5. OCR is out of scope because the selected documents contain embedded text.

## Text Cleaning

6. Collapse duplicate spaces and tabs.
7. Collapse repeated blank lines.
8. Remove recurring headers and footers.
9. Preserve ordinary punctuation and meaningful special characters.
10. Remove only control characters and invisible characters.

## Chunking

11. Implement three strategies: fixed-size, sentence-based, and paragraph-based.
12. Fixed-size chunks use 2,000 characters.
13. Fixed-size overlap is 500 characters.
14. Fixed-size measurement is based on characters, not words or tokens.
15. Sentence splitting uses `spacy.blank("en")` with the Sentencizer.
16. Documents are expected to be in English.
17. A paragraph longer than 2,000 characters is first split by sentence.
18. A single sentence longer than 2,000 characters falls back to fixed-size splitting with a 500-character overlap.

## PostgreSQL and pgvector

19. PostgreSQL with pgvector will run through Docker Compose.
20. Use a version-pinned PostgreSQL/pgvector image.
21. Use a named volume for persistence.
22. Add a PostgreSQL health check.
23. Do not add pgAdmin or unrelated services.
24. Create the `vector` extension idempotently.
25. Create the chunk table idempotently.
26. Initialization should run automatically rather than requiring manual SQL.
27. The application should also verify that the required schema exists during startup.

## Embeddings

28. Use the Google Gen AI Python SDK package `google-genai`.
29. Use the model `gemini-embedding-001`.
30. Request `output_dimensionality=768`.
31. Store embeddings in a `vector(768)` column.
32. Validate that every embedding has exactly 768 dimensions before persistence.
33. Stop indexing with a clear error if an embedding has an invalid dimension.
34. Do not insert partial document data after an embedding failure.

## Search

35. Use cosine similarity.
36. Use pgvector's cosine-distance operator `<=>`.
37. Use `vector_cosine_ops` for the future vector index.
38. Present similarity as `1 - cosine_distance`.
39. Return five search results by default.
40. Allow the user to override the result count through `top_k`.

## Indexing and Metadata

41. Calculate a SHA-256 hash from the document content.
42. Treat the combination of document, content hash, and chunking strategy as the indexing identity.
43. Skip insertion when the same content and strategy are already indexed.
44. Replace old chunks when the file content changes.
45. Perform deletion and replacement inside one transaction.
46. Store `source_file`, `document_hash`, `chunk_index`, `chunking_strategy`, `page_number`, `source_type`, and `created_at` metadata.
47. Use `NULL` for `page_number` when it is unavailable, including DOCX documents.
48. Store `source_type` as `PDF` or `DOCX`.

## Database Insertion

49. Prefer `COPY FROM STDIN` for chunk insertion.
50. Use `executemany()` as a fallback when COPY is unavailable.
51. Use one transaction for each document and chunking-strategy combination.
52. Commit only after all chunks have been inserted successfully.

## Gemini Failures

53. Use the default retry behavior supplied by the pinned Google Gen AI SDK.
54. Do not add a custom retry or backoff mechanism during the initial implementation.
55. If the SDK ultimately fails, fail the complete document-indexing transaction.

## CLI

56. Use Typer.
57. Provide `index`, `search`, and `reset` commands.
58. Preserve the required root-level `index_documents.py` entry point.
59. Keep CLI handlers separate from the application logic.

## Example Document and Tests

60. Use the English Bitcoin Whitepaper for README demonstrations.
61. Include a link and download instructions rather than committing the PDF by default.
62. Require a minimum of 80% line coverage before submission.
63. Fully test text cleaning, all three chunking strategies, embedding-dimension validation, duplicate-index prevention, and semantic-search query construction.
64. Use mocked Gemini responses in automated tests.
65. Run integration tests against PostgreSQL with pgvector.
66. Perform one manual smoke test against the real Gemini API before submission.

## Stage 1 Database Implementation

67. Use `pgvector/pgvector:0.8.2-pg17-bookworm`, selecting PostgreSQL 17 and pgvector 0.8.2 explicitly.
68. Use HNSW for the vector index with the `vector_cosine_ops` operator class and default construction parameters.
69. Enforce chunk identity with a uniqueness constraint over `source_file`, `document_hash`, `chunking_strategy`, and `chunk_index`.
70. Keep `rag_app/database/schema.sql` as the canonical idempotent schema instead of introducing a migration framework.
71. Mount the canonical schema directly into Docker's first-volume initialization directory rather than maintaining a second SQL copy.
72. Use a five-second Psycopg connection timeout so invalid targets fail promptly.
73. Keep autocommit disabled and let callers explicitly control database transactions.
74. Run application-level schema initialization and complete catalog verification because Docker entrypoint scripts do not rerun for existing volumes.

## Stage 2 Document Extraction Implementation

75. Represent extracted content with an immutable shared document model containing the source filename, canonical source type, and ordered text units so later chunking does not reopen parser objects.
76. Use one-based physical page numbers for PDF units and preserve gaps caused by blank pages.
77. Use `NULL` page numbers for every DOCX unit.
78. Traverse the DOCX main body with `python-docx` ordered inner content so paragraphs and tables remain interleaved.
79. Represent each non-empty DOCX table row as cleaned cells joined by the visible ` | ` separator while retaining empty-cell positions.
80. Detect recurring PDF margins only within at most two non-empty lines at each page edge, require occurrence on at least two and at least half of all pages, and normalize only the matching physical page-number token for footer comparison.
81. Keep header and footer comparison normalization separate from returned text and remove matches only from the edge region where they were detected.
82. Make shared cleaning deterministic and idempotent while preserving meaningful Unicode, punctuation, symbols, numbers, and line separation.
83. Detect empty or non-extractable documents only after margin removal, shared cleaning, and empty-unit removal.
84. Do not add OCR or password handling; scanned PDFs require embedded text and password-protected PDFs are rejected.
85. Expose extraction as an application API without adding a separate extraction CLI command or presenting indexing as complete.

## Stage 3 Chunking Implementation

86. Represent chunks and complete chunking results with shared immutable models containing no embeddings, database identifiers, or document hashes.
87. Apply one common 2,000-character maximum to fixed, sentence, and paragraph chunks, including separators inserted during grouping.
88. Use a 1,500-character fixed-window step so successive full windows have exactly 500 characters of overlap.
89. Do not overlap normal sentence or paragraph chunks; overlap applies only to the fixed strategy and oversized individual-sentence fallback.
90. Join grouped sentences with one regular space and grouped paragraphs with two newline characters.
91. Lazily initialize and reuse one `spacy.blank("en")` pipeline containing only the Sentencizer component, without loading a downloaded language model.
92. Split every oversized paragraph by sentence and route every oversized individual sentence through the one shared fixed-size splitter.
93. Process PDF pages independently for every strategy so no chunk spans physical pages and every chunk retains one unambiguous physical page number.
94. Assign continuous zero-based chunk indexes only after all ordered document chunks have been generated and empty results removed.
95. Remove only empty or whitespace-only chunks and preserve all meaningful short chunks without imposing a minimum content length.
96. Normalize external strategy strings by trimming and lowercasing, reject aliases or unsupported values, and retain only the canonical `fixed`, `sentence`, or `paragraph` value.
97. Define the canonical chunking strategy type once outside the CLI and share it between the CLI and processing layer.

## Stage 4 Gemini Embeddings Implementation

98. Embed one complete chunked document with one synchronous multiple-content Gemini request, including documents containing only one chunk.
99. Do not use Gemini asynchronous Batch API jobs, files, polling, or background processing for document embedding.
100. Use exactly `gemini-embedding-001` with `output_dimensionality=768` and reject every configured alternative.
101. Use the `RETRIEVAL_DOCUMENT` task type and the source filename basename as the common retrieval title for all chunks in the request.
102. Submit only chunk content in existing `chunk_index` order and pair returned vectors with chunks strictly by response position.
103. Require exactly one returned embedding for every submitted chunk and reject missing, null, empty, shorter, or longer response collections.
104. Require exactly 768 numeric non-boolean finite values in every returned vector without truncating, padding, repeating, or repairing values.
105. Reject zero vectors and vectors with non-finite norms, then manually L2-normalize every accepted vector and verify its unit norm.
106. Return one immutable embedded-document result that retains each original chunk and all source metadata without adding database identifiers, hashes, timestamps, or credentials.
107. Validate and normalize every vector before constructing the embedded document so any invalid vector prevents a partial result.
108. Inject a compatible Gemini client in tests and construct the official Developer API client only after input and credential validation in production.
109. Make no application-level retry, backoff, sleep, or second request; use only behavior provided by the pinned Google Gen AI SDK.
110. Load embedding-only configuration independently of PostgreSQL while retaining a separate database-only configuration loader independent of Gemini.
111. Exclude the Gemini API key from settings representations, public exceptions, logs, request content, and returned models.
112. Convert SDK API and transport failures into categorized application exceptions without copying raw provider details, while preserving the provider exception as the internal cause.
113. Log embedding metadata only: approved model, dimension, chunk count, source filename basename, strategy, response count, and safe failure category.
114. Share strict vector validation and L2 normalization through a numerical embedding module so later query embeddings can use identical behavior.
