CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    source_file TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunking_strategy TEXT NOT NULL,
    page_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chunks_pkey PRIMARY KEY (id),
    CONSTRAINT chunks_content_nonempty_check CHECK (char_length(content) > 0),
    CONSTRAINT chunks_document_hash_sha256_check
        CHECK (document_hash ~ '^[0-9A-Fa-f]{64}$'),
    CONSTRAINT chunks_source_type_check CHECK (source_type IN ('PDF', 'DOCX')),
    CONSTRAINT chunks_chunk_index_nonnegative_check CHECK (chunk_index >= 0),
    CONSTRAINT chunks_chunking_strategy_check
        CHECK (chunking_strategy IN ('fixed', 'sentence', 'paragraph')),
    CONSTRAINT chunks_page_number_positive_check
        CHECK (page_number IS NULL OR page_number > 0),
    CONSTRAINT chunks_docx_page_number_null_check
        CHECK (source_type <> 'DOCX' OR page_number IS NULL),
    CONSTRAINT chunks_document_version_strategy_chunk_key UNIQUE (
        source_file,
        document_hash,
        chunking_strategy,
        chunk_index
    )
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_file_chunking_strategy
    ON public.chunks (source_file, chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_chunks_document_hash_chunking_strategy
    ON public.chunks (document_hash, chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw_cosine
    ON public.chunks USING hnsw (embedding vector_cosine_ops);
