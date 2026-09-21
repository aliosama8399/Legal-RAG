import json

import psycopg

from ..VectorDBInterface import VectorDBInterface

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS law_documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS law_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES law_documents(id),
    chunk_id TEXT NOT NULL,
    article_number INTEGER NOT NULL,
    citation TEXT NOT NULL,
    source_page INTEGER NOT NULL,
    text_ar TEXT NOT NULL,
    text_en TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    metadata_json JSONB NOT NULL,
    UNIQUE(document_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS law_vector_config (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    dimension INTEGER NOT NULL,
    CHECK (id)
);
CREATE TABLE IF NOT EXISTS law_chat_history (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgreSQLProvider(VectorDBInterface):
    """Async PostgreSQL/pgvector storage using psycopg's async API."""

    name = "postgresql"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._initialized = False
        self._vector_dimension: int | None = None

    def _connect(self) -> psycopg.AsyncConnection:
        return psycopg.AsyncConnection.connect(self.dsn).__await__() if False else psycopg.AsyncConnection.connect(self.dsn)

    async def connect(self) -> None:
        """Create extensions and tables (idempotent)."""
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cursor.execute(_SCHEMA_SQL)
                await cursor.execute("SELECT dimension FROM law_vector_config LIMIT 1")
                row = await cursor.fetchone()
                self._vector_dimension = row[0] if row else None
        self._initialized = True

    async def disconnect(self) -> None:
        # Connections are opened per operation; nothing persistent to close.
        return None

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.connect()

    async def _ensure_vector_column(self, cursor, dimension: int) -> None:
        # pgvector columns have a fixed dimension; the first embed call fixes it.
        if self._vector_dimension is None:
            await cursor.execute(f"ALTER TABLE law_chunks ADD COLUMN IF NOT EXISTS embedding vector({dimension})")
            try:
                await cursor.execute(
                    "CREATE INDEX IF NOT EXISTS law_chunks_embedding_idx "
                    "ON law_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                )
            except Exception:
                pass  # index is an optimization; search still works without it
            await cursor.execute("INSERT INTO law_vector_config (id, dimension) VALUES (TRUE, %s)", (dimension,))
            self._vector_dimension = dimension
        elif self._vector_dimension != dimension:
            raise ValueError(
                f"PostgreSQL vector column dimension is {self._vector_dimension}, received {dimension}"
            )

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(repr(float(value)) for value in vector) + "]"

    async def save_document(self, filename, source_path, articles, chunks, embeddings, embedding_model):
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        document_id = await self.save_pending_document(filename, source_path, articles, chunks)
        await self.save_embeddings(document_id, embeddings, embedding_model)
        return document_id, len(embeddings[0]) if embeddings else 0

    async def save_pending_document(self, filename, source_path, articles, chunks, next_id_floor: int = 0):
        # PostgreSQL BIGSERIAL never reuses ids; the floor kwarg exists for
        # interface parity.
        await self._ensure_initialized()
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """INSERT INTO law_documents
                    (filename, source_path, article_count, chunk_count)
                    VALUES (%s, %s, %s, %s) RETURNING id""",
                    (filename, source_path, articles, len(chunks)),
                )
                row = await cursor.fetchone()
                document_id = row[0]
                for chunk in chunks:
                    await cursor.execute(
                        """INSERT INTO law_chunks
                        (document_id, chunk_id, article_number, citation, source_page, text_ar, text_en,
                         chunk_text, metadata_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            document_id, chunk["chunk_id"], chunk["article_number"], chunk["citation"],
                            chunk["source_page"], chunk["text_ar"], chunk["text_en"], chunk["chunk_text"],
                            json.dumps({key: chunk.get(key) for key in ("book", "chapter", "section", "topic", "is_repealed")}),
                        ),
                    )
        return int(document_id)

    async def get_document_chunks(self, document_id):
        await self._ensure_initialized()
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT chunk_id, chunk_text FROM law_chunks WHERE document_id = %s ORDER BY id",
                    (document_id,),
                )
                rows = await cursor.fetchall()
        if not rows:
            raise ValueError(f"Document {document_id} was not found")
        return [{"chunk_id": row[0], "chunk_text": row[1]} for row in rows]

    async def save_embeddings(self, document_id, embeddings, embedding_model, chunks=None):
        await self._ensure_initialized()
        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise ValueError("Cannot store an empty embedding set")
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await self._ensure_vector_column(cursor, dimension)
                await cursor.execute(
                    "SELECT id FROM law_chunks WHERE document_id = %s ORDER BY id", (document_id,)
                )
                rows = await cursor.fetchall()
                ids = [row[0] for row in rows]
                if not ids or len(ids) != len(embeddings):
                    raise ValueError("Embedding count does not match stored chunks")
                for chunk_row_id, vector in zip(ids, embeddings):
                    await cursor.execute(
                        "UPDATE law_chunks SET embedding = %s::vector WHERE id = %s",
                        (self._vector_literal(vector), chunk_row_id),
                    )
                await cursor.execute(
                    "UPDATE law_documents SET embedding_model = %s, embedding_dimension = %s WHERE id = %s",
                    (embedding_model, dimension, document_id),
                )
        return len(embeddings), dimension

    async def search(self, document_id, query_vector, top_k):
        await self._ensure_initialized()
        if self._vector_dimension is None:
            raise ValueError("No embedded documents were found — you must embed the file first")
        literal = self._vector_literal(query_vector)
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                if document_id is None:
                    await cursor.execute(
                        """
                        SELECT document_id, chunk_id, article_number, citation, source_page, chunk_text,
                               1 - (embedding <=> %s::vector) AS score
                        FROM law_chunks
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (literal, literal, top_k),
                    )
                else:
                    await cursor.execute(
                        """
                        SELECT document_id, chunk_id, article_number, citation, source_page, chunk_text,
                               1 - (embedding <=> %s::vector) AS score
                        FROM law_chunks
                        WHERE document_id = %s AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (literal, document_id, literal, top_k),
                    )
                rows = await cursor.fetchall()
        if not rows:
            message = (
                f"Document {document_id} was not found or has no embeddings — you must embed the file first"
                if document_id is not None
                else "No relevant chunks were found"
            )
            raise ValueError(message)
        columns = ["document_id", "chunk_id", "article_number", "citation", "source_page", "chunk_text", "score"]
        return [dict(zip(columns, row)) for row in rows]

    async def save_chat_message(self, question, answer, sources, document_id, query_vector):
        await self._ensure_initialized()
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """INSERT INTO law_chat_history (document_id, question, answer, sources_json)
                    VALUES (%s, %s, %s, %s) RETURNING id""",
                    (document_id, question, answer, json.dumps(sources)),
                )
                row = await cursor.fetchone()
        return int(row[0])

    async def list_chat_history(self, document_id=None, limit=50):
        await self._ensure_initialized()
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            async with connection.cursor() as cursor:
                if document_id is None:
                    await cursor.execute(
                        """SELECT id, document_id, question, answer, sources_json, created_at
                        FROM law_chat_history ORDER BY id DESC LIMIT %s""",
                        (limit,),
                    )
                else:
                    await cursor.execute(
                        """SELECT id, document_id, question, answer, sources_json, created_at
                        FROM law_chat_history WHERE document_id = %s ORDER BY id DESC LIMIT %s""",
                        (document_id, limit),
                    )
                rows = await cursor.fetchall()
        columns = ["id", "document_id", "question", "answer", "sources", "created_at"]
        return [
            {**dict(zip(columns, row)), "created_at": row[5].isoformat()}
            for row in rows
        ]
