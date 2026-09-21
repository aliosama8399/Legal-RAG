from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from ..VectorDBInterface import VectorDBInterface


class QdrantProvider(VectorDBInterface):
    """Async Qdrant client — works against a local path store or a server."""

    name = "qdrant"

    @staticmethod
    def _point_id(document_id: int, chunk_id: str) -> str:
        """Qdrant point IDs must be an unsigned int or a UUID. Derive a
        deterministic UUID so re-embedding a document overwrites the same
        points instead of duplicating them."""
        return str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_id}"))

    def __init__(self, collection: str, url: str = "", path: Path | None = None) -> None:
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self.client = AsyncQdrantClient(path=str(path))
        else:
            self.client = AsyncQdrantClient(url=url, timeout=10)
        self.collection = collection
        self.chat_history_collection = f"{collection}_chat_history"
        self.pending: dict[int, list[dict]] = {}
        self.next_document_id: int = 1
        self.next_chat_id: int = 1

    async def connect(self) -> None:
        # Recover next document id so restarts (e.g. uvicorn --reload) don't
        # reuse old document ids.
        try:
            await self.client.get_collection(self.collection)
        except Exception:
            # Collection does not exist yet — first run, start IDs at 1.
            self.next_document_id = 1
            return
        try:
            points, _ = await self.client.scroll(
                collection_name=self.collection, limit=10_000, with_payload=["document_id"]
            )
        except Exception:
            raise  # Collection exists but scroll failed; never silently reset ids.
        max_id = max((point.payload.get("document_id", 0) for point in points), default=0)
        self.next_document_id = max_id + 1

    async def disconnect(self) -> None:
        await self.client.close()

    async def save_pending_document(self, filename, source_path, articles, chunks, next_id_floor: int = 0):
        # Honor the caller's floor (derived from persisted pending files) so a
        # restart between upload and embed never reuses an id.
        document_id = max(self.next_document_id, next_id_floor)
        self.next_document_id = document_id + 1
        self.pending[document_id] = chunks
        return document_id

    async def get_document_chunks(self, document_id):
        try:
            return self.pending[document_id]
        except KeyError as error:
            raise ValueError(f"Document {document_id} was not found") from error

    async def _ensure_collection(self, dimension: int) -> None:
        try:
            collection = await self.client.get_collection(self.collection)
            if collection.config.params.vectors.size == dimension:
                return
            # Dimension mismatch – recreate with the correct dimension.
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        except Exception:
            # Collection does not exist yet – create it.
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    async def _index_points(self, document_id, filename, source_path, chunks, embeddings, embedding_model, dimension):
        await self._ensure_collection(dimension)
        points = [
            models.PointStruct(
                id=self._point_id(document_id, chunk["chunk_id"]),
                vector=vector,
                payload={**chunk, "document_id": document_id, "document_filename": filename,
                         "source_path": source_path, "embedding_model": embedding_model},
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        await self.client.upsert(collection_name=self.collection, points=points)

    async def save_document(self, filename, source_path, articles, chunks, embeddings, embedding_model):
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding")
        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise ValueError("Cannot index an empty embedding set")
        document_id = await self.save_pending_document(filename, source_path, articles, chunks)
        await self._index_points(document_id, filename, source_path, chunks, embeddings, embedding_model, dimension)
        return document_id, dimension

    async def save_embeddings(self, document_id, embeddings, embedding_model, chunks=None):
        if chunks is None:
            chunks = await self.get_document_chunks(document_id)
        if len(chunks) != len(embeddings):
            raise ValueError("Embedding count does not match stored chunks")
        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise ValueError("Cannot index an empty embedding set")
        await self._index_points(
            document_id, f"document-{document_id}", "", chunks, embeddings, embedding_model, dimension
        )
        return len(embeddings), dimension

    async def search(self, document_id, query_vector, top_k):
        if not await self.client.collection_exists(self.collection):
            raise ValueError(
                f"Document {document_id} was not found"
                if document_id is not None
                else "No relevant chunks were found"
            )
        query_filter = (
            models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            if document_id is not None
            else None
        )
        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        hits = response.points
        if not hits:
            message = (
                f"Document {document_id} was not found or has no embeddings — you must embed the file first"
                if document_id is not None
                else "No relevant chunks were found"
            )
            raise ValueError(message)
        return [{**hit.payload, "score": hit.score} for hit in hits]

    async def _ensure_chat_history_collection(self, dimension: int) -> None:
        try:
            collection = await self.client.get_collection(self.chat_history_collection)
            if collection.config.params.vectors.size != dimension:
                return
        except Exception:
            await self.client.create_collection(
                collection_name=self.chat_history_collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    async def save_chat_message(self, question, answer, sources, document_id, query_vector):
        await self._ensure_chat_history_collection(len(query_vector))
        chat_id = self.next_chat_id
        self.next_chat_id += 1
        point = models.PointStruct(
            id=chat_id,
            vector=query_vector,
            payload={
                "document_id": document_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        await self.client.upsert(collection_name=self.chat_history_collection, points=[point])
        return chat_id

    async def list_chat_history(self, document_id=None, limit=50):
        query_filter = (
            models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            if document_id is not None
            else None
        )
        try:
            points, _ = await self.client.scroll(
                collection_name=self.chat_history_collection,
                scroll_filter=query_filter,
                limit=limit,
            )
        except Exception:
            return []
        entries = [{"id": point.id, **point.payload} for point in points]
        entries.sort(key=lambda entry: entry["created_at"], reverse=True)
        return entries[:limit]
