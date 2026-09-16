from datetime import UTC, datetime

from ..VectorDBInterface import VectorDBInterface


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class InMemoryStorageProvider(VectorDBInterface):
    """Test-only provider. Production storage choices are PostgreSQL and Qdrant."""

    name = "in-memory-test"

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.chat_history: list[dict] = []

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def save_pending_document(self, filename, source_path, articles, chunks):
        self.documents.append(
            {
                "filename": filename,
                "source_path": source_path,
                "articles": articles,
                "chunks": chunks,
                "embeddings": None,
            }
        )
        return len(self.documents)

    async def get_document_chunks(self, document_id):
        try:
            return self.documents[document_id - 1]["chunks"]
        except IndexError as error:
            raise ValueError(f"Document {document_id} was not found") from error

    async def save_embeddings(self, document_id, embeddings, embedding_model, chunks=None):
        document = self.documents[document_id - 1]
        if len(document["chunks"]) != len(embeddings):
            raise ValueError("Embedding count does not match stored chunks")
        document["embeddings"] = embeddings
        document["embedding_model"] = embedding_model
        return len(embeddings), len(embeddings[0]) if embeddings else 0

    async def save_document(self, filename, source_path, articles, chunks, embeddings, embedding_model):
        document_id = await self.save_pending_document(filename, source_path, articles, chunks)
        _, dimension = await self.save_embeddings(document_id, embeddings, embedding_model)
        return document_id, dimension

    async def search(self, document_id, query_vector, top_k):
        if document_id is None:
            candidates = [(index + 1, document) for index, document in enumerate(self.documents)]
        else:
            if document_id > len(self.documents):
                raise ValueError(f"Document {document_id} was not found")
            candidates = [(document_id, self.documents[document_id - 1])]

        scored = []
        for doc_id, document in candidates:
            embeddings = document.get("embeddings")
            if not embeddings:
                continue
            scored.extend(
                {**chunk, "document_id": doc_id, "score": _cosine_similarity(query_vector, vector)}
                for chunk, vector in zip(document["chunks"], embeddings)
            )
        if not scored:
            message = (
                f"Document {document_id} has no embeddings yet"
                if document_id is not None
                else "No relevant chunks were found"
            )
            raise ValueError(message)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def save_chat_message(self, question, answer, sources, document_id, query_vector):
        chat_id = len(self.chat_history) + 1
        self.chat_history.append(
            {
                "id": chat_id,
                "document_id": document_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        return chat_id

    async def list_chat_history(self, document_id=None, limit=50):
        entries = [
            entry
            for entry in self.chat_history
            if document_id is None or entry["document_id"] == document_id
        ]
        return list(reversed(entries))[:limit]
