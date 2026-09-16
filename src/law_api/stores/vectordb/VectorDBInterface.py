from abc import ABC, abstractmethod


class VectorDBInterface(ABC):
    """Abstract base class for storage/vector-db providers.

    Every backend (Qdrant, PostgreSQL/pgvector, in-memory) must inherit
    from this class and implement every method below. Connection setup is
    done in ``connect`` (called from the app lifespan), not ``__init__``.
    """

    name: str

    @abstractmethod
    async def connect(self) -> None:
        """Open connections / recover state. Called once on app startup."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release connections. Called on app shutdown."""

    @abstractmethod
    async def save_pending_document(
        self,
        filename: str,
        source_path: str,
        articles: int,
        chunks: list[dict],
    ) -> int: ...

    @abstractmethod
    async def get_document_chunks(self, document_id: int) -> list[dict]: ...

    @abstractmethod
    async def save_embeddings(
        self,
        document_id: int,
        embeddings: list[list[float]],
        embedding_model: str,
        chunks: list[dict] | None = None,
    ) -> tuple[int, int]: ...

    @abstractmethod
    async def save_document(
        self,
        filename: str,
        source_path: str,
        articles: int,
        chunks: list[dict],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> tuple[int, int]: ...

    @abstractmethod
    async def search(
        self,
        document_id: int | None,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict]: ...

    @abstractmethod
    async def save_chat_message(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        document_id: int | None,
        query_vector: list[float],
    ) -> int: ...

    @abstractmethod
    async def list_chat_history(
        self,
        document_id: int | None = None,
        limit: int = 50,
    ) -> list[dict]: ...
