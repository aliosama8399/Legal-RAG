from abc import ABC, abstractmethod


class VectorDBInterface(ABC):
    """Abstract base class for tenant-isolated storage/vector-db providers.

    Every backend (Qdrant, PostgreSQL/pgvector) must inherit
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
        next_id_floor: int = 0,
        tenant_id: str = "org_default",
    ) -> int:
        """Register a pending (not yet embedded) document and return its id."""

    @abstractmethod
    async def get_document_chunks(
        self, document_id: int, tenant_id: str = "org_default"
    ) -> list[dict]: ...

    @abstractmethod
    async def save_embeddings(
        self,
        document_id: int,
        embeddings: list[list[float]],
        embedding_model: str,
        chunks: list[dict] | None = None,
        tenant_id: str = "org_default",
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
        tenant_id: str = "org_default",
    ) -> tuple[int, int]: ...

    @abstractmethod
    async def search(
        self,
        document_id: int | None,
        query_vector: list[float],
        top_k: int,
        tenant_id: str = "org_default",
    ) -> list[dict]: ...

    async def lexical_search(
        self,
        document_id: int | None,
        query_text: str,
        top_k: int,
        tenant_id: str = "org_default",
    ) -> list[dict]:
        """Optional lexical/BM25/exact-citation search for hybrid RRF fusion."""
        return []

    async def list_documents(self, tenant_id: str = "org_default") -> list[dict]:
        """List embedded documents scoped to a tenant."""
        return []

    async def delete_document(
        self, document_id: int, tenant_id: str = "org_default"
    ) -> bool:
        """Delete a document and its indexed chunks scoped to a tenant."""
        return False

    @abstractmethod
    async def save_chat_message(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        document_id: int | None,
        query_vector: list[float],
        tenant_id: str = "org_default",
    ) -> int | str: ...

    @abstractmethod
    async def list_chat_history(
        self,
        document_id: int | None = None,
        limit: int = 50,
        tenant_id: str = "org_default",
    ) -> list[dict]: ...
