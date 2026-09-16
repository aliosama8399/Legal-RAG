from ...helpers.config import Settings
from .VectorDBEnum import VectorDBEnum
from .VectorDBInterface import VectorDBInterface
from .providers import InMemoryStorageProvider, PostgreSQLProvider, QdrantProvider


class VectorDBProviderFactory:
    """Instantiate a storage provider from settings."""

    @staticmethod
    def create(settings: Settings) -> VectorDBInterface:
        provider = settings.storage_provider
        if provider == VectorDBEnum.QDRANT_LOCAL.value:
            return QdrantProvider(path=settings.qdrant_path, collection=settings.qdrant_collection)
        if provider == VectorDBEnum.QDRANT.value:
            return QdrantProvider(url=settings.qdrant_url, collection=settings.qdrant_collection)
        if provider == VectorDBEnum.PGVECTOR.value:
            return PostgreSQLProvider(settings.postgres_dsn)
        if provider == VectorDBEnum.IN_MEMORY_TEST.value:
            return InMemoryStorageProvider()
        choices = ", ".join(f"'{member.value}'" for member in VectorDBEnum)
        raise ValueError(f"LAW_API_STORAGE_PROVIDER must be one of: {choices}")
