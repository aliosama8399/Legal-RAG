from fastapi import FastAPI

from .config import settings
from .providers.embeddings.model_provider import EmbeddingProvider, create_embedding_provider
from .providers.llm.base import LLMProvider
from .providers.llm.factory import create_llm_provider
from .providers.storage.postgresql_provider import PostgreSQLProvider
from .providers.storage.qdrant_provider import QdrantProvider
from .routes.data_routes import DataRoutes
from .routes.health_routes import HealthRoutes
from .routes.rag_routes import RagRoutes
from .services import DocumentIngestionService, RAGQueryService
from .tracking.mlflow_tracker import MLflowTracker


def create_storage_provider():
    if settings.storage_provider == "qdrant-local":
        return QdrantProvider(path=settings.qdrant_path, collection=settings.qdrant_collection)
    if settings.storage_provider == "postgresql":
        return PostgreSQLProvider(settings.postgres_dsn)
    if settings.storage_provider == "qdrant":
        return QdrantProvider(url=settings.qdrant_url, collection=settings.qdrant_collection)
    raise ValueError("LAW_API_STORAGE_PROVIDER must be 'qdrant-local', 'qdrant', or 'postgresql'")


def create_app(
    embedder: EmbeddingProvider | None = None,
    storage=None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    application = FastAPI(title="Egyptian Civil Code Data API", version="2.0.0")
    storage_provider = storage or create_storage_provider()
    embedding_provider = embedder or create_embedding_provider(settings.embedding_name)
    llm_provider = llm or create_llm_provider(
        settings.llm_provider, settings.llm_model, settings.llm_api_key, settings.llm_base_url
    )
    ingestion_service = DocumentIngestionService(
        settings=settings,
        storage=storage_provider,
        embedder=embedding_provider,
        tracker=MLflowTracker(settings.mlflow_tracking_uri, settings.mlflow_experiment),
    )
    rag_service = RAGQueryService(
        storage=storage_provider,
        embedder=embedding_provider,
        llm=llm_provider,
        top_k=settings.rag_top_k,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    application.include_router(HealthRoutes().router)
    application.include_router(DataRoutes(ingestion_service).router)
    application.include_router(RagRoutes(rag_service).router)
    return application


app = create_app()
 