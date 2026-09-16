from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import Settings, settings
from .services import DocumentIngestionService, RAGQueryService
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.embeddings.EmbeddingProviderFactory import EmbeddingProviderFactory
from .stores.llm.LLMInterface import LLMInterface
from .stores.llm.LLMProviderFactory import LLMProviderFactory
from .stores.vectordb.VectorDBInterface import VectorDBInterface
from .stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from .tracking.mlflow_tracker import MLflowTracker

# Pre-configured instances for tests (set via create_app() before startup).
_preconfig: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared resources on startup; clean up on shutdown."""
    resolved_settings = _preconfig.get("settings") or settings

    storage = _preconfig.get("storage") or VectorDBProviderFactory.create(resolved_settings)
    embedder = _preconfig.get("embedder") or EmbeddingProviderFactory.create(resolved_settings.embedding_name)
    llm = _preconfig.get("llm") or LLMProviderFactory.create(
        resolved_settings.llm_provider,
        resolved_settings.llm_model,
        resolved_settings.llm_api_key,
        resolved_settings.llm_base_url,
    )
    # "tracker" absent from preconfig -> create a real tracker; explicitly None -> no tracking.
    tracker = _preconfig["tracker"] if "tracker" in _preconfig else MLflowTracker(
        resolved_settings.mlflow_tracking_uri, resolved_settings.mlflow_experiment
    )

    # Establish the database connection (Qdrant id recovery / Postgres schema init).
    await storage.connect()

    app.state.settings = resolved_settings
    app.state.storage = storage
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.tracker = tracker
    app.state.ingestion_service = DocumentIngestionService(resolved_settings, storage, embedder, tracker)
    app.state.rag_service = RAGQueryService(
        storage, embedder, llm,
        resolved_settings.rag_top_k, resolved_settings.llm_temperature, resolved_settings.llm_max_tokens,
    )

    yield

    # Shutdown: close the storage connection and end any dangling MLflow run.
    await storage.disconnect()
    if tracker is not None:
        try:
            import mlflow as mlflow_module

            mlflow_module.end_run()
        except Exception:
            pass


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingInterface | None = None,
    storage: VectorDBInterface | None = None,
    llm: LLMInterface | None = None,
    tracker: MLflowTracker | None = None,
    enable_tracking: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI app. Pre-configured instances are used for tests."""
    if settings is not None:
        _preconfig["settings"] = settings
    if storage is not None:
        _preconfig["storage"] = storage
    if embedder is not None:
        _preconfig["embedder"] = embedder
    if llm is not None:
        _preconfig["llm"] = llm
    if tracker is not None:
        _preconfig["tracker"] = tracker
    if not enable_tracking:
        _preconfig["tracker"] = None

    app = FastAPI(title="Egyptian Civil Code Data API", version="2.0.0", lifespan=lifespan)
    _include_routes(app)
    return app


def _include_routes(app: FastAPI) -> None:
    from .routes import data_routes, health_routes, rag_routes

    app.include_router(health_routes.router)
    app.include_router(data_routes.data_router)
    app.include_router(rag_routes.rag_router)


app = create_app()
