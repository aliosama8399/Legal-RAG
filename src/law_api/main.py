from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import Settings
from .lifecycle import initialize_services, shutdown_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared resources on startup; clean up on shutdown."""
    await initialize_services(app)
    yield
    await shutdown_services(app)


def create_app(
    settings: Settings | None = None,
    embedder=None,
    storage=None,
    llm=None,
    tracker=None,
    enable_tracking: bool = True,
    reranker=None,
) -> FastAPI:
    """Create and configure the FastAPI app.

    Pre-configured instances (tests) sit on ``app.state._preconfig`` and are
    consumed by the lazy initializer — works under plain uvicorn (lifespan)
    and under BentoML (mounted ASGI, lifespan never runs), because deps.py
    also calls ``ensure_services`` before every request.
    """
    preconfig: dict = {}
    if settings is not None:
        preconfig["settings"] = settings
    if storage is not None:
        preconfig["storage"] = storage
    if embedder is not None:
        preconfig["embedder"] = embedder
    if llm is not None:
        preconfig["llm"] = llm
    if tracker is not None:
        preconfig["tracker"] = tracker
    if not enable_tracking:
        preconfig["tracker"] = None
    if reranker is not None:
        preconfig["reranker"] = reranker

    app = FastAPI(title="Egyptian Civil Code Data API", version="2.0.0", lifespan=lifespan)
    app.state._preconfig = preconfig
    _include_routes(app)
    return app


def _include_routes(app: FastAPI) -> None:
    from .routes import data_routes, health_routes, rag_routes

    app.include_router(health_routes.router)
    app.include_router(data_routes.data_router)
    app.include_router(rag_routes.rag_router)


app = create_app()
