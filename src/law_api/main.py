from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import Settings
from .config import settings as default_settings
from .lifecycle import initialize_services, shutdown_services
from .middleware import RequestContextMiddleware, register_exception_handlers


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
    """Create and configure the enterprise multi-tenant FastAPI SaaS application."""
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

    resolved_settings = settings or default_settings
    app = FastAPI(
        title="Legal-RAG Enterprise SaaS API",
        version="3.0.0",
        description="Multi-tenant Legal AI platform with hybrid retrieval, async ingestion, RBAC, and metering.",
        lifespan=lifespan,
    )
    app.state._preconfig = preconfig
    app.add_middleware(
        RequestContextMiddleware, rate_limit_rpm=resolved_settings.rate_limit_rpm
    )
    register_exception_handlers(app)
    _include_routes(app)
    return app


def _include_routes(app: FastAPI) -> None:
    from .routes import data_routes, health_routes, rag_routes, saas_routes

    app.include_router(health_routes.router)
    app.include_router(saas_routes.saas_router)
    app.include_router(data_routes.data_router)
    app.include_router(rag_routes.rag_router)


app = create_app()
