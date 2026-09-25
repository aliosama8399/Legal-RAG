"""FastAPI dependency-injection entries.

Routes pull controllers/services from ``app.state`` via these callables.
Resources are initialized lazily on first use (bento-mounted apps don't run
the FastAPI lifespan), via an idempotent lock in ``lifecycle``.
"""

from fastapi import Request

from .controllers.DataController import DataController
from .controllers.NLPController import NLPController
from .lifecycle import ensure_services


async def get_data_controller(request: Request) -> DataController:
    await ensure_services(request.app)
    return DataController(request.app.state.ingestion_service, request.app.state.settings)


async def get_nlp_controller(request: Request) -> NLPController:
    await ensure_services(request.app)
    return NLPController(request.app.state.rag_service)
