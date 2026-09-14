from fastapi import APIRouter

from ..controllers.rag.rag_query_controller import RagQueryController
from ..services import RAGQueryService


class RagRoutes:
    """Route module for retrieval and RAG answer workflows."""

    def __init__(self, rag_service: RAGQueryService) -> None:
        self.router = APIRouter(prefix="/api/v1")
        self.router.include_router(RagQueryController(rag_service).router)
