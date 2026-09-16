"""FastAPI dependency-injection entries.

Routes pull controllers/services from ``app.state`` via these callables.
Tests can override them with ``app.dependency_overrides``.
"""

from fastapi import Request

from .config import Settings
from .controllers.DataController import DataController
from .controllers.NLPController import NLPController
from .services import DocumentIngestionService, RAGQueryService
from .stores.embeddings.EmbeddingInterface import EmbeddingInterface
from .stores.llm.LLMInterface import LLMInterface
from .stores.vectordb.VectorDBInterface import VectorDBInterface
from .tracking.mlflow_tracker import MLflowTracker


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_embedding_provider(request: Request) -> EmbeddingInterface:
    return request.app.state.embedder


def get_llm_provider(request: Request) -> LLMInterface:
    return request.app.state.llm


def get_storage_provider(request: Request) -> VectorDBInterface:
    return request.app.state.storage


def get_tracker(request: Request) -> MLflowTracker:
    return request.app.state.tracker


def get_ingestion_service(request: Request) -> DocumentIngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RAGQueryService:
    return request.app.state.rag_service


def get_data_controller(request: Request) -> DataController:
    return DataController(request.app.state.ingestion_service, request.app.state.settings)


def get_nlp_controller(request: Request) -> NLPController:
    return NLPController(request.app.state.rag_service)
