from fastapi import APIRouter

from ..controllers.data.upload_data_controller import UploadDataController
from ..schemas import EmbedResponse
from ..services import DocumentIngestionService


class DataRoutes:
    """Route module for data workflows."""

    def __init__(self, ingestion_service: DocumentIngestionService) -> None:
        self.router = APIRouter(prefix="/api/v1")
        controller = UploadDataController(ingestion_service)
        self.router.include_router(controller.router)
        self.router.add_api_route(
            "/documents/{document_id}/embed",
            controller.embed,
            methods=["POST"],
            response_model=EmbedResponse,
            tags=["embeddings"],
        )