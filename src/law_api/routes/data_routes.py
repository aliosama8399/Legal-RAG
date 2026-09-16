import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from ..controllers.DataController import DataController
from ..deps import get_data_controller
from ..schemas import EmbedResponse, ErrorResponse, UploadResponse

logger = logging.getLogger(__name__)

data_router = APIRouter(prefix="/api/v1")


@data_router.post(
    "/documents/upload",
    response_model=UploadResponse,
    tags=["data"],
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    status_code=201,
)
async def upload_document(
    controller: Annotated[DataController, Depends(get_data_controller)],
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    """Upload a law PDF, extract bilingual articles and citation-preserving chunks."""
    controller.validate_upload(file)
    content = await controller.read_upload(file)
    pdf_path = controller.save_pdf(content)
    try:
        return await controller.upload_and_chunk(pdf_path, Path(file.filename or "").name)
    except Exception:
        logger.exception("Upload failed")
        raise


@data_router.post(
    "/documents/{document_id}/embed",
    response_model=EmbedResponse,
    tags=["embeddings"],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def embed_document(
    document_id: int,
    controller: Annotated[DataController, Depends(get_data_controller)],
) -> EmbedResponse:
    """Embed the pending chunks of a document and store them in the vector store."""
    return await controller.embed_and_store(document_id)
