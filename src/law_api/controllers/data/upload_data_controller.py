from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ...config import settings
from ...schemas import EmbedResponse, ErrorResponse, UploadResponse
from ...services import DocumentIngestionService


class UploadDataController:
    """Controller for the document upload use case."""

    def __init__(self, service: DocumentIngestionService) -> None:
        self.service = service
        self.router = APIRouter(prefix="/documents", tags=["data"])
        self.router.add_api_route(
            "/upload",
            self.upload,
            methods=["POST"],
            response_model=UploadResponse,
            responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
            status_code=status.HTTP_201_CREATED,
        )

    async def upload(self, file: Annotated[UploadFile, File()]) -> UploadResponse:
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")

        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = settings.upload_dir / f"{uuid4().hex}.pdf"
        pdf_path.write_bytes(content)
        try:
            return UploadResponse(**self.service.upload_and_chunk(pdf_path, filename))
        except ValueError as error:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Document processing failed") from error

        
    async def embed(self, document_id: int) -> EmbedResponse:
        try:
            return EmbedResponse(**self.service.embed_and_store(document_id))
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail="Embedding failed") from error