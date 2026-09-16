from pathlib import Path
from typing import Annotated

from fastapi import File, HTTPException, UploadFile

from ..config import Settings
from ..schemas import EmbedResponse, UploadResponse
from ..services import DocumentIngestionService
from .BaseController import BaseController


class DataController(BaseController):
    """Business logic for the document upload / embedding use cases."""

    def __init__(self, service: DocumentIngestionService, settings: Settings) -> None:
        super().__init__(settings)
        self.service = service

    def validate_upload(self, file: UploadFile) -> None:
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    async def read_upload(self, file: Annotated[UploadFile, File()]) -> bytes:
        content = await file.read(self.app_settings.max_upload_bytes + 1)
        if len(content) > self.app_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        return content

    def save_pdf(self, content: bytes) -> Path:
        """Persist the uploaded PDF under its content hash.

        Identical uploads reuse the same file instead of saving another
        copy (repeated test uploads of the same law PDF previously created
        a full duplicate each time).
        """
        import hashlib

        digest = hashlib.sha256(content).hexdigest()
        self.app_settings.upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.app_settings.upload_dir / f"{digest}.pdf"
        if not pdf_path.exists():
            pdf_path.write_bytes(content)
        return pdf_path

    async def upload_and_chunk(self, pdf_path: Path, filename: str) -> UploadResponse:
        try:
            return UploadResponse(**(await self.service.upload_and_chunk(pdf_path, filename)))
        except ValueError as error:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Document processing failed") from error

    async def embed_and_store(self, document_id: int) -> EmbedResponse:
        try:
            return EmbedResponse(**(await self.service.embed_and_store(document_id)))
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail="Embedding failed") from error
