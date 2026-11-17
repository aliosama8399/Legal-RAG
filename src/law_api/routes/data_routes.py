import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile

from ..auth.deps import get_tenant_context, require_role
from ..auth.models import TenantContext
from ..controllers.DataController import DataController
from ..deps import get_data_controller
from ..exceptions import ResourceNotFoundError
from ..schemas import (
    DocumentListResponse,
    EmbedResponse,
    ErrorResponse,
    IngestionJobResponse,
    UploadResponse,
)

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
    request: Request,
    controller: Annotated[DataController, Depends(get_data_controller)],
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    """Upload a law PDF, extract bilingual articles and citation-preserving chunks."""
    controller.validate_upload(file)
    content = await controller.read_upload(file)
    pdf_path = controller.save_pdf(content)
    try:
        result = await controller.upload_and_chunk(
            pdf_path, Path(file.filename or "").name, tenant_id=ctx.tenant_id
        )
    except Exception:
        logger.exception("Upload failed")
        raise
    request.app.state.tenant_store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="document.uploaded",
        resource_type="document",
        resource_id=str(result.document_id),
        metadata={"filename": result.filename, "chunks": result.chunks},
        ip_address=request.client.host if request.client else "",
        request_id=getattr(request.state, "request_id", ""),
    )
    return result


@data_router.post(
    "/documents/{document_id}/embed",
    response_model=EmbedResponse,
    tags=["embeddings"],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def embed_document(
    request: Request,
    document_id: int,
    controller: Annotated[DataController, Depends(get_data_controller)],
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
) -> EmbedResponse:
    """Embed the pending chunks of a document and store them in the vector store."""
    result = await controller.embed_and_store(document_id, tenant_id=ctx.tenant_id)
    store = request.app.state.tenant_store
    request_id = getattr(request.state, "request_id", "")
    store.record_usage(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        operation="document_embed",
        model=result.embedding_model,
        prompt_tokens=result.embedded_chunks * 64,
        completion_tokens=0,
        request_id=request_id,
    )
    store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="document.embedded",
        resource_type="document",
        resource_id=str(document_id),
        metadata={"chunks": result.embedded_chunks, "model": result.embedding_model},
        ip_address=request.client.host if request.client else "",
        request_id=request_id,
    )
    return result


@data_router.post(
    "/documents/ingest",
    response_model=IngestionJobResponse,
    tags=["data"],
    status_code=202,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def ingest_document_async(
    request: Request,
    controller: Annotated[DataController, Depends(get_data_controller)],
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
    file: Annotated[UploadFile, File()],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> IngestionJobResponse:
    """Enqueue asynchronous PDF extraction, chunking, and vector indexing (HTTP 202)."""
    controller.validate_upload(file)
    content = await controller.read_upload(file)
    pdf_path = controller.save_pdf(content)
    filename = Path(file.filename or "").name
    job = request.app.state.job_manager.submit_ingest(
        pdf_path=pdf_path,
        filename=filename,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        idempotency_key=idempotency_key,
        request_id=getattr(request.state, "request_id", ""),
    )
    return IngestionJobResponse(**job)


@data_router.get(
    "/documents/jobs/{job_id}",
    response_model=IngestionJobResponse,
    tags=["data"],
    responses={404: {"model": ErrorResponse}},
)
async def get_ingestion_job(
    request: Request,
    job_id: str,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> IngestionJobResponse:
    """Poll the status and progress of an asynchronous document ingestion job."""
    job = request.app.state.tenant_store.get_ingestion_job(ctx.tenant_id, job_id)
    if job is None:
        raise ResourceNotFoundError(f"Ingestion job '{job_id}' was not found")
    return IngestionJobResponse(**job)


@data_router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["data"],
)
async def list_documents(
    controller: Annotated[DataController, Depends(get_data_controller)],
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> DocumentListResponse:
    """List documents belonging to the active organization."""
    docs = await controller.list_documents(tenant_id=ctx.tenant_id)
    return DocumentListResponse(results=docs)


@data_router.delete(
    "/documents/{document_id}",
    status_code=204,
    tags=["data"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    request: Request,
    document_id: int,
    controller: Annotated[DataController, Depends(get_data_controller)],
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
) -> None:
    """Delete a tenant-owned document and purge its vectors."""
    await controller.delete_document(document_id, tenant_id=ctx.tenant_id)
    request.app.state.tenant_store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="document.deleted",
        resource_type="document",
        resource_id=str(document_id),
        ip_address=request.client.host if request.client else "",
        request_id=getattr(request.state, "request_id", ""),
    )
