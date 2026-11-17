"""Asynchronous ingestion pipeline worker with job tracking, recovery, and idempotency."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..auth.store import TenantStore
from ..services import DocumentIngestionService

logger = logging.getLogger(__name__)


class IngestionJobManager:
    """Schedules and tracks background PDF extraction, chunking, and embedding jobs."""

    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        tenant_store: TenantStore,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.tenant_store = tenant_store
        self._tasks: set[asyncio.Task] = set()
        self._active_jobs: dict[str, asyncio.Task] = {}
        stale_count = self.tenant_store.fail_stale_ingestion_jobs()
        if stale_count:
            logger.warning("Marked %d orphaned ingestion job(s) as FAILED after startup", stale_count)

    def submit_ingest(
        self,
        *,
        pdf_path: Path,
        filename: str,
        tenant_id: str,
        user_id: str,
        idempotency_key: str | None = None,
        request_id: str = "",
    ) -> dict:
        job, deduplicated = self.tenant_store.create_ingestion_job(
            tenant_id=tenant_id,
            filename=filename,
            idempotency_key=idempotency_key,
        )
        if deduplicated:
            existing_task = self._active_jobs.get(job["id"])
            if job["status"] == "READY" or (existing_task is not None and not existing_task.done()):
                return job
            # Orphaned QUEUED/PROCESSING job from a crashed or restarted worker — re-run it.
            job = self.tenant_store.reset_ingestion_job(job["id"])

        task = asyncio.create_task(
            self._run_job(
                job_id=job["id"],
                pdf_path=pdf_path,
                filename=filename,
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
            )
        )
        self._tasks.add(task)
        self._active_jobs[job["id"]] = task
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run_job(
        self,
        *,
        job_id: str,
        pdf_path: Path,
        filename: str,
        tenant_id: str,
        user_id: str,
        request_id: str = "",
    ) -> None:
        try:
            self.tenant_store.update_ingestion_job(
                job_id,
                status="PROCESSING",
                stage="extracting",
                progress_pct=20,
            )
            upload_res = await self.ingestion_service.upload_and_chunk(
                pdf_path, filename, tenant_id=tenant_id
            )
            document_id = upload_res["document_id"]
            total_chunks = max(1, upload_res["chunks"])

            self.tenant_store.update_ingestion_job(
                job_id,
                status="PROCESSING",
                stage="embedding",
                progress_pct=60,
                document_id=document_id,
                articles=upload_res["articles"],
                chunks=upload_res["chunks"],
            )

            def _on_embed_progress(done: int, total: int) -> None:
                pct = 60 + int(35 * done / max(1, total))
                stage = "indexing" if done >= total else f"embedding ({done}/{total})"
                self.tenant_store.update_ingestion_job(
                    job_id,
                    status="PROCESSING",
                    stage=stage,
                    progress_pct=min(98, pct),
                    document_id=document_id,
                    articles=upload_res["articles"],
                    chunks=total_chunks,
                )

            embed_res = await self.ingestion_service.embed_and_store(
                document_id, tenant_id=tenant_id, progress_callback=_on_embed_progress
            )

            self.tenant_store.update_ingestion_job(
                job_id,
                status="READY",
                stage="completed",
                progress_pct=100,
                document_id=document_id,
                articles=upload_res["articles"],
                chunks=embed_res["embedded_chunks"],
            )
            self.tenant_store.record_usage(
                tenant_id=tenant_id,
                user_id=user_id,
                operation="document_ingest",
                model=embed_res["embedding_model"],
                prompt_tokens=embed_res["embedded_chunks"] * 64,
                completion_tokens=0,
                request_id=request_id,
            )
            self.tenant_store.record_audit_log(
                tenant_id=tenant_id,
                actor_id=user_id,
                action="document.ingested",
                resource_type="document",
                resource_id=str(document_id),
                metadata={
                    "job_id": job_id,
                    "filename": filename,
                    "chunks": embed_res["embedded_chunks"],
                },
                request_id=request_id,
            )
        except Exception as error:
            logger.exception("Async ingestion job failed job_id=%s", job_id)
            self.tenant_store.update_ingestion_job(
                job_id,
                status="FAILED",
                stage="failed",
                progress_pct=0,
                error=str(error),
            )
        finally:
            self._active_jobs.pop(job_id, None)

    async def shutdown(self) -> None:
        if not self._tasks:
            return
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._active_jobs.clear()
