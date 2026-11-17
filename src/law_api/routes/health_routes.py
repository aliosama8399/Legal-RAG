from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from ..config import settings
from ..lifecycle import ensure_services
from ..schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


async def _collect_health_status(request: Request) -> tuple[int, int, bool]:
    await ensure_services(request.app)
    docs = await request.app.state.ingestion_service.list_documents(tenant_id="org_default")
    ready_docs = [doc for doc in docs if doc.get("status", "ready") == "ready"]
    documents_indexed = len(ready_docs)
    chunks_indexed = sum(int(doc.get("chunk_count", 0)) for doc in ready_docs)
    embedder = getattr(request.app.state, "embedder", None)
    reranker = getattr(request.app.state, "reranker", None)
    models_loaded = bool(
        (embedder is None or getattr(embedder, "is_loaded", True))
        and (reranker is None or getattr(reranker, "is_loaded", True))
    )
    return documents_indexed, chunks_indexed, models_loaded


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    require_indexed: bool = False,
) -> HealthResponse | JSONResponse:
    documents_indexed, chunks_indexed, models_loaded = await _collect_health_status(request)
    if require_indexed and (documents_indexed == 0 or not models_loaded):
        payload = HealthResponse(
            status="unhealthy",
            documents_indexed=documents_indexed,
            chunks_indexed=chunks_indexed,
            models_loaded=models_loaded,
        )
        return JSONResponse(status_code=503, content=payload.model_dump())
    return HealthResponse(
        status="ok",
        documents_indexed=documents_indexed,
        chunks_indexed=chunks_indexed,
        models_loaded=models_loaded,
    )


@router.get("/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health_ready(request: Request) -> HealthResponse | JSONResponse:
    documents_indexed, chunks_indexed, models_loaded = await _collect_health_status(request)
    is_ready = documents_indexed > 0 and models_loaded
    payload = HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        documents_indexed=documents_indexed,
        chunks_indexed=chunks_indexed,
        models_loaded=models_loaded,
    )
    if not is_ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    # Republish the eval job's Ragas scores as gauges on every scrape, before
    # rendering. This is deliberately synchronous: under BentoML the ASGI app
    # runs on a per-request event loop, so a background task started during a
    # request never runs again. Reading one small file per scrape is cheap and
    # makes the faithfulness alert gauge correct even before the first RAG
    # request (this endpoint has no controller dependency, so it does not
    # trigger lazy service initialization).
    from ..tracking.eval_bridge import refresh_eval_gauges

    status = refresh_eval_gauges(
        settings.eval_scores_path,
        settings.eval_scores_path.with_name("latest_drift.json"),
        settings.eval_live_scores_path,
    )

    body = generate_latest()
    return Response(
        content=body,
        media_type=CONTENT_TYPE_LATEST,
        # Surfaced so "the alert gauges are missing" can be told apart from
        # "the bridge never ran" without attaching a debugger. `abs_exists` is
        # the one that bites: BentoML serves from a worker whose CWD is the
        # bento's src dir, so a relative path resolves somewhere else entirely.
        headers={
            "X-Eval-Bridge": (
                f"scores={status['scores']} published={status['published']} "
                f"live={status['live']}"
            ),
            "X-Eval-Bridge-Path": f"{settings.eval_scores_path} exists={status['abs_exists']}",
        },
    )
