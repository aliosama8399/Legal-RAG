from fastapi import APIRouter, Response

from ..config import settings
from ..schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


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
