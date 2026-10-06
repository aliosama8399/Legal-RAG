"""Application lifecycle: shared resource initialization.

BentoML mounts the FastAPI app as a plain ASGI sub-app, so FastAPI's
lifespan never runs inside bento. All shared resources are therefore
initialized lazily and idempotently (first request) instead of in a
startup hook — safe under both plain uvicorn and bento.
"""

import asyncio
from time import perf_counter

from fastapi import FastAPI

from .config import Settings, settings as default_settings
from .services import DocumentIngestionService, RAGQueryService
from .stores.embeddings.EmbeddingProviderFactory import EmbeddingProviderFactory
from .stores.llm.LLMProviderFactory import LLMProviderFactory
from .stores.rerankers.RerankerProviderFactory import RerankerProviderFactory
from .stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from .tracking.langfuse_tracker import LangfuseTracker
from .tracking.mlflow_tracker import MLflowTracker


def _initialized(app: FastAPI) -> bool:
    # Deliberately a completion flag, not `hasattr(state, "rag_service")`. That
    # attribute is assigned before the last initialization steps, so a failure
    # late in the sequence left the app looking ready: the first request 500'd
    # and every later request skipped initialization entirely (silently
    # disabling the metrics bridge).
    return getattr(app.state, "rag_ready", False)


async def ensure_services(app: FastAPI) -> None:
    """Initialize stored + shared resources on first use if needed.

    Idempotent and thread/coroutine-safe via a per-app asyncio.Lock so
    concurrent first requests don't initialize twice.
    """
    if _initialized(app):
        return
    lock = getattr(app.state, "_init_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state._init_lock = lock
    async with lock:
        if not _initialized(app):
            await initialize_services(app)


async def initialize_services(
    app: FastAPI,
    settings: Settings | None = None,
    preconfig: dict | None = None,
) -> None:
    """Build and attach providers/services onto app.state."""
    preconfig = preconfig if preconfig is not None else getattr(app.state, "_preconfig", {})
    resolved = settings or preconfig.get("settings") or default_settings
    started = perf_counter()

    storage = preconfig.get("storage") or VectorDBProviderFactory.create(resolved)
    embedder = preconfig.get("embedder") or EmbeddingProviderFactory.create(resolved.embedding_name)
    llm = preconfig.get("llm") or LLMProviderFactory.create(
        resolved.llm_provider, resolved.llm_model, resolved.llm_api_key, resolved.llm_base_url
    )
    if "tracker" in preconfig:
        tracker = preconfig["tracker"]
    else:
        tracker = MLflowTracker(resolved.mlflow_tracking_uri, resolved.mlflow_experiment)
    reranker = preconfig.get("reranker")
    if reranker is None and resolved.rag_rerank_enabled and resolved.rag_reranker:
        reranker = RerankerProviderFactory.create(resolved.rag_reranker, resolved.rag_rerank_max_length)

    await storage.connect()

    state = app.state
    state.settings = resolved
    state.storage = storage
    state.embedder = embedder
    state.llm = llm
    state.reranker = reranker
    state.tracker = tracker
    state.langfuse_tracker = LangfuseTracker(
        resolved.langfuse_host,
        resolved.langfuse_public_key,
        resolved.langfuse_secret_key,
        resolved.langfuse_project_name,
    )
    state.ingestion_service = DocumentIngestionService(resolved, storage, embedder, tracker)
    state.rag_service = RAGQueryService(
        storage, embedder, llm,
        resolved.rag_top_k, resolved.llm_temperature, resolved.llm_max_tokens,
        langfuse=state.langfuse_tracker,
        reranker=reranker,
        rerank_candidates=resolved.rag_rerank_candidates,
        # Live evaluation judges with the same endpoint the batch eval uses, so
        # there is one judge to reason about rather than two.
        eval_judge_base_url=resolved.eval_judge_base_url,
        eval_judge_model=resolved.eval_judge_model,
        # Absolute, like the scores path: BentoML's worker CWD is not /app.
        live_eval_path=str(resolved.eval_live_scores_path),
    )
    # Set last: everything above must have succeeded for the app to count as
    # ready. The Ragas/drift gauges are NOT refreshed here — see
    # tracking/eval_bridge.py for why a background task cannot survive under
    # BentoML; /metrics republishes them synchronously on every scrape.
    state.rag_ready = True


async def shutdown_services(app: FastAPI) -> None:
    """Release connections — idempotent."""
    if not _initialized(app):
        return
    await app.state.storage.disconnect()
    app.state.langfuse_tracker.flush()
    tracker = app.state.tracker
    if tracker is not None:
        try:
            import mlflow as mlflow_module

            mlflow_module.end_run()
        except Exception:
            pass
