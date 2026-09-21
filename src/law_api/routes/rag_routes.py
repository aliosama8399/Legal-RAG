import logging
from typing import Annotated

import orjson
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..controllers.NLPController import NLPController
from ..deps import get_nlp_controller
from ..schemas import AskRequest, AskResponse, ChatHistoryResponse, ErrorResponse, SearchResponse

logger = logging.getLogger(__name__)

rag_router = APIRouter(prefix="/api/v1", tags=["rag"])


async def _map_errors(call):
    try:
        return await call()
    except ValueError as error:
        status_code = 404 if "not found" in str(error).lower() else 422
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except Exception as error:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@rag_router.post(
    "/search",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def search(
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
) -> SearchResponse:
    """Semantic search over embedded law chunks."""
    return await _map_errors(lambda: controller.search(request))


@rag_router.post(
    "/ask",
    response_model=AskResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def ask(
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
) -> AskResponse:
    """Retrieval-grounded answer with article citations."""
    return await _map_errors(lambda: controller.ask(request))


@rag_router.post(
    "/ask/stream",
    response_class=StreamingResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def ask_stream(
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
) -> StreamingResponse:
    """Streaming retrieval-grounded answer (Server-Sent Events).

    Events: ``sources`` (retrieved chunks) -> ``token`` (answer deltas)
    -> ``done`` (final answer + citations).
    """

    async def event_stream():
        try:
            async for event in controller.ask_stream(request):
                yield f"data: {orjson.dumps(event).decode('utf-8')}\n\n"
        except ValueError as error:
            detail = {"type": "error", "detail": str(error)}
            yield f"data: {orjson.dumps(detail).decode('utf-8')}\n\n"
        except Exception:
            logger.exception("Streaming RAG request failed")
            detail = {"type": "error", "detail": "RAG request failed"}
            yield f"data: {orjson.dumps(detail).decode('utf-8')}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@rag_router.get(
    "/history",
    response_model=ChatHistoryResponse,
    responses={422: {"model": ErrorResponse}},
)
async def history(
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
    document_id: int | None = None,
    limit: int = 50,
) -> ChatHistoryResponse:
    """Recent chat history, optionally filtered by document."""
    return await _map_errors(lambda: controller.history(document_id, limit))
