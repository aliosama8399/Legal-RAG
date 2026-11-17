import logging
from typing import Annotated

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..auth.deps import get_tenant_context, require_role
from ..auth.models import TenantContext
from ..controllers.NLPController import NLPController
from ..deps import get_nlp_controller
from ..exceptions import AppError, ResourceNotFoundError
from ..schemas import (
    AskRequest,
    AskResponse,
    ChatHistoryResponse,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessageRequest,
    ConversationSummary,
    ConversationTurnResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    SearchResponse,
)

logger = logging.getLogger(__name__)

rag_router = APIRouter(prefix="/api/v1", tags=["rag"])


async def _map_errors(call):
    try:
        return await call()
    except AppError:
        raise
    except ValueError as error:
        status_code = 404 if "not found" in str(error).lower() else 422
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@rag_router.post(
    "/search",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def search(
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> SearchResponse:
    """Tenant-isolated hybrid semantic + lexical search over embedded law chunks."""
    return await _map_errors(lambda: controller.search(request, tenant_id=ctx.tenant_id))


@rag_router.post(
    "/ask",
    response_model=AskResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def ask(
    http_request: Request,
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> AskResponse:
    """Retrieval-grounded answer with article citations and tenant usage metering."""
    request_id = getattr(http_request.state, "request_id", "")
    return await _map_errors(
        lambda: controller.ask(
            request, tenant_id=ctx.tenant_id, user_id=ctx.user_id, request_id=request_id
        )
    )


@rag_router.post(
    "/ask/stream",
    response_class=StreamingResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def ask_stream(
    http_request: Request,
    request: AskRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> StreamingResponse:
    """Streaming retrieval-grounded answer (Server-Sent Events)."""
    request_id = getattr(http_request.state, "request_id", "")

    async def event_stream():
        try:
            async for event in controller.ask_stream(
                request, tenant_id=ctx.tenant_id, user_id=ctx.user_id, request_id=request_id
            ):
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
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    document_id: int | None = None,
    limit: int = 50,
) -> ChatHistoryResponse:
    """Recent chat history scoped to the current tenant."""
    return await _map_errors(
        lambda: controller.history(document_id, limit, tenant_id=ctx.tenant_id)
    )


# ---- Multi-turn Conversations & Feedback ----


@rag_router.post(
    "/conversations",
    response_model=ConversationSummary,
    tags=["conversations"],
    status_code=201,
)
async def create_conversation(
    http_request: Request,
    body: ConversationCreateRequest,
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
) -> ConversationSummary:
    """Create a stateful multi-turn legal research conversation thread."""
    conv = http_request.app.state.tenant_store.create_conversation(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        title=body.title,
        document_id=body.document_id,
    )
    return ConversationSummary(**conv, message_count=0)


@rag_router.get(
    "/conversations",
    response_model=ConversationListResponse,
    tags=["conversations"],
)
async def list_conversations(
    http_request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    limit: int = 50,
) -> ConversationListResponse:
    """List tenant-scoped conversation threads."""
    rows = http_request.app.state.tenant_store.list_conversations(ctx.tenant_id, limit=limit)
    return ConversationListResponse(results=rows)


@rag_router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    tags=["conversations"],
    responses={404: {"model": ErrorResponse}},
)
async def get_conversation(
    http_request: Request,
    conversation_id: str,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> ConversationDetailResponse:
    """Fetch a conversation thread and its complete message/citation history."""
    conv = http_request.app.state.tenant_store.get_conversation(ctx.tenant_id, conversation_id)
    if conv is None:
        raise ResourceNotFoundError(f"Conversation '{conversation_id}' was not found")
    return ConversationDetailResponse(**conv)


@rag_router.post(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
    tags=["conversations"],
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def post_conversation_message(
    http_request: Request,
    conversation_id: str,
    body: ConversationMessageRequest,
    controller: Annotated[NLPController, Depends(get_nlp_controller)],
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
) -> ConversationTurnResponse:
    """Send a question within a conversation thread with multi-turn context preservation."""
    request_id = getattr(http_request.state, "request_id", "")
    result = await _map_errors(
        lambda: controller.service.ask_in_conversation(
            conversation_id=conversation_id,
            question=body.question,
            top_k=body.top_k,
            document_id=body.document_id,
            evaluate=body.evaluate,
            expected_article=body.expected_article,
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            request_id=request_id,
        )
    )
    return ConversationTurnResponse(**result)


@rag_router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    tags=["conversations"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_conversation(
    http_request: Request,
    conversation_id: str,
    ctx: Annotated[TenantContext, Depends(require_role("attorney", "admin", "owner"))],
) -> None:
    """Delete a conversation thread."""
    deleted = http_request.app.state.tenant_store.delete_conversation(
        ctx.tenant_id, conversation_id
    )
    if not deleted:
        raise ResourceNotFoundError(f"Conversation '{conversation_id}' was not found")


@rag_router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["feedback"],
    status_code=201,
)
async def submit_feedback(
    http_request: Request,
    body: FeedbackRequest,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> FeedbackResponse:
    """Record user feedback (thumbs up/down) and forward score to Langfuse if enabled."""
    store = http_request.app.state.tenant_store
    saved = store.record_feedback(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        trace_id=body.trace_id,
        score=body.score,
        message_id=body.message_id,
        comment=body.comment,
    )
    langfuse = getattr(http_request.app.state, "langfuse_tracker", None)
    client = getattr(langfuse, "_client", None) if langfuse is not None else None
    if client is not None and body.trace_id:
        try:
            client.create_score(
                trace_id=body.trace_id,
                name="user_feedback",
                value=float(body.score),
                comment=body.comment or None,
            )
            langfuse.flush()
        except Exception:
            pass
    store.record_audit_log(
        tenant_id=ctx.tenant_id,
        actor_id=ctx.user_id,
        action="feedback.submitted",
        resource_type="trace",
        resource_id=body.trace_id,
        metadata={"score": body.score, "message_id": body.message_id},
        ip_address=http_request.client.host if http_request.client else "",
        request_id=getattr(http_request.state, "request_id", ""),
    )
    return FeedbackResponse(**saved)
