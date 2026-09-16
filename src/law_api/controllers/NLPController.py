from ..schemas import AskRequest, AskResponse, ChatHistoryResponse, SearchResponse
from ..services import RAGQueryService
from .BaseController import BaseController


class NLPController(BaseController):
    """Business logic for retrieval and grounded-answer use cases."""

    def __init__(self, service: RAGQueryService) -> None:
        super().__init__()
        self.service = service

    async def search(self, request: AskRequest) -> SearchResponse:
        results = await self.service.search(request.question, request.top_k, request.document_id)
        return SearchResponse(query=request.question, results=results)

    async def ask(self, request: AskRequest) -> AskResponse:
        result = await self.service.ask(request.question, request.top_k, request.document_id)
        return AskResponse(**result)

    async def history(self, document_id: int | None = None, limit: int = 50) -> ChatHistoryResponse:
        return ChatHistoryResponse(results=await self.service.history(document_id, limit))
