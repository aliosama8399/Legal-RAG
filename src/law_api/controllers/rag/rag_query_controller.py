from fastapi import APIRouter, HTTPException

from ...schemas import AskRequest, AskResponse, ErrorResponse, SearchResponse
from ...services import RAGQueryService


class RagQueryController:
    """Controller for retrieval and grounded-answer use cases."""

    def __init__(self, service: RAGQueryService) -> None:
        self.service = service
        self.router = APIRouter(prefix="/documents", tags=["rag"])
        self.router.add_api_route(
            "/{document_id}/search",
            self.search,
            methods=["POST"],
            response_model=SearchResponse,
            responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        )
        self.router.add_api_route(
            "/{document_id}/ask",
            self.ask,
            methods=["POST"],
            response_model=AskResponse,
            responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        )

    async def search(self, document_id: int, request: AskRequest) -> SearchResponse:
        results = self._run(lambda: self.service.search(document_id, request.question, request.top_k))
        return SearchResponse(document_id=document_id, query=request.question, results=results)

    async def ask(self, document_id: int, request: AskRequest) -> AskResponse:
        result = self._run(lambda: self.service.ask(document_id, request.question, request.top_k))
        return AskResponse(**result)

    @staticmethod
    def _run(call):
        try:
            return call()
        except ValueError as error:
            status_code = 404 if "not found" in str(error).lower() else 422
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail="RAG request failed") from error
