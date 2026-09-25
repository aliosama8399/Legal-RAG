from abc import ABC, abstractmethod


class RerankerInterface(ABC):
    """Abstract cross-encoder reranker.

    Implementation scores query↔document pairs; higher score = more relevant.
    """

    name: str
    model_id: str

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...
