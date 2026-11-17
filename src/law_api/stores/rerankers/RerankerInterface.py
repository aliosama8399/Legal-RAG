from abc import ABC, abstractmethod


class RerankerInterface(ABC):
    """Abstract cross-encoder reranker.

    Implementation scores query↔document pairs; higher score = more relevant.
    """

    name: str
    model_id: str

    @property
    def is_loaded(self) -> bool:
        return True

    async def warmup(self) -> None:
        """Optionally preload reranker weights at startup."""
        return None

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...
