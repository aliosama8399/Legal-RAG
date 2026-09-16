from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    """Abstract base class for embedding providers."""

    name: str
    model_id: str

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]: ...
