from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    """Abstract base class for embedding providers."""

    name: str
    model_id: str

    @property
    def is_loaded(self) -> bool:
        return True

    async def warmup(self) -> None:
        """Optionally preload model weights at startup."""
        return None

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[list[float]]: ...
