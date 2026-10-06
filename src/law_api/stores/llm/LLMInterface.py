from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    Every LLM backend (OpenAI, vLLM, Ollama, local transformers) must
    inherit from this class and implement ``generate`` and
    ``generate_stream``.
    """

    name: str
    model_id: str
    # Token counts of the most recent generate()/generate_stream() call, as
    # {"prompt": int, "completion": int}. Providers that cannot report usage
    # leave it as None; callers must treat that as "unknown", not zero.
    last_usage: dict[str, int] | None = None

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Yield answer tokens incrementally."""
        yield ""  # pragma: no cover — abstract
