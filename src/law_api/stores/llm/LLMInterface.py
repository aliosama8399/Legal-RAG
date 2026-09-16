from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    Every LLM backend (OpenAI, Ollama, local transformers) must inherit
    from this class and implement ``generate``.
    """

    name: str
    model_id: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...
