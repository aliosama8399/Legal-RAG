from typing import Protocol


class LLMProvider(Protocol):
    name: str
    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...
