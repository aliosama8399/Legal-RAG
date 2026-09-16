import asyncio

import httpx

from ..LLMInterface import LLMInterface

MAX_ATTEMPTS = 3


class OllamaProvider(LLMInterface):
    """Async client for a local/remote Ollama server REST API
    (default http://127.0.0.1:11434).

    Retries transient disconnections: Ollama drops the first connections
    while it loads a model into memory.
    """

    name = "ollama"

    def __init__(self, model_id: str, base_url: str = "") -> None:
        self.model_id = model_id
        self._base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=300.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except httpx.HTTPStatusError as error:
                # Non-retryable: e.g. 404 when the model is not pulled.
                detail = error.response.text.strip() or error.response.reason_phrase
                raise ValueError(
                    f"Ollama request failed with status {error.response.status_code}: {detail}. "
                    f"If the model is missing, run: ollama pull {self.model_id}"
                ) from error
            except httpx.TransportError as error:
                last_error = error
                if attempt < MAX_ATTEMPTS - 1:
                    # Ollama may disconnect while loading the model — back off and retry.
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(
            f"Ollama server disconnected after {MAX_ATTEMPTS} attempts: {last_error}. "
            "Is Ollama running (ollama serve) and the model pulled (ollama pull "
            f"{self.model_id})?"
        ) from last_error
