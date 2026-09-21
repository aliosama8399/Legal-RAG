from collections.abc import AsyncIterator

from ..LLMInterface import LLMInterface


class OpenAIProvider(LLMInterface):
    """Async OpenAI-compatible chat client; the same class targets
    self-hosted OpenAI-compatible servers (vLLM) via ``base_url``."""

    name = "openai"

    def __init__(self, model_id: str, api_key: str = "", base_url: str = "") -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key or "not-needed", base_url=self._base_url or None)
        return self._client

    def _messages(self, prompt: str, system: str | None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        client = self._ensure_client()
        try:
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=self._messages(prompt, system),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            import httpx
            from openai import APIConnectionError

            if isinstance(error, (APIConnectionError, httpx.ConnectError, httpx.TransportError)):
                raise RuntimeError(
                    f"Cannot reach the LLM server at '{self._base_url}' — is the container up "
                    "and the model loaded? Check: docker compose ps vllm && docker compose logs vllm "
                    "(vLLM only accepts connections after the model finishes loading)"
                ) from error
            raise
        return response.choices[0].message.content or ""

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model_id,
                messages=self._messages(prompt, system),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as error:
            import httpx
            from openai import APIConnectionError

            if isinstance(error, (APIConnectionError, httpx.ConnectError, httpx.TransportError)):
                raise RuntimeError(
                    f"Cannot reach the LLM server at '{self._base_url}' — is the container up "
                    "and the model loaded? Check: docker compose ps vllm && docker compose logs vllm "
                    "(vLLM only accepts connections after the model finishes loading)"
                ) from error
            raise
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
