from collections.abc import AsyncIterator

from ..LLMInterface import LLMInterface


def _usage_from(response) -> dict[str, int]:
    """Pull prompt/completion token counts off an OpenAI-style response.

    The OpenAI SDK and vLLM disagree on the exact attribute name across
    versions (`prompt_tokens` vs `input_tokens`), so both are accepted.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"prompt": 0, "completion": 0}

    def read(*names: str) -> int:
        for name in names:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    return {
        "prompt": read("prompt_tokens", "input_tokens"),
        "completion": read("completion_tokens", "output_tokens"),
    }


class OpenAIProvider(LLMInterface):
    """Async OpenAI-compatible chat client; the same class targets
    self-hosted OpenAI-compatible servers (vLLM) via ``base_url``.

    When the Langfuse SDK is importable and configured, the client is the
    official ``langfuse.openai.AsyncOpenAI`` wrapper: every call then becomes a
    ``generation`` observation with the model name and token usage attached
    automatically, instead of us hand-rolling those fields. It stays an
    automatic no-op when tracing is off, and falls back to the plain client if
    the import fails.
    """

    name = "openai"

    def __init__(self, model_id: str, api_key: str = "", base_url: str = "") -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        # Token counts of the most recent call, read by the caller for metrics.
        self.last_usage: dict[str, int] | None = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            client_class = AsyncOpenAI
            try:
                from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

                client_class = LangfuseAsyncOpenAI
            except Exception:
                pass
            self._client = client_class(
                api_key=self._api_key or "not-needed", base_url=self._base_url or None
            )
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
        self.last_usage = _usage_from(response)
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
        self.last_usage = None
        try:
            stream = await client.chat.completions.create(
                model=self.model_id,
                messages=self._messages(prompt, system),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                # vLLM only reports usage on a streamed response when asked.
                # Without this, streaming requests report zero tokens and the
                # Grafana cost/hour panel undercounts every /ask/stream call.
                stream_options={"include_usage": True},
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
            # The usage-only chunk arrives last and carries no choices.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = _usage_from(usage)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
