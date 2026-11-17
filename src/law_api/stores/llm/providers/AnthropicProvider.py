"""Anthropic Claude Messages API provider implementing LLMInterface via httpx."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from ..LLMInterface import LLMInterface

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


def _resolve_anthropic_api_key(explicit: str) -> str:
    return (
        explicit
        or os.getenv("LAW_API_LLM_API_KEY", "")
        or os.getenv("ANTHROPIC_API_KEY", "")
    )


class AnthropicProvider(LLMInterface):
    """Async Anthropic Claude client via the Messages REST API."""

    name = "anthropic"

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-latest",
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        self.model_id = model_id or "claude-3-5-sonnet-latest"
        self._api_key = _resolve_anthropic_api_key(api_key)
        normalized = (base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
        self._base_url = normalized.removesuffix("/v1")
        self.last_usage: dict[str, int] | None = None

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ValueError(
                "Anthropic API key is required: set LAW_API_LLM_API_KEY or ANTHROPIC_API_KEY"
            )
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        *,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if system:
            payload["system"] = system
        return payload

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.last_usage = None
        payload = self._build_payload(prompt, system, temperature, max_tokens, stream=False)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=120.0) as client:
            try:
                response = await client.post(
                    "/v1/messages",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                detail = error.response.text.strip() or error.response.reason_phrase
                raise RuntimeError(
                    f"Anthropic request failed ({error.response.status_code}): {detail}"
                ) from error
            except httpx.TransportError as error:
                raise RuntimeError(
                    f"Cannot reach Anthropic API at '{self._base_url}': {error}"
                ) from error

        data = response.json()
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            self.last_usage = {
                "prompt": int(usage.get("input_tokens") or 0),
                "completion": int(usage.get("output_tokens") or 0),
            }
        blocks = data.get("content") or []
        return "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        self.last_usage = None
        payload = self._build_payload(prompt, system, temperature, max_tokens, stream=True)
        prompt_tokens = 0
        completion_tokens = 0

        async with (
            httpx.AsyncClient(base_url=self._base_url, timeout=120.0) as client,
            client.stream(
                "POST",
                "/v1/messages",
                headers=self._headers(),
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "message_start":
                    msg_usage = (event.get("message") or {}).get("usage") or {}
                    prompt_tokens = int(msg_usage.get("input_tokens") or prompt_tokens)
                elif event_type == "message_delta":
                    delta_usage = event.get("usage") or {}
                    completion_tokens = int(delta_usage.get("output_tokens") or completion_tokens)
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text", "")
                    if text:
                        yield str(text)
                if prompt_tokens or completion_tokens:
                    self.last_usage = {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                    }
