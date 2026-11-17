"""Google Gemini REST API provider implementing LLMInterface via httpx."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from ..LLMInterface import LLMInterface

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _resolve_gemini_api_key(explicit: str) -> str:
    return (
        explicit
        or os.getenv("LAW_API_LLM_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))


class GeminiProvider(LLMInterface):
    """Async Google Gemini client supporting both single-shot generation and SSE streaming."""

    name = "gemini"

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        self.model_id = model_id or "gemini-2.0-flash"
        self._api_key = _resolve_gemini_api_key(api_key)
        self._base_url = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
        self.last_usage: dict[str, int] | None = None

    def _model_path(self) -> str:
        return self.model_id.removeprefix("models/")

    @staticmethod
    def _build_payload(
        prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> dict:
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ValueError(
                "Gemini API key is required: set LAW_API_LLM_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY"
            )
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
        }

    def _record_usage(self, payload: dict) -> None:
        usage = payload.get("usageMetadata")
        if isinstance(usage, dict):
            self.last_usage = {
                "prompt": int(usage.get("promptTokenCount") or 0),
                "completion": int(usage.get("candidatesTokenCount") or 0),
            }

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.last_usage = None
        payload = self._build_payload(prompt, system, temperature, max_tokens)
        async with httpx.AsyncClient(base_url=self._base_url, timeout=120.0) as client:
            try:
                response = await client.post(
                    f"/models/{self._model_path()}:generateContent",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                detail = error.response.text.strip() or error.response.reason_phrase
                raise RuntimeError(
                    f"Gemini request failed ({error.response.status_code}): {detail}"
                ) from error
            except httpx.TransportError as error:
                raise RuntimeError(
                    f"Cannot reach Gemini API at '{self._base_url}': {error}"
                ) from error
        data = response.json()
        self._record_usage(data)
        return _extract_gemini_text(data)

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        self.last_usage = None
        payload = self._build_payload(prompt, system, temperature, max_tokens)
        async with (
            httpx.AsyncClient(base_url=self._base_url, timeout=120.0) as client,
            client.stream(
                "POST",
                f"/models/{self._model_path()}:streamGenerateContent?alt=sse",
                headers=self._headers(),
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items = chunk if isinstance(chunk, list) else [chunk]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    self._record_usage(item)
                    text = _extract_gemini_text(item)
                    if text:
                        yield text
