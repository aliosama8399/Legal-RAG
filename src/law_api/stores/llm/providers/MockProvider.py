"""Deterministic CPU-only mock LLM provider for CI, testing, and GPU-less environments."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from ..LLMInterface import LLMInterface

_CITATION_BLOCK_RE = re.compile(r"\[([^\]]+)\]\s*([^\n\[]+)")


class MockLLMProvider(LLMInterface):
    """Synthesizes a deterministic, citation-grounded answer directly from retrieved articles."""

    name = "mock"

    def __init__(self, model_id: str = "mock-legal-synthesizer") -> None:
        self.model_id = model_id
        self.last_usage: dict[str, int] | None = None

    def _synthesize(self, prompt: str, system: str | None = None) -> str:
        lowered_sys = (system or "").lower()
        lowered_prompt = (prompt or "").lower()
        if "strict grader" in lowered_sys or '"verdict"' in lowered_prompt:
            answer = '{"verdict": "yes", "reason": "The answer is supported by the cited Egyptian Civil Code articles."}'
        elif "strict json list" in lowered_prompt or '[{"question":' in lowered_prompt:
            answer = (
                '[{"question": "What legal rule does this article establish?", '
                '"answer": "It establishes the binding legal principle stated in the article."}]'
            )
        else:
            matches = _CITATION_BLOCK_RE.findall(prompt)
            if not matches:
                answer = (
                    "No relevant articles were found in the provided context to answer this question."
                )
            else:
                top_citation, top_text = matches[0]
                snippet = " ".join(top_text.strip().split())[:240]
                answer = (
                    f"Based on {top_citation}, {snippet} "
                    f"(Cited: {', '.join(c for c, _ in matches[:2])})."
                )
        self.last_usage = {
            "prompt": max(16, len(prompt.split())),
            "completion": max(12, len(answer.split())),
        }
        return answer

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return self._synthesize(prompt, system)

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        answer = self._synthesize(prompt, system)
        words = answer.split(" ")
        for idx, word in enumerate(words):
            yield word + (" " if idx < len(words) - 1 else "")

