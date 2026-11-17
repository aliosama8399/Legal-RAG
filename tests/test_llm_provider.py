"""Tests for multi-provider LLM factory, REST adapters, pricing, and provider-level evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from eval import dataset as eval_dataset
from law_api.guardrails.live_eval import evaluate_live
from law_api.stores.llm.LLMProviderFactory import LLMProviderFactory
from law_api.stores.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
)
from law_api.tracking.prometheus_metrics import resolve_model_pricing


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_factory_creates_ollama_provider():
    provider = LLMProviderFactory.create("ollama", "llama3.1")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_id == "llama3.1"
    assert provider._base_url == "http://127.0.0.1:11434"


def test_factory_vllm_uses_openai_compatible_client():
    provider = LLMProviderFactory.create("vllm", "Qwen/Qwen2.5-1.5B-Instruct")
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "vllm"
    assert provider._base_url == "http://127.0.0.1:8000/v1"


def test_factory_creates_commercial_and_mock_providers():
    chatgpt = LLMProviderFactory.create("openai", "gpt-4o-mini", api_key="sk-test")
    assert isinstance(chatgpt, OpenAIProvider)
    assert chatgpt.name == "openai"
    assert chatgpt.model_id == "gpt-4o-mini"

    gemini = LLMProviderFactory.create("gemini", "gemini-2.0-flash", api_key="gem-key")
    assert isinstance(gemini, GeminiProvider)
    assert gemini.name == "gemini"
    assert LLMProviderFactory.create("google", "gemini-2.5-pro", api_key="g-key").name == "gemini"

    claude = LLMProviderFactory.create("claude", "claude-3-5-sonnet-latest", api_key="ant-key")
    assert isinstance(claude, AnthropicProvider)
    assert claude.name == "anthropic"
    assert LLMProviderFactory.create("anthropic", "", api_key="a-key").model_id == "claude-3-5-sonnet-latest"

    openrouter = LLMProviderFactory.create("openrouter", "openai/gpt-4o", api_key="or-key")
    assert isinstance(openrouter, OpenAIProvider)
    assert openrouter.name == "openrouter"
    assert "openrouter.ai" in openrouter._base_url

    groq = LLMProviderFactory.create("groq", "llama-3.3-70b-versatile", api_key="grq-key")
    assert isinstance(groq, OpenAIProvider)
    assert groq.name == "groq"

    deepseek = LLMProviderFactory.create("deepseek", "deepseek-chat", api_key="ds-key")
    assert isinstance(deepseek, OpenAIProvider)
    assert deepseek.name == "deepseek"

    mock = LLMProviderFactory.create("mock", "")
    assert isinstance(mock, MockLLMProvider)
    assert mock.model_id == "mock-legal-synthesizer"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="LAW_API_LLM_PROVIDER"):
        LLMProviderFactory.create("nope", "x")


@pytest.mark.asyncio
async def test_ollama_generate_parses_chat_response(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "Article 147 says the vendor has a privilege."}}

    async def fake_post(self, url, json=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    provider = OllamaProvider("llama3.1")
    answer = await provider.generate(
        "Question: what is a privilege?",
        system="You are a legal assistant.",
        temperature=0.2,
        max_tokens=256,
    )

    assert answer == "Article 147 says the vendor has a privilege."
    assert captured["url"] == "/api/chat"
    body = captured["json"]
    assert body["model"] == "llama3.1"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_predict": 256}
    assert body["messages"] == [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Question: what is a privilege?"},
    ]


@pytest.mark.asyncio
async def test_ollama_generate_without_system_prompt(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    async def fake_post(self, url, json=None):
        del url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    provider = OllamaProvider("llama3.1")
    assert await provider.generate("hi") == "ok"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_gemini_generate_and_stream(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "Based on Article 147, the contract binds parties."}]}}
                ],
                "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 15},
            }

    async def fake_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    def fake_stream(self, method, url, headers=None, json=None):
        del self, method, headers, json
        captured["stream_url"] = url
        lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Based on "}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"Article 147."}]}}],"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":9}}',
        ]
        return _FakeStreamResponse(lines)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    monkeypatch.setattr("httpx.AsyncClient.stream", fake_stream)

    provider = GeminiProvider("gemini-2.0-flash", api_key="gem-secret")
    reply = await provider.generate("Explain Art 147", system="Be concise.", temperature=0.1, max_tokens=256)
    assert reply == "Based on Article 147, the contract binds parties."
    assert captured["url"] == "/models/gemini-2.0-flash:generateContent"
    assert captured["headers"]["x-goog-api-key"] == "gem-secret"
    assert captured["json"]["systemInstruction"] == {"parts": [{"text": "Be concise."}]}
    assert provider.last_usage == {"prompt": 42, "completion": 15}

    streamed = [chunk async for chunk in provider.generate_stream("Explain Art 147")]
    assert "".join(streamed) == "Based on Article 147."
    assert captured["stream_url"] == "/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
    assert provider.last_usage == {"prompt": 40, "completion": 9}


@pytest.mark.asyncio
async def test_anthropic_generate_and_stream(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "content": [{"type": "text", "text": "Contract makes law of parties (Article 147)."}],
                "usage": {"input_tokens": 31, "output_tokens": 12},
            }

    async def fake_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    def fake_stream(self, method, url, headers=None, json=None):
        del self, method, headers, json
        captured["stream_url"] = url
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":28}}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Contract binds "}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"parties."}}',
            'data: {"type":"message_delta","usage":{"output_tokens":7}}',
        ]
        return _FakeStreamResponse(lines)

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
    monkeypatch.setattr("httpx.AsyncClient.stream", fake_stream)

    provider = AnthropicProvider("claude-3-5-sonnet-latest", api_key="ant-secret")
    reply = await provider.generate("Explain Art 147", system="Be strict.", temperature=0.0, max_tokens=200)
    assert reply == "Contract makes law of parties (Article 147)."
    assert captured["url"] == "/v1/messages"
    assert captured["headers"]["x-api-key"] == "ant-secret"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["system"] == "Be strict."
    assert provider.last_usage == {"prompt": 31, "completion": 12}

    streamed = [chunk async for chunk in provider.generate_stream("Explain Art 147")]
    assert "".join(streamed) == "Contract binds parties."
    assert captured["stream_url"] == "/v1/messages"
    assert provider.last_usage == {"prompt": 28, "completion": 7}


@pytest.mark.asyncio
async def test_live_eval_works_across_providers(monkeypatch):
    results = [
        {"citation": "Egyptian Civil Code, Article 147", "article_number": 147, "chunk_text": "Contract law."}
    ]

    mock_payload = await evaluate_live(
        question="What binds parties?",
        answer="Article 147 binds parties.",
        results=results,
        top_k=5,
        expected_article=147,
        metrics=("faithfulness", "answer_relevancy"),
        judge_provider="mock",
        judge_model="mock-judge",
    )
    assert mock_payload["judge_provider"] == "mock"
    assert mock_payload["judge_model"] == "mock-judge"
    assert mock_payload["hit_at_k"] == 1.0
    assert mock_payload["faithfulness"] == 1.0
    assert mock_payload["answer_relevancy"] == 1.0

    class FakeHttpResp:
        def __init__(self, body: dict) -> None:
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    async def fake_judge_post(self, url, headers=None, json=None):
        del self, headers, json
        if ":generateContent" in url:
            return FakeHttpResp(
                {"candidates": [{"content": {"parts": [{"text": '{"verdict":"yes","reason":"grounded"}'}]}}]}
            )
        return FakeHttpResp(
            {"content": [{"type": "text", "text": '{"verdict":"yes","reason":"relevant"}'}]}
        )

    monkeypatch.setattr("httpx.AsyncClient.post", fake_judge_post)

    gem_payload = await evaluate_live(
        question="Q",
        answer="A",
        results=results,
        top_k=5,
        expected_article=None,
        metrics=("faithfulness",),
        judge_provider="gemini",
        judge_model="gemini-2.0-flash",
        judge_api_key="g-key",
    )
    assert gem_payload["judge_provider"] == "gemini"
    assert gem_payload["faithfulness"] == 1.0

    claude_payload = await evaluate_live(
        question="Q",
        answer="A",
        results=results,
        top_k=5,
        expected_article=None,
        metrics=("answer_relevancy",),
        judge_provider="claude",
        judge_model="claude-3-5-sonnet-latest",
        judge_api_key="c-key",
    )
    assert claude_payload["judge_provider"] == "claude"
    assert claude_payload["answer_relevancy"] == 1.0


def test_dataset_generate_questions_uses_configured_provider(tmp_path, monkeypatch):
    articles_file = tmp_path / "articles.jsonl"
    out_file = tmp_path / "candidates.jsonl"
    articles_file.write_text(
        json.dumps(
            {
                "article_number": 147,
                "text_en": "The contract makes the law of the parties.",
                "citation": "Egyptian Civil Code, Article 147",
                "is_repealed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_dataset, "CANDIDATES_PATH", out_file)
    monkeypatch.setattr(
        eval_dataset,
        "settings",
        replace(eval_dataset.settings, eval_judge_provider="mock", eval_judge_model="mock-gen"),
    )

    items = eval_dataset.generate_questions(articles_file, count=1, per_article=1, which="judge")
    assert len(items) == 1
    assert items[0]["article_number"] == 147
    assert out_file.exists()


def test_model_pricing_lookup():
    assert resolve_model_pricing("gpt-4o-mini") == (0.15, 0.60)
    assert resolve_model_pricing("gemini-2.0-flash") == (0.10, 0.40)
    assert resolve_model_pricing("claude-3-5-sonnet-latest") == (3.00, 15.00)
    assert resolve_model_pricing("Qwen/Qwen2.5-1.5B-Instruct") == (0.0, 0.0)


@pytest.mark.asyncio
async def test_live_eval_infers_retrieval_metrics_and_continuous_bilingual_scores():
    results = [
        {
            "citation": "Egyptian Civil Code, Article 543",
            "article_number": 543,
            "chunk_text": "A loan ends upon expiration of the term.",
            "text_en": "A loan ends upon expiration of the term.",
        },
        {
            "citation": "Egyptian Civil Code, Article 558",
            "article_number": 558,
            "chunk_text": "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع بشيء معين.",
            "text_en": "A lease is a contract by which the lessor undertakes to enable the lessee to enjoy a specific thing.",
        },
    ]

    class _StrictJudge:
        name = "openai"
        model_id = "gpt-4o-mini"

        async def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=200):
            del system, temperature, max_tokens
            assert "A lease is a contract" in prompt
            return '{"score": 0.88, "verdict": "yes", "reason": "Supported by Article 558."}'

    payload = await evaluate_live(
        question="What is a lease contract under Egyptian law?",
        answer="Based on Article 558, a lease is a contract enabling the lessee to enjoy a specific thing.",
        results=results,
        top_k=5,
        expected_article=None,
        metrics=("faithfulness", "answer_relevancy"),
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        judge_llm=_StrictJudge(),
    )
    assert payload["hit_at_k"] == 1.0
    assert payload["reciprocal_rank"] == 0.5
    assert 0.7 < payload["faithfulness"] < 1.0
    assert 0.6 < payload["answer_relevancy"] < 1.0
