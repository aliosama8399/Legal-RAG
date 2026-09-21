"""Tests for LLM provider selection and response parsing (HTTP is mocked)."""

import pytest

from law_api.stores.llm.LLMProviderFactory import LLMProviderFactory
from law_api.stores.llm.providers import OllamaProvider


def test_factory_creates_ollama_provider():
    provider = LLMProviderFactory.create("ollama", "llama3.1")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_id == "llama3.1"
    assert provider._base_url == "http://127.0.0.1:11434"


def test_factory_vllm_uses_openai_compatible_client():
    from law_api.stores.llm.providers import OpenAIProvider

    provider = LLMProviderFactory.create("vllm", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    assert isinstance(provider, OpenAIProvider)
    assert provider._base_url == "http://127.0.0.1:8000/v1"


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
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)

    provider = OllamaProvider("llama3.1")
    assert await provider.generate("hi") == "ok"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
