from .LLMEnum import LLMEnum
from .LLMInterface import LLMInterface
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    LocalTransformersProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
)

VLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class LLMProviderFactory:
    """Instantiate an LLM provider by name.

    Supports commercial APIs (OpenAI/ChatGPT, Gemini, Anthropic/Claude,
    OpenRouter, Groq, DeepSeek), self-hosted engines (vLLM, Ollama, local
    Transformers), and a deterministic CPU ``mock`` mode for CI.
    """

    @staticmethod
    def create(provider: str, model_id: str, api_key: str = "", base_url: str = "") -> LLMInterface:
        normalized = (provider or "").strip().lower()
        if normalized == LLMEnum.OPENAI.value:
            return OpenAIProvider(
                model_id=model_id or "gpt-4o-mini",
                api_key=api_key,
                base_url=base_url,
                provider_name="openai",
            )
        if normalized == LLMEnum.VLLM.value:
            return OpenAIProvider(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url or VLLM_DEFAULT_BASE_URL,
                provider_name="vllm",
            )
        if normalized == LLMEnum.OPENROUTER.value:
            return OpenAIProvider(
                model_id=model_id or "openai/gpt-4o-mini",
                api_key=api_key,
                base_url=base_url or OPENROUTER_DEFAULT_BASE_URL,
                provider_name="openrouter",
            )
        if normalized == LLMEnum.GROQ.value:
            return OpenAIProvider(
                model_id=model_id or "llama-3.3-70b-versatile",
                api_key=api_key,
                base_url=base_url or GROQ_DEFAULT_BASE_URL,
                provider_name="groq",
            )
        if normalized == LLMEnum.DEEPSEEK.value:
            return OpenAIProvider(
                model_id=model_id or "deepseek-chat",
                api_key=api_key,
                base_url=base_url or DEEPSEEK_DEFAULT_BASE_URL,
                provider_name="deepseek",
            )
        if normalized in (LLMEnum.GEMINI.value, LLMEnum.GOOGLE.value):
            return GeminiProvider(
                model_id=model_id or "gemini-2.0-flash",
                api_key=api_key,
                base_url=base_url,
            )
        if normalized in (LLMEnum.ANTHROPIC.value, LLMEnum.CLAUDE.value):
            return AnthropicProvider(
                model_id=model_id or "claude-3-5-sonnet-latest",
                api_key=api_key,
                base_url=base_url,
            )
        if normalized == LLMEnum.OLLAMA.value:
            return OllamaProvider(model_id=model_id, base_url=base_url)
        if normalized == LLMEnum.LOCAL.value:
            return LocalTransformersProvider(model_id=model_id)
        if normalized == LLMEnum.MOCK.value:
            return MockLLMProvider(model_id=model_id or "mock-legal-synthesizer")
        choices = ", ".join(f"'{member.value}'" for member in LLMEnum)
        raise ValueError(f"LAW_API_LLM_PROVIDER must be one of: {choices}")
