from .LLMEnum import LLMEnum
from .LLMInterface import LLMInterface
from .providers import LocalTransformersProvider, OllamaProvider, OpenAIProvider


class LLMProviderFactory:
    """Instantiate an LLM provider by name."""

    @staticmethod
    def create(provider: str, model_id: str, api_key: str = "", base_url: str = "") -> LLMInterface:
        if provider == LLMEnum.OPENAI.value:
            return OpenAIProvider(model_id=model_id, api_key=api_key, base_url=base_url)
        if provider == LLMEnum.OLLAMA.value:
            return OllamaProvider(model_id=model_id, base_url=base_url)
        if provider == LLMEnum.LOCAL.value:
            return LocalTransformersProvider(model_id=model_id)
        choices = ", ".join(f"'{member.value}'" for member in LLMEnum)
        raise ValueError(f"LAW_API_LLM_PROVIDER must be one of: {choices}")
