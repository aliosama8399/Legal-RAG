from .base import LLMProvider
from .local_transformers_provider import LocalTransformersProvider
from .openai_provider import OpenAIProvider


def create_llm_provider(provider: str, model_id: str, api_key: str = "", base_url: str = "") -> LLMProvider:
    if provider == "openai":
        return OpenAIProvider(model_id=model_id, api_key=api_key, base_url=base_url)
    if provider == "local":
        return LocalTransformersProvider(model_id=model_id)
    raise ValueError("LAW_API_LLM_PROVIDER must be 'openai' or 'local'")
