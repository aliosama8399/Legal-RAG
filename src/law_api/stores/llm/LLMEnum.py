from enum import Enum


class LLMEnum(Enum):
    OPENAI = "openai"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LOCAL = "local"
