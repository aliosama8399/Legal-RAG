from enum import Enum


class LLMEnum(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCAL = "local"
