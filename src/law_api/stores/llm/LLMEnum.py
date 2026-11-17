from enum import Enum


class LLMEnum(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    CLAUDE = "claude"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LOCAL = "local"
    MOCK = "mock"
