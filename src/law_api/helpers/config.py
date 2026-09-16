import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "providers.yaml"
load_dotenv(PROJECT_ROOT / ".env")


def _load_provider_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


_yaml_config = _load_provider_config()


def _yaml_value(*keys: str, default: str = "") -> str:
    value: object = _yaml_config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return str(value) if value is not None else default


def _env_or_yaml(environment_name: str, *keys: str, default: str = "") -> str:
    return os.getenv(environment_name, _yaml_value(*keys, default=default))


@dataclass(frozen=True)
class Settings:
    upload_dir: Path = Path(_env_or_yaml("LAW_API_UPLOAD_DIR", "paths", "upload_dir", default="storage/uploads"))
    pending_dir: Path = Path(_env_or_yaml("LAW_API_PENDING_DIR", "paths", "pending_dir", default="storage/pending"))
    storage_provider: str = _env_or_yaml("LAW_API_STORAGE_PROVIDER", "storage", "provider", default="qdrant-local")
    postgres_dsn: str = os.getenv("LAW_API_POSTGRES_DSN", "")
    qdrant_url: str = _env_or_yaml("LAW_API_QDRANT_URL", "storage", "qdrant", "url")
    qdrant_path: Path = Path(_env_or_yaml("LAW_API_QDRANT_PATH", "storage", "qdrant", "path", default="storage/qdrant"))
    qdrant_collection: str = _env_or_yaml("LAW_API_QDRANT_COLLECTION", "storage", "qdrant", "collection", default="law_chunks")
    embedding_name: str = _env_or_yaml("LAW_API_EMBEDDING", "embedding", "name", default="bge-m3")
    mlflow_tracking_uri: str = _env_or_yaml("MLFLOW_TRACKING_URI", "mlflow", "tracking_uri", default="http://127.0.0.1:5001")
    mlflow_experiment: str = _env_or_yaml("MLFLOW_EXPERIMENT", "mlflow", "experiment", default="legal-rag-embeddings")
    llm_provider: str = _env_or_yaml("LAW_API_LLM_PROVIDER", "llm", "provider", default="openai")
    llm_model: str = _env_or_yaml("LAW_API_LLM_MODEL", "llm", "model", default="gpt-4o-mini")
    llm_base_url: str = _env_or_yaml("LAW_API_LLM_BASE_URL", "llm", "base_url")
    llm_api_key: str = os.getenv("LAW_API_LLM_API_KEY", "")
    llm_temperature: float = float(_env_or_yaml("LAW_API_LLM_TEMPERATURE", "llm", "temperature", default="0.0"))
    llm_max_tokens: int = int(_env_or_yaml("LAW_API_LLM_MAX_TOKENS", "llm", "max_tokens", default="512"))
    rag_top_k: int = int(_env_or_yaml("LAW_API_RAG_TOP_K", "rag", "top_k", default="5"))
    max_upload_bytes: int = int(_env_or_yaml("LAW_API_MAX_UPLOAD_BYTES", "paths", "max_upload_bytes", default=str(50 * 1024 * 1024)))


settings = Settings()


def get_settings() -> Settings:
    return settings
