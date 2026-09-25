from .RerankerEnum import RerankerEnum
from .RerankerInterface import RerankerInterface
from .providers import CrossEncoderReranker

RERANKER_MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}


class RerankerProviderFactory:
    """Instantiate a cross-encoder reranker by short name.

    ``max_length`` truncates scored pairs at tokenize time — the dominant
    CPU-speed lever (smaller = faster).
    """

    @staticmethod
    def create(name: str, max_length: int = 256) -> RerankerInterface:
        try:
            model_id = RERANKER_MODELS[name]
        except KeyError as error:
            choices = ", ".join(sorted(RERANKER_MODELS))
            raise ValueError(f"Unknown reranker model '{name}'. Choose one of: {choices}") from error
        return CrossEncoderReranker(name=name, model_id=model_id, max_length=max_length)
