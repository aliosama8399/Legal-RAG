from .EmbeddingInterface import EmbeddingInterface
from .providers import SentenceTransformerProvider

EMBEDDING_MODELS = {
    "gate-arabert-v1": "Omartificial-Intelligence-Space/GATE-AraBERT-v1",
    "arabert-all-nli-triplet-matryoshka": (
        "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka"
    ),
    "bge-m3": "BAAI/bge-m3",
}


class EmbeddingProviderFactory:
    """Instantiate an embedding provider by short name."""

    @staticmethod
    def create(name: str) -> EmbeddingInterface:
        try:
            model_id = EMBEDDING_MODELS[name]
        except KeyError as error:
            choices = ", ".join(sorted(EMBEDDING_MODELS))
            raise ValueError(f"Unknown embedding model '{name}'. Choose one of: {choices}") from error
        return SentenceTransformerProvider(name=name, model_id=model_id)
