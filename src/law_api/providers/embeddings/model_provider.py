from typing import Protocol

EMBEDDING_MODELS = {
    "gate-arabert-v1": "Omartificial-Intelligence-Space/GATE-AraBERT-v1",
    "arabert-all-nli-triplet-matryoshka": (
        "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka"
    ),
    "bge-m3": "BAAI/bge-m3",
}


class EmbeddingProvider(Protocol):
    name: str
    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerProvider:
    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def create_embedding_provider(name: str) -> EmbeddingProvider:
    try:
        model_id = EMBEDDING_MODELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(EMBEDDING_MODELS))
        raise ValueError(f"Unknown embedding model '{name}'. Choose one of: {choices}") from error
    return SentenceTransformerProvider(name=name, model_id=model_id)