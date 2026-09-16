import asyncio

from ..EmbeddingInterface import EmbeddingInterface


class SentenceTransformerProvider(EmbeddingInterface):
    """Sentence-Transformers embedding model, loaded lazily on first encode.

    Encoding is blocking CPU work, so it runs in a worker thread to keep
    the FastAPI event loop responsive.
    """

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._model = None

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)
