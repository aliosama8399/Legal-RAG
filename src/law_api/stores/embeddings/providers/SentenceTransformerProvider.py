import asyncio

from ..EmbeddingInterface import EmbeddingInterface


class SentenceTransformerProvider(EmbeddingInterface):
    """Sentence-Transformers embedding model, preloaded on startup or lazy-loaded on first encode.

    Encoding is blocking CPU work, so it runs in a worker thread to keep
    the FastAPI event loop responsive.
    """

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_id, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_id)
            self._model.max_seq_length = min(
                getattr(self._model, "max_seq_length", 256) or 256, 256
            )
        return self._model

    async def warmup(self) -> None:
        await asyncio.to_thread(self._ensure_model)

    def _encode_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        return self._encode_batch_sync(texts)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 64:
            return await asyncio.to_thread(self._encode_batch_sync, texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            vectors.extend(await asyncio.to_thread(self._encode_batch_sync, batch))
            await asyncio.sleep(0.01)
        return vectors
