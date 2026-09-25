import asyncio

from ..RerankerInterface import RerankerInterface


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder via sentence-transformers (e.g. BAAI/bge-reranker-v2-m3).

    Scoring is blocking CPU work, so it runs in a worker thread.
    """

    def __init__(self, name: str, model_id: str, max_length: int = 256) -> None:
        self.name = name
        self.model_id = model_id
        # Truncate query+chunk pairs at tokenize time: without this the
        # tokenizer scores at the model's full max_length (up to 8192 for
        # v2-m3) — full-length paragraphs on CPU are extremely slow.
        self.max_length = max_length
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=self.max_length)
        return self._model

    def _score(self, pairs: list[list[str]]) -> list[float]:
        model = self._ensure_model()
        scores = model.predict(pairs)
        return list(scores)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        return await asyncio.to_thread(self._score, pairs)
