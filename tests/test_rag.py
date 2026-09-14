"""End-to-end test of the RAG pipeline (fake embedder + Qdrant local mode +
extractive LLM), so no model download is required."""

from __future__ import annotations



from legal_rag.rag import RAGPipeline
from legal_rag.vectordb.base import Chunk
from legal_rag.vectordb.qdrant import QdrantStore
from legal_rag.llm.base import ExtractiveLLM


class FakeEmbedder:
    """Deterministic term-overlap embedding over a fixed vocabulary."""

    dim = 8
    _vocab = ["contract", "sale", "obligation", "lease", "servitude", "law", "الحق", "العقد"]

    def _feats(self, text: str) -> list[float]:
        t = text.lower()
        v = [float(t.count(w)) for w in self._vocab]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._feats(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._feats(text)


def _build_pipeline() -> RAGPipeline:
    store = QdrantStore(":memory:", "rag_test")
    store.ensure_collection(8)
    store.add([
        Chunk(id="1:article", vector=[1, 0, 0, 0, 0, 0, 0, 0], article_number=1,
              citation="Egyptian Civil Code, Article 1", text_en="the contract is the law",
              text_ar="العقد شريعة المتعاقدين"),
        Chunk(id="2:article", vector=[0, 0, 1, 0, 0, 0, 0, 0], article_number=2,
              citation="Egyptian Civil Code, Article 2", text_en="sale obligation"),
    ])
    return RAGPipeline(store=store, embedder=FakeEmbedder(), llm=ExtractiveLLM())


def test_ask_returns_answer_and_citation_sources():
    rag = _build_pipeline()
    result = rag.ask("what is a contract?")
    assert "answer" in result
    assert result["sources"]
    assert result["sources"][0] == "Egyptian Civil Code, Article 1"


def test_ask_sources_are_article_citations_not_chunk_ids():
    rag = _build_pipeline()
    result = rag.ask("sale")
    for src in result["sources"]:
        assert src.startswith("Egyptian Civil Code, Article ")


def test_retrieve_respects_top_k():
    rag = _build_pipeline()
    hits = rag.retrieve("contract", top_k=1)
    assert len(hits) == 1