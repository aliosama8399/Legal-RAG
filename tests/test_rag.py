"""Unit & integration tests for Hybrid RAG retrieval, RRF fusion, and Mock LLM generation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from law_api.services import RAGQueryService, _bilingual_chunk_text, _fuse_rrf
from law_api.stores.llm.providers import MockLLMProvider
from law_api.stores.vectordb.providers.QdrantProvider import QdrantProvider


class FakeEmbedder:
    name = "fake-embedder"
    model_id = "fake-embedder"
    _vocab: ClassVar[list[str]] = [
        "contract",
        "sale",
        "obligation",
        "lease",
        "servitude",
        "law",
        "nullity",
        "damages",
        "هبة",
        "المستقبلة",
        "gift",
    ]

    def _feats(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [float(lowered.count(w)) for w in self._vocab]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._feats(t) for t in texts]


async def _build_rag_service(tmp_path: Path) -> RAGQueryService:
    store = QdrantProvider(collection="rag_test", path=tmp_path / "qdrant")
    await store.connect()
    embedder = FakeEmbedder()
    chunks = [
        {
            "chunk_id": "147:paragraph:1",
            "article_number": 147,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 147",
            "source_page": 24,
            "text_ar": "(1) العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
            "text_en": "The contract makes the law of the parties. It can be revoked only by mutual consent.",
            "chunk_text": "(1) العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
            "is_repealed": False,
        },
        {
            "chunk_id": "148:paragraph:1",
            "article_number": 148,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 148",
            "source_page": 24,
            "text_ar": "يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما يوجبه حسن النية.",
            "text_en": "A contract must be performed in accordance with its contents and in good faith.",
            "chunk_text": "A contract must be performed in accordance with its contents and in good faith.",
            "is_repealed": False,
        },
        {
            "chunk_id": "450:paragraph:1",
            "article_number": 450,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 450",
            "source_page": 72,
            "text_ar": "",
            "text_en": "When the purchaser has given notice to the vendor of the defect in the thing sold.",
            "chunk_text": "When the purchaser has given notice to the vendor of the defect in the thing sold.",
            "is_repealed": False,
        },
        {
            "chunk_id": "492:paragraph:1",
            "article_number": 492,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 492",
            "source_page": 78,
            "text_ar": "تقع هبة الأموال المستقبلة باطلة.",
            "text_en": "A gift of future property is void.",
            "chunk_text": "تقع هبة الأموال المستقبلة باطلة.",
            "is_repealed": False,
        },
        {
            "chunk_id": "499:paragraph:1",
            "article_number": 499,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 499",
            "source_page": 79,
            "text_ar": "إذا اشترط الواهب عوضا عن الهبة وفاء ديونه فلا يكون الموهوب له ملزما إلا بوفاء الديون.",
            "text_en": "If the donor stipulates that the donee shall pay the donor's debts.",
            "chunk_text": "إذا اشترط الواهب عوضا عن الهبة وفاء ديونه فلا يكون الموهوب له ملزما إلا بوفاء الديون.",
            "is_repealed": False,
        },
        {
            "chunk_id": "558:paragraph:1",
            "article_number": 558,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 558",
            "source_page": 88,
            "text_ar": "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع بشيء معين مدة معينة.",
            "text_en": "A lease is a contract by which the lessor undertakes to enable the lessee to enjoy a specific thing.",
            "chunk_text": "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكن المستأجر من الانتفاع بشيء معين مدة معينة.",
            "is_repealed": False,
        },
        {
            "chunk_id": "60:paragraph:1",
            "article_number": 60,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 60",
            "source_page": 10,
            "text_ar": "",
            "text_en": "Contract repealed provision placeholder.",
            "chunk_text": "Contract repealed provision placeholder.",
            "is_repealed": True,
        },
    ]
    embeddings = await embedder.encode([_bilingual_chunk_text(c) for c in chunks])
    await store.save_document(
        "civil_code.pdf",
        "civil_code.pdf",
        3,
        chunks,
        embeddings,
        embedder.model_id,
        tenant_id="org_default",
    )
    return RAGQueryService(
        storage=store,
        embedder=embedder,
        llm=MockLLMProvider(),
        top_k=3,
        hybrid_search=True,
    )


def test_rrf_penalizes_repealed_articles_unless_explicitly_queried():
    dense = [
        {"document_id": 1, "chunk_id": "60:1", "is_repealed": True, "score": 0.9},
        {"document_id": 1, "chunk_id": "147:1", "is_repealed": False, "score": 0.85},
    ]
    lexical = [
        {"document_id": 1, "chunk_id": "60:1", "is_repealed": True, "score": 1.0},
        {"document_id": 1, "chunk_id": "147:1", "is_repealed": False, "score": 0.9},
    ]
    fused = _fuse_rrf(dense, lexical, question="What is binding force of a contract?")
    assert fused[0]["chunk_id"] == "147:1"


@pytest.mark.asyncio
async def test_ask_returns_answer_and_article_citations(tmp_path: Path):
    rag = await _build_rag_service(tmp_path)
    try:
        result = await rag.ask("What does Article 147 say about contract?", top_k=2)
        assert "Egyptian Civil Code, Article 147" in result["answer"]
        assert result["trace_id"]
        assert result["sources"]
        assert result["sources"][0]["article_number"] == 147
    finally:
        await rag.storage.disconnect()


@pytest.mark.asyncio
async def test_ask_stream_emits_sources_tokens_and_done(tmp_path: Path):
    rag = await _build_rag_service(tmp_path)
    try:
        events = [event async for event in rag.ask_stream("good faith contract", top_k=2)]
        types = [e["type"] for e in events]
        assert types[0] == "sources"
        assert "token" in types
        assert types[-1] == "done"
        assert events[-1]["trace_id"]
    finally:
        await rag.storage.disconnect()


@pytest.mark.asyncio
async def test_bilingual_retrieval_ranks_lease_558_and_gift_492_first(tmp_path: Path):
    rag = await _build_rag_service(tmp_path)
    try:
        lease_res = await rag.ask("What is a lease contract?", top_k=3)
        assert lease_res["sources"]
        assert lease_res["sources"][0]["article_number"] == 558

        gift_res = await rag.ask("ما حكم هبة الأموال المستقبلة؟", top_k=3)
        assert gift_res["sources"]
        assert gift_res["sources"][0]["article_number"] == 492
    finally:
        await rag.storage.disconnect()


@pytest.mark.asyncio
async def test_ask_declines_out_of_scope_criminal_query(tmp_path: Path):
    rag = await _build_rag_service(tmp_path)
    try:
        result = await rag.ask("What is the penalty for theft?", top_k=2)
        assert result["sources"] == []
        assert "out of scope" in result["answer"].lower()
        assert "imprisonment" not in result["answer"].lower()

        non_civil = await rag.ask(
            "What is the corporate capital gains tax rate for SaaS IPOs?", top_k=2
        )
        assert non_civil["sources"] == []
        assert "out of scope" in non_civil["answer"].lower()

        stream_events = [
            event async for event in rag.ask_stream("ما هي عقوبة السرقة؟", top_k=2)
        ]
        done = stream_events[-1]
        assert done["type"] == "done"
        assert done["sources"] == []
        assert "out of scope" in done["answer"].lower()
    finally:
        await rag.storage.disconnect()
