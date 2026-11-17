"""Tests for the Qdrant vector-store provider with multi-tenant isolation and hybrid lexical search."""

from __future__ import annotations

from pathlib import Path

import pytest

from law_api.stores.vectordb.providers.QdrantProvider import QdrantProvider


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "1:paragraph:1",
            "article_number": 1,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 1",
            "source_page": 1,
            "chunk_text": "Provisions of laws govern all matters to which they apply in letter or spirit.",
            "is_repealed": False,
        },
        {
            "chunk_id": "147:paragraph:1",
            "article_number": 147,
            "chunk_number": 1,
            "citation": "Egyptian Civil Code, Article 147",
            "source_page": 24,
            "chunk_text": "The contract makes the law of the parties.",
            "is_repealed": False,
        },
    ]


@pytest.mark.asyncio
async def test_tenant_isolation_between_orgs(tmp_path: Path):
    qdrant_store = QdrantProvider(collection="test_collection", path=tmp_path / "qdrant")
    await qdrant_store.connect()
    try:
        chunks = _sample_chunks()
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        doc_a, _ = await qdrant_store.save_document(
            "tenant_a.pdf", "a.pdf", 2, chunks, vectors, "test-model", tenant_id="org_alpha"
        )
        hits_alpha = await qdrant_store.search(
            None, [0.0, 1.0, 0.0, 0.0], top_k=5, tenant_id="org_alpha"
        )
        assert len(hits_alpha) == 2
        assert hits_alpha[0]["article_number"] == 147

        with pytest.raises(ValueError, match="No relevant chunks were found"):
            await qdrant_store.search(None, [0.0, 1.0, 0.0, 0.0], top_k=5, tenant_id="org_beta")

        assert await qdrant_store.list_documents(tenant_id="org_beta") == []
        assert len(await qdrant_store.list_documents(tenant_id="org_alpha")) == 1
        assert await qdrant_store.delete_document(doc_a, tenant_id="org_beta") is False
        assert await qdrant_store.delete_document(doc_a, tenant_id="org_alpha") is True
        assert await qdrant_store.list_documents(tenant_id="org_alpha") == []
    finally:
        await qdrant_store.disconnect()


@pytest.mark.asyncio
async def test_lexical_search_matches_explicit_article_number(tmp_path: Path):
    qdrant_store = QdrantProvider(collection="test_collection", path=tmp_path / "qdrant")
    await qdrant_store.connect()
    try:
        chunks = _sample_chunks()
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        await qdrant_store.save_document(
            "code.pdf", "code.pdf", 2, chunks, vectors, "test-model", tenant_id="org_alpha"
        )
        hits = await qdrant_store.lexical_search(
            None, "What does Article 147 state?", top_k=2, tenant_id="org_alpha"
        )
        assert hits
        assert hits[0]["article_number"] == 147
    finally:
        await qdrant_store.disconnect()
