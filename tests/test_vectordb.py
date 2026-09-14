"""Tests for the vector-store provider abstraction (uses Qdrant local mode,
so no server is required)."""

from __future__ import annotations

import pytest

from legal_rag.vectordb.base import Chunk
from legal_rag.vectordb.qdrant import QdrantStore


@pytest.fixture()
def store() -> QdrantStore:
    s = QdrantStore(":memory:", "test_collection")
    s.ensure_collection(dim=4)
    return s


def test_add_and_count(store):
    chunks = [
        Chunk(id=f"{n}:article", vector=[1.0, 0.0, 0.0, 0.0],
              article_number=n, citation=f"Egyptian Civil Code, Article {n}")
        for n in range(1, 4)
    ]
    store.add(chunks)
    assert store.count() == 3


def test_query_returns_nearest_with_citation(store):
    store.add([
        Chunk(id="1:article", vector=[1.0, 0.0, 0.0, 0.0], article_number=1,
              citation="Egyptian Civil Code, Article 1"),
        Chunk(id="2:article", vector=[0.0, 1.0, 0.0, 0.0], article_number=2,
              citation="Egyptian Civil Code, Article 2"),
    ])
    hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert hits[0].citation == "Egyptian Civil Code, Article 1"
    assert hits[0].id == "1:article"


def test_query_respects_metadata_filter(store):
    store.add([
        Chunk(id="1:article", vector=[1.0, 0.0, 0.0, 0.0], article_number=1,
              citation="A1"),
        Chunk(id="2:article", vector=[1.0, 0.0, 0.0, 0.0], article_number=2,
              citation="A2"),
    ])
    hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"article_number": 2})
    assert [h.payload["article_number"] for h in hits] == [2]