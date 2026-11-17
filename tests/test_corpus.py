"""Validation tests for the bilingual Egyptian Civil Code extraction and chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.prepare_law import (
    MAX_ARTICLE,
    REPEALED_RANGES,
    _repealed,
    build_chunks,
    normalize_arabic,
)

AR_EXPECTED_EMPTY = {1022}
MAX_ARTICLE_CHARS = 6000


def _corpus_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "data" / "processed" / "law_articles.jsonl"


def _load_corpus() -> list[dict]:
    path = _corpus_path()
    if not path.exists():
        pytest.skip("src/data/processed/law_articles.jsonl not present on disk")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _load_corpus()


@pytest.fixture(scope="module")
def by_number(corpus) -> dict[int, dict]:
    return {r["article_number"]: r for r in corpus}


def test_normalize_arabic_digits_and_spacing():
    assert normalize_arabic("مادة (١٤٧)  العقد") == "مادة (147) العقد"


def test_build_chunks_preserves_citation_and_hierarchy():
    sample = [
        {
            "article_number": 147,
            "book": "Obligations",
            "chapter": "Contracts",
            "section": "Effects",
            "topic": "Binding Force",
            "text_ar": "(1) العقد شريعة المتعاقدين. (2) ولا يجوز نقضه.",
            "text_en": "The contract makes the law of the parties.",
            "is_repealed": False,
            "source_page": 24,
            "citation": "Egyptian Civil Code, Article 147",
        }
    ]
    chunks = build_chunks(sample)
    assert len(chunks) == 2
    assert all(c["citation"] == "Egyptian Civil Code, Article 147" for c in chunks)
    assert all(c["article_number"] == 147 for c in chunks)


def test_record_count_equals_max_article(corpus):
    assert [r["article_number"] for r in corpus] == list(range(1, MAX_ARTICLE + 1))


def test_article_numbers_contiguous(by_number):
    missing = [n for n in range(1, MAX_ARTICLE + 1) if n not in by_number]
    assert len(by_number) == MAX_ARTICLE
    assert missing == []


def test_repealed_articles_flagged(by_number):
    flagged = {n for n, r in by_number.items() if r["is_repealed"]}
    expected = set()
    for lo, hi in REPEALED_RANGES:
        expected.update(range(lo, hi + 1))
    assert flagged == expected


def test_non_repealed_not_flagged(by_number):
    for n, r in by_number.items():
        if not _repealed(n):
            assert not r["is_repealed"], f"article {n} wrongly flagged repealed"


def test_record_length_sane(by_number):
    for r in by_number.values():
        for field in ("text_ar", "text_en"):
            assert len(r[field]) <= MAX_ARTICLE_CHARS


def test_citation_format(by_number):
    for n, r in by_number.items():
        assert r["citation"] == f"Egyptian Civil Code, Article {n}"


def test_metadata_fields_present(by_number):
    for r in by_number.values():
        for k in (
            "article_number",
            "book",
            "chapter",
            "section",
            "topic",
            "text_ar",
            "text_en",
            "is_repealed",
            "source_page",
            "citation",
        ):
            assert k in r
