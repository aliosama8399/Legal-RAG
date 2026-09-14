"""Validation tests for the built corpus (handbook STEP 0, rule 3).

Run from the repo root with:  python -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.ingest.models import MAX_ARTICLE, REPEALED_RANGES, is_in_repealed_range

# Articles 55-80 and 389-417 are repealed (flagged, not dropped).
# Article 1022 is the only present article whose source Arabic text is missing
# (a translation mismatch in the source PDF) - see ingest/corpus.py.
AR_EXPECTED_EMPTY = {1022}

# A single article should never exceed this many chars; a larger record means a
# failed split where two articles were merged.
MAX_ARTICLE_CHARS = 6000


def _load_corpus() -> list[dict]:
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus" / "corpus.jsonl"
    records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    return records


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _load_corpus()


@pytest.fixture(scope="module")
def by_number(corpus) -> dict[int, dict]:
    return {r["article_number"]: r for r in corpus}


def test_record_count_equals_max_article(corpus):
    assert [r["article_number"] for r in corpus] == list(range(1, MAX_ARTICLE + 1))


def test_article_numbers_contiguous(by_number):
    missing = [n for n in range(1, MAX_ARTICLE + 1) if n not in by_number]
    # the only genuine gaps are repealed ranges, which we still keep as records
    assert len(by_number) == MAX_ARTICLE
    assert missing == []


def test_no_unexplained_gaps(by_number):
    for n in range(1, MAX_ARTICLE + 1):
        rec = by_number[n]
        if not is_in_repealed_range(n):
            assert rec["text_en"], f"article {n} has empty English text"
            assert rec["text_ar"] or n in AR_EXPECTED_EMPTY, (
                f"article {n} has empty Arabic text"
            )


def test_repealed_articles_flagged(by_number):
    flagged = {n for n, r in by_number.items() if r["is_repealed"]}
    expected = set()
    for lo, hi in REPEALED_RANGES:
        expected.update(range(lo, hi + 1))
    assert flagged == expected


def test_non_repealed_not_flagged(by_number):
    for n, r in by_number.items():
        if not is_in_repealed_range(n):
            assert not r["is_repealed"], f"article {n} wrongly flagged repealed"


def test_record_length_sane(by_number):
    for n, r in by_number.items():
        for field in ("text_ar", "text_en"):
            assert len(r[field]) <= MAX_ARTICLE_CHARS, (
                f"article {n} {field} too long ({len(r[field])}): split failed"
            )


def test_citation_format(by_number):
    for n, r in by_number.items():
        assert r["citation"] == f"Egyptian Civil Code, Article {n}"


def test_source_page_nonzero_for_present_articles(by_number):
    for n, r in by_number.items():
        if not is_in_repealed_range(n):
            assert r["source_page"] > 0, f"article {n} missing source_page"


def test_metadata_fields_present(by_number):
    for r in by_number.values():
        for k in ("article_number", "book", "chapter", "section", "topic",
                  "text_ar", "text_en", "is_repealed", "source_page", "citation"):
            assert k in r