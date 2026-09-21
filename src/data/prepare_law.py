"""Extract bilingual law articles and citation-preserving chunks.

This file is intentionally self-contained. It does not import the legacy
``legal_rag.ingest`` pipeline. ``pdfplumber`` is used because word coordinates
are required to separate the English left column from the Arabic right column.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

COLUMN_BOUNDARY = 292.0
MAX_ARTICLE = 1149
REPEALED_RANGES = ((54, 80), (389, 417))
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
EN_HEADER = re.compile(r"^Article\s*(\d{1,4})\b\s*(.*)$", re.IGNORECASE)
AR_HEADER = re.compile(r"^مادة\s*[()\s]*([٠-٩0-9]+)")
PARAGRAPH = re.compile(r"(?=\(\s*\d+\s*\))")


def normalize_arabic(text: str) -> str:
    """Normalize digits, punctuation, and spacing without guessing Arabic words."""
    text = unicodedata.normalize("NFC", text).translate(ARABIC_DIGITS)
    text = re.sub(r"\(\s*(\d+)\s*\(", r"(\1)", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_arabic(character: str) -> bool:
    return any(
        start <= ord(item) <= end
        for item in character
        for start, end in (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFEFF),
        )
    )


def _is_weak_digit_run(text: str) -> bool:
    """Arabic-Indic and ASCII digits (and simple punctuation among them, e.g.
    parenthesised paragraph numbers) form their own left-to-right "weak" run
    inside RTL text per the Unicode bidi algorithm -- they must never be
    internally reversed, even though Arabic-Indic digits share the Arabic
    Unicode block with the letters around them."""
    stripped = text.strip("()., ")
    return bool(stripped) and all(ch.isdigit() or ch in "()., " for ch in text)


def _word_glyphs(word: dict, chars: list[dict]) -> list[str]:
    """Return this word's glyphs in on-page extraction order, as pdfplumber
    itself identified them.

    Crucially this reads from ``page.chars`` (bounding-box matched to the
    word), not from the already-flattened ``word["text"]`` string. When the
    PDF's font fuses two letters into a single rendered shape -- Arabic's
    mandatory Lam+Alef ligature is the common case -- pdfminer/pdfplumber
    already represents that as ONE character object whose text is the
    two-codepoint value (e.g. 'لأ'). Reading glyphs at this level means a
    fused ligature is naturally treated as one atomic unit during reversal,
    for any word, without pattern-matching specific letter sequences or
    specific words.
    """
    return [
        c["text"]
        for c in chars
        if word["x0"] - 0.5 <= c["x0"] <= word["x1"] + 0.5 and abs(c["top"] - word["top"]) < 3
    ]


def _ordered_word_text(word: dict, chars: list[dict], rtl: bool) -> str:
    glyphs = _word_glyphs(word, chars)
    if not rtl:
        return "".join(glyphs)
    # Digits/numbering are a weak LTR run even inside an RTL word/line
    # (e.g. a paragraph marker like "(٢)") -- keep their internal order,
    # only letters get their glyph order reversed.
    if _is_weak_digit_run("".join(glyphs)):
        return "".join(glyphs)
    return "".join(reversed(glyphs))


def _column_lines(words: list[dict], chars: list[dict], rtl: bool) -> list[str]:
    # Bucket chars by vertical position once per page. _word_glyphs tolerates
    # |c.top - word.top| < 3, so a word can only match chars in the three
    # adjacent buckets — this turns the per-word scan from O(all page chars)
    # into O(chars in a ~9pt band), a ~10-20x speedup with identical output.
    chars_by_bucket: dict[int, list[dict]] = {}
    for char in chars:
        chars_by_bucket.setdefault(int(char["top"] // 3), []).append(char)

    def candidates_for(word: dict) -> list[dict]:
        base = int(word["top"] // 3)
        candidates: list[dict] = []
        for bucket in (base - 1, base, base + 1):
            candidates.extend(chars_by_bucket.get(bucket, ()))
        return candidates

    grouped: list[list[dict]] = []
    for word in sorted(words, key=lambda item: item["top"]):
        if not grouped or abs(word["top"] - grouped[-1][0]["top"]) > 3.5:
            grouped.append([])
        grouped[-1].append(word)
    lines = []
    for line in grouped:
        ordered = sorted(line, key=lambda item: item["x0"], reverse=rtl)
        values = [_ordered_word_text(item, candidates_for(item), rtl) for item in ordered]
        lines.append(" ".join(values))
    return lines


def _page_columns(pdf_path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    english: list[tuple[int, str]] = []
    arabic: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            words = page.extract_words()
            chars = page.chars
            english.extend((page_number, line) for line in _column_lines([w for w in words if w["x0"] < COLUMN_BOUNDARY], chars, False))
            arabic.extend((page_number, line) for line in _column_lines([w for w in words if w["x0"] >= COLUMN_BOUNDARY], chars, True))
    return english, arabic


def _repealed(number: int) -> bool:
    return any(first <= number <= last for first, last in REPEALED_RANGES)


def _article_records(pdf_path: Path) -> list[dict]:
    english, arabic = _page_columns(pdf_path)
    articles: dict[int, dict] = {}
    current: dict | None = None
    for page, line in english:
        line = line.strip()
        match = EN_HEADER.match(line)
        if match and 1 <= int(match.group(1)) <= MAX_ARTICLE:
            number = int(match.group(1))
            if number in articles:
                continue
            current = {
                "article_number": number,
                "book": "",
                "chapter": "",
                "section": "",
                "topic": "",
                "text_en": match.group(2).strip(),
                "text_ar": "",
                "is_repealed": _repealed(number),
                "source_page": page,
                "citation": f"Egyptian Civil Code, Article {number}",
            }
            articles[number] = current
        elif current and line and not _is_arabic(line):
            current["text_en"] = f"{current['text_en']} {line}".strip()

    arabic_segments: list[tuple[str, list[str]]] = []
    digits: str | None = None
    body: list[str] = []
    pending_header = False
    for _, line in arabic:
        line = line.strip()
        if pending_header and re.fullmatch(r"[٠-٩0-9]+\)?", line):
            match = re.match(r"[٠-٩0-9]+", line)
            assert match is not None
            if digits is not None:
                arabic_segments.append((digits, body))
            digits, body = match.group(0), []
            pending_header = False
            continue
        match = AR_HEADER.match(line)
        if match:
            if digits is not None:
                arabic_segments.append((digits, body))
            digits, body = match.group(1), []
        elif "مادة" in line and not re.search(r"[٠-٩0-9]", line):
            pending_header = True
        elif digits is not None and line:
            body.append(line)
    if digits is not None:
        arabic_segments.append((digits, body))

    assigned: set[int] = set()
    for raw_digits, segment_body in arabic_segments:
        normalized_digits = raw_digits.translate(ARABIC_DIGITS)
        candidates = {
            number
            for number in (int(normalized_digits), int(normalized_digits[::-1]))
            if number in articles and number not in assigned
        }
        if candidates:
            number = min(candidates)
            articles[number]["text_ar"] = normalize_arabic(" ".join(segment_body))
            assigned.add(number)

    records = []
    for number in range(1, MAX_ARTICLE + 1):
        record = articles.get(number, {
            "article_number": number,
            "book": "",
            "chapter": "",
            "section": "",
            "topic": "",
            "text_en": "This article has been repealed and is no longer in force." if _repealed(number) else "",
            "text_ar": "",
            "is_repealed": _repealed(number),
            "source_page": 0,
            "citation": f"Egyptian Civil Code, Article {number}",
        })
        records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _windowed(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows with overlap."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)] or [text]


def build_chunks(
    articles: list[dict],
    *,
    paragraphs_per_chunk: int = 1,
    chunk_chars: int | None = None,
    overlap_chars: int = 0,
) -> list[dict]:
    """Chunk articles into citation-preserving chunks.

    - Default: one paragraph per chunk (backward-compatible original behavior).
    - ``paragraphs_per_chunk``: group N consecutive paragraphs per chunk.
    - ``chunk_chars``: windowed chunking (overrides paragraph grouping);
      ``overlap_chars`` sets the window overlap.
    """
    chunks: list[dict] = []
    for article in articles:
        text = article["text_ar"] or article["text_en"]
        if chunk_chars:
            parts = _windowed(text, chunk_chars, overlap_chars)
            chunk_prefix = f"window{chunk_chars}+{overlap_chars}"
        else:
            parts = [part.strip() for part in PARAGRAPH.split(text) if part.strip()] or [text]
            if paragraphs_per_chunk > 1:
                parts = [
                    " ".join(parts[i : i + paragraphs_per_chunk])
                    for i in range(0, len(parts), paragraphs_per_chunk)
                ]
            chunk_prefix = f"paragraphs{paragraphs_per_chunk}" if paragraphs_per_chunk > 1 else "paragraph"
        for number, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": f"{article['article_number']}:{chunk_prefix}:{number}",
                    "article_number": article["article_number"],
                    "book": article["book"],
                    "chapter": article["chapter"],
                    "section": article["section"],
                    "topic": article["topic"],
                    "text_ar": article["text_ar"],
                    "text_en": article["text_en"],
                    "chunk_text": part,
                    "chunk_number": number,
                    "is_repealed": article["is_repealed"],
                    "source_page": article["source_page"],
                    "citation": article["citation"],
                }
            )
    return chunks


def prepare(
    pdf_path: Path,
    output_dir: Path,
    *,
    paragraphs_per_chunk: int = 1,
    chunk_chars: int | None = None,
    overlap_chars: int = 0,
) -> tuple[Path, Path]:
    articles = _article_records(pdf_path)
    article_path = output_dir / "law_articles.jsonl"
    _write_jsonl(article_path, articles)

    chunks = build_chunks(
        articles,
        paragraphs_per_chunk=paragraphs_per_chunk,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    chunk_path = output_dir / "law_chunks.jsonl"
    _write_jsonl(chunk_path, chunks)
    return article_path, chunk_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Canonical bilingual law PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for article and chunk JSONL files",
    )
    parser.add_argument("--paragraphs-per-chunk", type=int, default=1)
    parser.add_argument("--chunk-chars", type=int, default=None, help="Windowed chunking size (overrides paragraphs)")
    parser.add_argument("--overlap-chars", type=int, default=0)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")
    article_path, chunk_path = prepare(
        args.pdf,
        args.output_dir,
        paragraphs_per_chunk=args.paragraphs_per_chunk,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"wrote articles to {article_path}")
    print(f"wrote chunks to {chunk_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())