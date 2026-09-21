"""Build the evaluation QA dataset: LLM-assisted question generation + curation.

Generate:  python -m eval.dataset generate --articles 30 --per-article 2
Curate:    python -m eval.dataset curate

Candidates: data/eval/qa_candidates.jsonl {question, answer, article_number, citation}
Curated:    data/eval/qa_dataset.jsonl {question, ground_truth_answer, ground_truth_article}
"""

import argparse
import json
import random
import re
from pathlib import Path

from openai import OpenAI

from law_api.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTICLES = PROJECT_ROOT / "src" / "data" / "processed" / "law_articles.jsonl"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "eval" / "qa_candidates.jsonl"
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl"

GENERATION_SYSTEM = (
    "You are creating an evaluation dataset for a legal RAG system over the "
    "Egyptian Civil Code. Reply in strict JSON only."
)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _client(which: str) -> tuple[OpenAI, str]:
    base_url, model = (
        (settings.eval_judge_base_url, settings.eval_judge_model)
        if which == "judge"
        else (settings.eval_generation_base_url, settings.eval_generation_model)
    )
    return OpenAI(api_key="not-needed", base_url=base_url), model


def _parse_json_items(raw: str) -> list[dict]:
    text = raw.strip()
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [item for item in data if isinstance(item, dict)]


def generate_questions(
    articles_path: Path,
    *,
    count: int,
    per_article: int,
    which: str = "judge",
    seed: int = 42,
) -> list[dict]:
    """Sample articles and ask the configured LLM to write questions + answers."""
    random.seed(seed)
    articles = [
        article
        for article in _load_jsonl(articles_path)
        if (article.get("text_en") or article.get("text_ar")) and not article.get("is_repealed")
    ]
    if not articles:
        raise ValueError(f"No usable articles found at {articles_path} — run prepare_law first")
    sampled = random.sample(articles, min(count, len(articles)))
    client, model = _client(which)

    candidates: list[dict] = []
    for article in sampled:
        text = article.get("text_en") or article.get("text_ar")
        prompt = (
            f"Article {article['article_number']}:\n{text[:4000]}\n\n"
            f"Write {per_article} short factual questions a user could ask that are "
            "answerable ONLY from this article, each with its short answer. "
            'Strict JSON list: [{"question": "...", "answer": "..."}]'
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
        except Exception as error:
            print(f"article {article['article_number']}: generation failed: {error}")
            continue
        for item in _parse_json_items(response.choices[0].message.content or ""):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                candidates.append(
                    {
                        "question": question,
                        "answer": answer,
                        "article_number": article["article_number"],
                        "citation": article.get(
                            "citation", f"Egyptian Civil Code, Article {article['article_number']}"
                        ),
                    }
                )
    _write_jsonl(CANDIDATES_PATH, candidates)
    return candidates


def curate(
    candidates_path: Path = CANDIDATES_PATH,
    dataset_path: Path = DATASET_PATH,
) -> list[dict]:
    """Filter candidates into the final dataset: dedupe, drop malformed entries."""
    candidates = _load_jsonl(candidates_path)
    seen: set[str] = set()
    curated: list[dict] = []
    for item in candidates:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        number = item.get("article_number")
        if len(question) < 10 or "?" not in question:
            continue
        if len(answer) < 3 or number is None:
            continue
        key = re.sub(r"\s+", " ", question.lower())
        if key in seen:
            continue
        seen.add(key)
        curated.append(
            {
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_article": int(number),
                "citation": item.get("citation", f"Egyptian Civil Code, Article {int(number)}"),
            }
        )
    _write_jsonl(dataset_path, curated)
    return curated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="LLM-assisted candidate generation")
    generate.add_argument("--articles", type=int, default=30)
    generate.add_argument("--per-article", type=int, default=2)
    generate.add_argument("--which", choices=["judge", "generation"], default="judge")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES)

    curate_parser = subparsers.add_parser("curate", help="Filter candidates into the dataset")
    curate_parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    curate_parser.add_argument("--out", type=Path, default=DATASET_PATH)

    args = parser.parse_args()
    if args.command == "generate":
        candidates = generate_questions(
            args.articles_path,
            count=args.articles,
            per_article=args.per_article,
            which=args.which,
            seed=args.seed,
        )
        print(f"wrote {len(candidates)} candidates to {CANDIDATES_PATH}")
    else:
        curated = curate(args.candidates, args.out)
        print(f"wrote {len(curated)} curated questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
