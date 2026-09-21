"""Evaluation orchestrator: runs all phases and emits a markdown report.

Phases:
    1. dataset    — LLM-assisted QA candidates + curation (skipped if the dataset exists)
    2. retrieval  — embedding models x top_k sweep (HitRate/MRR/Recall)
    3. chunks     — chunk size sweep (paragraph groups + char windows)
    4. ragas      — six quality metrics on the GENERATION model (judge = vllm-judge)

Report: data/eval/evaluation_report.md — best embedding model, best top_k,
best chunk size, ragas scores. Locust is run separately (interactive load).

Run:
    python -m eval.run_evaluation
    python -m eval.run_evaluation --skip-dataset --skip-chunks   # ragas only
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from .chunk_sweep import _configs, run_sweep as run_chunk_sweep
from .dataset import DATASET_PATH, curate, generate_questions
from .ragas_eval import run_ragas_eval
from .retrieval_eval import evaluate_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "src" / "data" / "raw" / "egyptian_civil_code.pdf"
DEFAULT_CHUNKS = PROJECT_ROOT / "src" / "data" / "processed" / "law_chunks.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "evaluation_report.md"


def _write_report(
    retrieval_results: list[dict],
    chunk_results: list[dict],
    ragas_scores: dict,
    params: dict,
) -> Path:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"- Generation model (evaluated): `{params['generation_model']}`",
        f"- Judge model (scoring only): `{params['judge_model']}`",
        "",
    ]

    if retrieval_results:
        lines += ["## Retrieval sweep (embedding models x top_k)", "", "| model | top_k | hit_rate | mrr | recall |", "|---|---|---|---|---|"]
        for item in retrieval_results:
            top_k = next(int(key.split("@")[1]) for key in item if key.startswith("hit_rate@"))
            lines.append(
                f"| {item['embedding_model']} | {top_k} | {item[f'hit_rate@{top_k}']} "
                f"| {item[f'mrr@{top_k}']} | {item[f'recall@{top_k}']} |"
            )
        best = max(retrieval_results, key=lambda i: next(v for k, v in i.items() if k.startswith("mrr@")))
        best_top_k = next(int(k.split("@")[1]) for k in best if k.startswith("hit_rate@"))
        lines += ["", f"**Best embedding model:** `{best['embedding_model']}` (top_k={best_top_k})", ""]

    if chunk_results:
        lines += ["## Chunk size sweep", "", "| config | chunks | hit_rate | mrr | recall |", "|---|---|---|---|---|"]
        for item in chunk_results:
            top_k = next(int(key.split("@")[1]) for key in item if key.startswith("hit_rate@"))
            lines.append(
                f"| {item['chunk_label']} | {item['chunk_count']} | {item[f'hit_rate@{top_k}']} "
                f"| {item[f'mrr@{top_k}']} | {item[f'recall@{top_k}']} |"
            )
        best = max(chunk_results, key=lambda i: next(v for k, v in i.items() if k.startswith("mrr@")))
        lines += ["", f"**Best chunk size:** `{best['chunk_label']}`", ""]

    if ragas_scores:
        lines += [
            "## Ragas quality (generation model performance)",
            "",
            "| metric | score |",
            "|---|---|",
        ]
        lines += [f"| {name} | {score:.4f} |" for name, score in sorted(ragas_scores.items())]
        lines += [""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    from .dataset import DEFAULT_ARTICLES
    from .retrieval_eval import _load_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dataset", action="store_true", help="Reuse the existing qa_dataset.jsonl")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--articles", type=int, default=30)
    parser.add_argument("--per-article", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=["gate-arabert-v1", "arabert-all-nli-triplet-matryoshka", "bge-m3"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()

    # ---- Phase 1: dataset ----
    if args.skip_dataset and DATASET_PATH.is_file():
        print(f"dataset: reusing {DATASET_PATH}")
    else:
        print("dataset: generating LLM-assisted candidates ...")
        try:
            generate_questions(args.articles_path, count=args.articles, per_article=args.per_article)
        except Exception as error:
            print(f"dataset generation failed: {error}")
            if not DATASET_PATH.is_file():
                return 1
        curated = curate()
        print(f"dataset: {len(curated)} curated questions")

    dataset = _load_jsonl(DATASET_PATH)
    if not dataset:
        print("dataset is empty — cannot evaluate")
        return 1

    retrieval_results: list[dict] = []
    chunk_results: list[dict] = []
    ragas_scores: dict = {}

    # ---- Phase 2: retrieval sweep ----
    if not args.skip_retrieval:
        chunks = _load_jsonl(args.chunks)
        if chunks:
            print("retrieval: sweeping embedding models x top_k ...")
            for model_name in args.models:
                for top_k in args.top_k:
                    try:
                        metrics = asyncio.run(evaluate_config(dataset, chunks, model_name, top_k))
                    except Exception as error:
                        print(f"  {model_name} top_k={top_k} FAILED: {error}")
                        continue
                    print(f"  {json.dumps(metrics)}")
                    retrieval_results.append(metrics)
        else:
            print("retrieval: chunks file missing — skipping")

    # ---- Phase 3: chunk size sweep ----
    if not args.skip_chunks and args.pdf.is_file():
        from data.prepare_law import _article_records

        print("chunks: parsing PDF (once) ...")
        articles = _article_records(args.pdf)
        configs = _configs([1, 2, 3], [512, 1024], [0, 128])
        print(f"chunks: {len(configs)} configurations ...")
        chunk_results = asyncio.run(run_chunk_sweep(dataset, articles, "bge-m3", 5, configs))

    # ---- Phase 4: ragas ----
    if not args.skip_ragas:
        print("ragas: evaluating the generation model (judge scores) ...")
        from .ragas_eval import _run_pipeline

        from law_api.config import settings as app_settings

        try:
            rows = asyncio.run(
                _run_pipeline(dataset, _load_jsonl(args.chunks), app_settings.embedding_name, 5)
            )
            ragas_scores = run_ragas_eval(rows)
        except Exception as error:
            print(f"ragas evaluation failed: {error}")

    report = _write_report(
        retrieval_results,
        chunk_results,
        ragas_scores,
        params={
            "generation_model": "not evaluated" if args.skip_ragas else _generation_label(),
            "judge_model": _judge_label(),
        },
    )
    print(f"\nreport written to {report}")
    return 0


def _generation_label() -> str:
    from law_api.config import settings

    return settings.eval_generation_model


def _judge_label() -> str:
    from law_api.config import settings

    return settings.eval_judge_model


if __name__ == "__main__":
    raise SystemExit(main())
