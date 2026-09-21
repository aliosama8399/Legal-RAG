"""Chunk size sweep: evaluate chunking configurations by retrieval quality.

Parses the PDF once, rebuilds chunks per configuration (paragraph groups or
char windows with overlap), runs the retrieval harness per config, and logs
every run to MLflow.

Run:
    python -m eval.chunk_sweep --pdf src/data/raw/egyptian_civil_code.pdf
    python -m eval.chunk_sweep --model bge-m3 --paragraph-groups 1 2 3 --windows 512 1024 --overlaps 0 128
"""

import argparse
import asyncio
import json
from pathlib import Path

from data.prepare_law import _article_records, build_chunks

from law_api.config import settings

from .dataset import DATASET_PATH, _load_jsonl
from .retrieval_eval import evaluate_config, DEFAULT_CHUNKS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "src" / "data" / "raw" / "egyptian_civil_code.pdf"


def _configs(paragraph_groups: list[int], windows: list[int], overlaps: list[int]) -> list[dict]:
    configs = [
        {"paragraphs_per_chunk": groups, "chunk_chars": None, "overlap_chars": 0}
        for groups in paragraph_groups
    ]
    for size in windows:
        for overlap in overlaps:
            if overlap >= size:
                continue
            configs.append(
                {"paragraphs_per_chunk": 1, "chunk_chars": size, "overlap_chars": overlap}
            )
    return configs


async def run_sweep(
    dataset: list[dict],
    articles: list[dict],
    model_name: str,
    top_k: int,
    configs: list[dict],
) -> list[dict]:
    results = []
    for config in configs:
        chunks = build_chunks(articles, **config)
        label = (
            f"window{config['chunk_chars']}+{config['overlap_chars']}"
            if config["chunk_chars"]
            else f"paragraphs{config['paragraphs_per_chunk']}"
        )
        print(f"evaluating chunking={label} chunks={len(chunks)} ...")
        try:
            metrics = await evaluate_config(
                dataset, chunks, model_name, top_k, collection=f"chunk_sweep_{label.replace('.', '_')}"
            )
        except Exception as error:
            print(f"  FAILED: {error}")
            continue
        _log_mlflow(metrics, extra_params={**config, "chunk_label": label, "chunk_count": len(chunks)})
        print(f"  {json.dumps(metrics)}")
        results.append({**metrics, "chunk_label": label, "chunk_count": len(chunks), **config})
    return results


def _log_mlflow(metrics: dict, extra_params: dict | None = None) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"chunks-{extra_params['chunk_label']}"):
            mlflow.log_params({"embedding_model": metrics["embedding_model"], **extra_params})
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            )
    except Exception as error:
        print(f"mlflow logging failed (best-effort): {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paragraph-groups", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--windows", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--overlaps", nargs="+", type=int, default=[0, 128])
    args = parser.parse_args()

    dataset = _load_jsonl(args.dataset)
    if not dataset:
        parser.error(f"Dataset is empty or missing: {args.dataset} — run `python -m eval.dataset` first")
    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")

    print(f"parsing {args.pdf.name} (once) ...")
    articles = _article_records(args.pdf)
    configs = _configs(args.paragraph_groups, args.windows, args.overlaps)
    print(f"{len(configs)} chunking configurations to evaluate")

    results = asyncio.run(run_sweep(dataset, articles, args.model, args.top_k, configs))
    if not results:
        print("No configurations evaluated successfully")
        return 1

    best = max(results, key=lambda item: item[f"mrr@{args.top_k}"])
    print("\nBest chunking configuration by MRR:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
