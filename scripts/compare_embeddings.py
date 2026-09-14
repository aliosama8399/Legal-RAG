"""Compare the configured Arabic embedding models with MLflow.

Examples:
    uv run python scripts/compare_embeddings.py
    uv run python scripts/compare_embeddings.py --models bge-m3 gate-arabert-v1
    uv run python scripts/compare_embeddings.py --limit 100
    uv run python scripts/compare_embeddings.py --pdf src/data/raw/egyptian_civil_code.pdf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from law_api.config import settings
from law_api.providers.embeddings.model_provider import EMBEDDING_MODELS, create_embedding_provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "src" / "data" / "processed" / "law_chunks.jsonl"
DEFAULT_PDF = PROJECT_ROOT / "src" / "data" / "raw" / "egyptian_civil_code.pdf"


def read_chunks(path: Path, limit: int | None) -> list[dict]:
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return chunks[:limit] if limit else chunks


def prepare_chunks(pdf_path: Path) -> Path:
    from data.prepare_law import prepare

    _, chunks_path = prepare(pdf_path, pdf_path.parent / "processed")
    return chunks_path


def run_experiment(model_name: str, chunks: list[dict], batch_size: int) -> dict:
    import mlflow

    provider = create_embedding_provider(model_name)
    texts = [chunk["chunk_text"] for chunk in chunks]
    started = time.perf_counter()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(provider.encode(texts[start : start + batch_size]))
    elapsed = time.perf_counter() - started
    dimension = len(vectors[0]) if vectors else 0
    mlflow.log_params(
        {
            "embedding_alias": model_name,
            "embedding_model_id": provider.model_id,
            "chunk_count": len(texts),
            "batch_size": batch_size,
            "normalized_embeddings": True,
        }
    )
    mlflow.log_metrics(
        {
            "embedding_dimension": dimension,
            "embedding_seconds": elapsed,
            "chunks_per_second": len(texts) / elapsed if elapsed else 0.0,
            "embedded_chunks": len(vectors),
        }
    )
    return {
        "model": model_name,
        "model_id": provider.model_id,
        "chunks": len(vectors),
        "dimension": dimension,
        "seconds": round(elapsed, 3),
        "chunks_per_second": round(len(vectors) / elapsed, 2) if elapsed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EMBEDDING_MODELS),
        default=sorted(EMBEDDING_MODELS),
        help="Embedding aliases to compare",
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--experiment", default="embedding-model-comparison")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")

    chunks_path = args.chunks
    if args.pdf:
        chunks_path = prepare_chunks(args.pdf)
    if not chunks_path.is_file():
        parser.error(f"Chunks file does not exist: {chunks_path}")

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    chunks = read_chunks(chunks_path, args.limit)
    if not chunks:
        parser.error("No chunks were found")

    print(f"chunks={len(chunks)} experiment={args.experiment} tracking_uri={args.tracking_uri}")
    for model_name in args.models:
        with mlflow.start_run(run_name=f"{model_name}-comparison"):
            result = run_experiment(model_name, chunks, args.batch_size)
            mlflow.set_tag("dataset", str(chunks_path))
            print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
