"""Retrieval evaluation harness: HitRate@k / MRR@k / Recall@k against gold articles.

Sweeps embedding models x top_k values, builds a temporary in-memory Qdrant
index per configuration (chunks embedded with the candidate model), evaluates,
and logs every run to MLflow.

Run:
    python -m eval.retrieval_eval --dataset data/eval/qa_dataset.jsonl
    python -m eval.retrieval_eval --models bge-m3 gate-arabert-v1 --top-k 1 3 5 10
"""

import argparse
import json
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models

from law_api.config import settings
from law_api.stores.embeddings.EmbeddingProviderFactory import EmbeddingProviderFactory

from .dataset import DATASET_PATH, _load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = PROJECT_ROOT / "src" / "data" / "processed" / "law_chunks.jsonl"


async def _build_index(chunks: list[dict], model_name: str, collection: str) -> tuple[AsyncQdrantClient, int]:
    """Embed all chunks with the candidate model into a temp in-memory collection."""
    from uuid import NAMESPACE_URL, uuid5

    provider = EmbeddingProviderFactory.create(model_name)
    texts = [chunk["chunk_text"] for chunk in chunks]
    vectors: list[list[float]] = []
    batch = 64
    for start in range(0, len(texts), batch):
        vectors.extend(await provider.encode(texts[start : start + batch]))
    dimension = len(vectors[0]) if vectors else 0
    if not dimension:
        raise ValueError("Embedding produced an empty vector set")

    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
            vector=vector,
            payload={**chunk},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    await client.upsert(collection_name=collection, points=points)
    return client, dimension


def _retrieval_metrics(ranks: list[int], top_k: int, total_queries: int) -> dict:
    """ranks: 1-based rank of the first gold-article hit per query (0 = miss)."""
    hits = [rank for rank in ranks if rank > 0]
    hit_rate = len(hits) / total_queries if total_queries else 0.0
    mrr = sum(1.0 / rank for rank in hits) / total_queries if total_queries else 0.0
    recall = len(hits) / total_queries if total_queries else 0.0  # single gold article per query
    return {
        f"hit_rate@{top_k}": round(hit_rate, 4),
        f"mrr@{top_k}": round(mrr, 4),
        f"recall@{top_k}": round(recall, 4),
        "total_queries": total_queries,
    }


async def evaluate_config(
    dataset: list[dict],
    chunks: list[dict],
    model_name: str,
    top_k: int,
    collection: str = "retrieval_eval",
) -> dict:
    """Evaluate one (embedding model, top_k) configuration."""
    client, dimension = await _build_index(chunks, model_name, collection)
    try:
        provider = EmbeddingProviderFactory.create(model_name)
        questions = [item["question"] for item in dataset]
        query_vectors = []
        batch = 32
        for start in range(0, len(questions), batch):
            query_vectors.extend(await provider.encode(questions[start : start + batch]))

        ranks: list[int] = []
        for item, query_vector in zip(dataset, query_vectors):
            gold = int(item["ground_truth_article"])
            response = await client.query_points(
                collection_name=collection, query=query_vector, limit=top_k
            )
            rank = 0
            for position, point in enumerate(response.points, start=1):
                if int(point.payload.get("article_number", -1)) == gold:
                    rank = position
                    break
            ranks.append(rank)
        metrics = _retrieval_metrics(ranks, top_k, len(dataset))
        metrics["embedding_model"] = model_name
        metrics["embedding_dimension"] = dimension
        return metrics
    finally:
        await client.close()


def _log_mlflow(metrics: dict, extra_params: dict | None = None) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"retrieval-{metrics['embedding_model']}"):
            mlflow.set_tag("eval_phase", "retrieval")
            params = {"embedding_model": metrics["embedding_model"], "top_k": metrics.get("top_k", "")}
            if extra_params:
                params.update(extra_params)
            mlflow.log_params(params)
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            )
    except Exception as error:
        print(f"mlflow logging failed (best-effort): {error}")


def _log_langfuse(metrics: dict) -> None:
    """Publish one retrieval result to Langfuse as a trace carrying its scores.

    Retrieval makes no LLM call, so there is no generation to trace — but the
    sweep is still an evaluation, and writing it as scores means the embedding
    comparison is visible in Langfuse next to the Ragas traces, not only in
    MLflow.
    """
    from .ragas_eval import _langfuse_client

    client = _langfuse_client()
    if client is None:
        return
    model_name = metrics.get("embedding_model", "unknown")
    top_k = metrics.get("top_k", "")
    scored = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key not in ("total_queries",)
    }
    try:
        with client.start_as_current_observation(
            as_type="span",
            name="retrieval-eval",
            input={"embedding_model": model_name, "top_k": top_k},
            metadata={"total_queries": str(metrics.get("total_queries", ""))},
        ) as span:
            for name, value in scored.items():
                if span is not None:
                    span.score(name=name, value=value)
            if span is not None:
                span.update(output=scored)
        client.flush()
    except Exception as error:  # noqa: BLE001 - tracing is best-effort
        print(f"langfuse logging failed (best-effort): {error}")


async def run_sweep(
    dataset: list[dict],
    chunks: list[dict],
    model_names: list[str],
    top_k_values: list[int],
) -> list[dict]:
    """Evaluate all (model, top_k) combinations; log each to MLflow + Langfuse."""
    results = []
    for model_name in model_names:
        for top_k in top_k_values:
            print(f"evaluating {model_name} top_k={top_k} ...")
            try:
                metrics = await evaluate_config(dataset, chunks, model_name, top_k)
            except Exception as error:
                print(f"  FAILED: {error}")
                continue
            _log_mlflow(metrics)
            _log_langfuse(metrics)
            print(f"  {json.dumps(metrics)}")
            results.append(metrics)
    return results


def main() -> int:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gate-arabert-v1", "arabert-all-nli-triplet-matryoshka", "bge-m3"],
        default=["gate-arabert-v1", "arabert-all-nli-triplet-matryoshka", "bge-m3"],
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()

    dataset = _load_jsonl(args.dataset)
    if not dataset:
        parser.error(f"Dataset is empty or missing: {args.dataset} — run `python -m eval.dataset` first")
    chunks = _load_jsonl(args.chunks)
    if not chunks:
        parser.error(f"Chunks file is empty or missing: {args.chunks}")

    results = asyncio.run(run_sweep(dataset, chunks, args.models, args.top_k))
    if not results:
        print("No configurations evaluated successfully")
        return 1

    best = max(results, key=lambda item: item["mrr@5" if "mrr@5" in item else list(item)[0]])
    print("\nBest configuration by MRR:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
