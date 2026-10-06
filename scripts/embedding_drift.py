"""Embedding drift: cosine similarity of query embeddings against a baseline.

The requirement is "cosine similarity on query embeddings vs baseline, drift
logged". `scripts/compare_embeddings.py` did not do this — it compared embedding
*throughput* between different models, and `Distance.COSINE` elsewhere in the
codebase is only index configuration, never an evaluation.

This script keeps a per-model baseline (the centroid of a reference question
set) and scores new questions against it:

    python -m scripts.embedding_drift --set-baseline
    python -m scripts.embedding_drift --questions data/eval/qa_dataset.jsonl

Results are logged to MLflow and written to data/eval/latest_drift.json, which
the application republishes as the `rag_embedding_drift_cosine` gauge so the
Prometheus rule can fire when the query distribution moves away from the corpus
the index was built for.
"""

import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASELINE_PATH = PROJECT_ROOT / "data" / "eval" / "embedding_baseline.json"
DRIFT_PATH = PROJECT_ROOT / "data" / "eval" / "latest_drift.json"

# Drift is a DROP against the baseline, not an absolute similarity. Individual
# questions sit around 0.85 cosine from their own centroid even with zero drift,
# so an absolute "< 0.90" rule would alarm forever. The alert fires when the
# current mean falls this far below the baseline's own mean.
DROP_TOLERANCE = 0.05


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot take the centroid of zero vectors")
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def fingerprint(questions: list[str]) -> str:
    """Stable id for a question set, so a baseline records what it covered."""
    joined = "\n".join(sorted(q.strip() for q in questions))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def load_questions(path: Path, limit: int = 0) -> list[str]:
    questions: list[str] = []
    if not path.is_file():
        raise SystemExit(f"questions file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question = record.get("question") or record.get("user_input")
        if question:
            questions.append(question)
        if limit and len(questions) >= limit:
            break
    if not questions:
        raise SystemExit(f"no questions found in {path}")
    return questions


async def embed_all(model_name: str, questions: list[str], batch: int = 32) -> list[list[float]]:
    from law_api.stores.embeddings.EmbeddingProviderFactory import EmbeddingProviderFactory

    provider = EmbeddingProviderFactory.create(model_name)
    vectors: list[list[float]] = []
    for start in range(0, len(questions), batch):
        vectors.extend(await provider.encode(questions[start : start + batch]))
    return vectors


def score(model_name: str, vectors: list[list[float]], baseline: dict) -> dict:
    """Mean/min/p5 cosine of each vector against the baseline centroid.

    `delta_vs_baseline` is the meaningful number: how far the current questions
    sit from the centroid compared with the baseline's own questions.
    """
    center = baseline["centroid"]
    scores = sorted(cosine(vector, center) for vector in vectors)
    count = len(scores)
    p5 = scores[max(0, math.ceil(0.05 * count) - 1)]
    mean = sum(scores) / count
    reference = baseline.get("baseline_mean_cosine")
    result = {
        "model": model_name,
        "questions": count,
        "mean_cosine": round(mean, 6),
        "min_cosine": round(scores[0], 6),
        "p5_cosine": round(p5, 6),
        "drop_tolerance": DROP_TOLERANCE,
        "baseline_mean_cosine": round(reference, 6) if isinstance(reference, (int, float)) else None,
    }
    if isinstance(reference, (int, float)):
        delta = mean - reference
        result["delta_vs_baseline"] = round(delta, 6)
        result["drift_detected"] = delta < -DROP_TOLERANCE
    else:
        # No reference yet: drift cannot be judged, and saying "no drift" would
        # be a claim we cannot support.
        result["delta_vs_baseline"] = None
        result["drift_detected"] = None
    return result


def log_to_mlflow(result: dict) -> None:
    try:
        import mlflow

        from law_api.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"drift-{result['model']}"):
            mlflow.set_tag("eval_phase", "drift")
            mlflow.log_param("embedding_model", result["model"])
            mlflow.log_metric("drift_mean_cosine", result["mean_cosine"])
            mlflow.log_metric("drift_min_cosine", result["min_cosine"])
            mlflow.log_metric("drift_p5_cosine", result["p5_cosine"])
            delta = result.get("delta_vs_baseline")
            if isinstance(delta, (int, float)):
                mlflow.log_metric("drift_delta_vs_baseline", delta)
        print("logged drift metrics to MLflow")
    except Exception as error:  # noqa: BLE001 - logging is best-effort
        print(f"mlflow logging skipped (best-effort): {error}")


def write_drift_file(result: dict) -> None:
    DRIFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"drift published to {DRIFT_PATH} (consumed by the Prometheus gauge)")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl")
    parser.add_argument("--model", default=None, help="Embedding model (default: settings.embedding_name)")
    parser.add_argument("--limit", type=int, default=0, help="Only use the first N questions")
    parser.add_argument("--set-baseline", action="store_true", help="(Re)create the baseline from these questions")
    args = parser.parse_args()

    from law_api.config import settings

    model_name = args.model or settings.embedding_name
    questions = load_questions(args.questions, args.limit)
    print(f"embedding {len(questions)} questions with {model_name} ...")
    vectors = await embed_all(model_name, questions)

    if args.set_baseline:
        # Score once against a provisional centroid so the baseline records its
        # own mean; that number is the reference future runs are compared to.
        provisional = {"centroid": centroid(vectors)}
        reference = score(model_name, vectors, provisional)
        baseline = {
            "model": model_name,
            "fingerprint": fingerprint(questions),
            "questions": len(questions),
            "dimension": len(vectors[0]),
            "centroid": provisional["centroid"],
            "baseline_mean_cosine": reference["mean_cosine"],
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline), encoding="utf-8")
        print(f"baseline written to {BASELINE_PATH}")
        print(
            f"  {len(questions)} questions, dim {baseline['dimension']}, "
            f"reference mean cosine {baseline['baseline_mean_cosine']}"
        )
        result = score(model_name, vectors, baseline)
        result["baseline_created"] = True
    else:
        if not BASELINE_PATH.is_file():
            raise SystemExit(
                f"no baseline at {BASELINE_PATH}. Create one first:\n"
                "  python -m scripts.embedding_drift --set-baseline"
            )
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if baseline.get("model") != model_name:
            print(
                f"warning: baseline was built with {baseline.get('model')} but scoring {model_name}; "
                "embeddings from different models are not comparable"
            )
        result = score(model_name, vectors, baseline)

    print(json.dumps(result, indent=2))
    log_to_mlflow(result)
    write_drift_file(result)
    if result["drift_detected"] is True:
        print(
            f"DRIFT: mean cosine fell {abs(result['delta_vs_baseline']):.4f} below the "
            f"baseline ({result['baseline_mean_cosine']}), past the "
            f"{result['drop_tolerance']} tolerance"
        )
    elif result["drift_detected"] is None:
        print("drift not judged: the baseline has no reference mean yet, re-run --set-baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
