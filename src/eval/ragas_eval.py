"""Ragas quality evaluation of the GENERATION model over the QA dataset.

The generation model (vLLM, LAW_API_EVAL_GENERATION_* — the same backend
serving /ask) is the model under test. The judge model
(LAW_API_EVAL_JUDGE_* on a second vLLM) only scores its answers with the
six metrics. Scores are attributed to the generation model + RAG config
(embedding model, chunking, top_k), never to the judge.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall,
factual_correctness, noise_sensitivity.

Run:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --model bge-m3 --top-k 5
    python -m eval.ragas_eval --limit 2                      # quick smoke test
    python -m eval.ragas_eval --metrics faithfulness         # cheapest single metric
    python -m eval.ragas_eval --metrics faithfulness answer_relevancy context_recall
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
from datetime import UTC, datetime  # noqa: F401 - kept for eval run naming
from pathlib import Path

from openai import AsyncOpenAI

from law_api.config import settings
from law_api.stores.embeddings.EmbeddingProviderFactory import EmbeddingProviderFactory

from .dataset import DATASET_PATH, _load_jsonl
from .retrieval_eval import DEFAULT_CHUNKS, _build_index

SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer only using the provided "
    "Egyptian Civil Code articles and cite the article number for every claim. "
    "If the answer is not contained in the provided articles, say so."
)

# Short names for the Ragas metrics so a run can select a cheap subset. The
# full six is the default in the docs but not affordable on a 6 GB card: the
# judge is the same 1.5B engine as generation, and every metric is several long
# structured generations per question.
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "noise_sensitivity",
)
# The four canonical Ragas quality metrics for a RAG pipeline. This is the
# default because the requirement is "all 4 metrics logged" on >= 50 questions.
# `context_precision` and `context_recall` are the expensive ones: each is
# several long structured generations per question, and the judge is the same
# 1.5B engine as generation on a 6 GB card. Expect ~2 h for 56 questions at
# max_workers=2; use `--limit` while iterating and run the full set unattended.
DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _build_metrics(names):
    from ragas import metrics as M

    constructors = {
        "faithfulness": M.Faithfulness,
        "answer_relevancy": M.AnswerRelevancy,
        "context_precision": M.LLMContextPrecisionWithReference,
        "context_recall": M.LLMContextRecall,
        "factual_correctness": M.FactualCorrectness,
        "noise_sensitivity": M.NoiseSensitivity,
    }
    unknown = [name for name in names if name not in constructors]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; choose from {sorted(constructors)}")
    return [constructors[name]() for name in names]


async def _run_pipeline(
    dataset: list[dict],
    chunks: list[dict],
    model_name: str,
    top_k: int,
    concurrency: int = 8,
    langfuse_client=None,
) -> list[dict]:
    """Retrieval + generation over the dataset; returns ragas sample rows.

    When a Langfuse client is supplied, every question becomes its own trace
    that appears while the run is still going: a `retriever` observation for the
    Qdrant hits and a `generation` observation for the vLLM call. The
    generation observation is emitted by the official langfuse.openai wrapper,
    so the model name and token usage are captured automatically.
    """
    client, _ = await _build_index(chunks, model_name, "ragas_eval")
    provider = EmbeddingProviderFactory.create(model_name)
    # The Langfuse OpenAI wrapper is a drop-in AsyncOpenAI: it traces each call
    # as a generation observation when tracing is on, and behaves identically
    # when it is off or unavailable.
    generation_class = AsyncOpenAI
    if langfuse_client is not None:
        try:
            from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

            generation_class = LangfuseAsyncOpenAI
        except Exception:
            generation_class = AsyncOpenAI
    generation = generation_class(
        api_key="not-needed", base_url=settings.eval_generation_base_url
    )
    # vLLM batches concurrent requests, so this is much faster than one at a time.
    gate = asyncio.Semaphore(concurrency)
    done = 0
    trace_span = (
        langfuse_client.start_as_current_observation
        if langfuse_client is not None
        else None
    )

    async def one(item: dict) -> dict:
        nonlocal done
        question = item["question"]
        trace_id = ""
        async with gate:
            query_vector = (await provider.encode([question]))[0]
            response = await client.query_points(
                collection_name="ragas_eval", query=query_vector, limit=top_k
            )
            contexts = [point.payload.get("chunk_text", "") for point in response.points]
            context = "\n\n".join(contexts)
            prompt = f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"
            # One trace per question, opened BEFORE the generation call so it is
            # visible in the UI while vLLM is still working.
            if trace_span is None:
                answer = await _answer(generation, prompt)
            else:
                with trace_span(
                    as_type="span",
                    name="ragas-question",
                    input={"question": question, "top_k": top_k},
                    metadata={
                        "embedding_model": model_name,
                        "generation_model": settings.eval_generation_model,
                    },
                ) as span:
                    trace_id = (span.trace_id if span is not None else "") or ""
                    with trace_span(
                        as_type="retriever",
                        name="vector-search",
                        input={"question": question, "top_k": top_k},
                    ) as retriever:
                        retriever.update(
                            output={
                                "sources": [
                                    point.payload.get("citation", "")
                                    for point in response.points
                                ],
                                "returned": len(contexts),
                            }
                        )
                    answer = await _answer(generation, prompt)
                    if span is not None:
                        span.update(
                            output={
                                "answer": answer,
                                "reference": item["ground_truth_answer"],
                            }
                        )
        done += 1
        print(f"generated {done}/{len(dataset)}", flush=True)
        return {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth_answer"],
            # Carried so the Ragas scores can be attached to THIS trace later.
            "trace_id": trace_id,
        }

    try:
        return list(await asyncio.gather(*(one(item) for item in dataset)))
    finally:
        await client.close()


async def _answer(generation, prompt: str) -> str:
    response = await generation.chat.completions.create(
        model=settings.eval_generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _judge_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.eval_judge_model,
            base_url=settings.eval_judge_base_url,
            api_key="not-needed",
            temperature=0,
            # A 1.5B judge on an eager (no CUDA-graph) vLLM engine takes minutes
            # for the structured judge outputs, which outlived the library's
            # client default and killed the run with Job TimeoutError. Cap it off
            # explicitly and let retries absorb a transient failure.
            timeout=None,
            max_retries=3,
        )
    )


def _eval_embeddings():
    # Reuse the retrieval embedder (already cached) instead of a LangChain HuggingFace class.
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.run_config import RunConfig

    provider = EmbeddingProviderFactory.create(settings.embedding_name)

    class _AppEmbeddings(BaseRagasEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self.run_config = RunConfig()

        def embed_query(self, text: str) -> list[float]:
            return provider._encode_sync([text])[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return provider._encode_sync(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return (await provider.encode([text]))[0]

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await provider.encode(texts)

    return _AppEmbeddings()


def _langfuse_client():
    """One Langfuse client for the whole run, or None when keys are unset."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        print("langfuse keys not configured — skipping trace (best-effort)")
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            # v4 spells the endpoint `base_url`; `host` is the legacy alias.
            base_url=settings.langfuse_host,
        )
        # Wrong keys/host otherwise fail silently and the UI just stays empty.
        client.auth_check()
        expected = settings.langfuse_project_name
        if expected:
            names = [project.name for project in client.api.projects.get().data]
            if expected not in names:
                print(f"langfuse keys belong to project {names}, not '{expected}' - not logging")
                return None
        return client
    except Exception as error:  # noqa: BLE001 - tracing is best-effort
        print(f"langfuse unavailable at {settings.langfuse_host} (check host and keys): {error}")
        return None


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _aggregate(per_sample: list[dict]) -> dict:
    """Mean per metric over the questions where the judge produced a valid score."""
    scores: dict = {}
    for name in per_sample[0] if per_sample else []:
        values = [row[name] for row in per_sample if _finite(row.get(name))]
        if values:
            scores[name] = sum(values) / len(values)
        else:
            print(f"ragas: metric '{name}' returned no valid score (judge output unparsable)")
    return scores


def _ensure_dataset(client, rows: list[dict]) -> None:
    """Keep the QA set in Langfuse so it can be re-run as an experiment later."""
    dataset_name = "legal-rag-qa"
    try:
        client.create_dataset(
            name=dataset_name, description="Egyptian Civil Code QA evaluation set"
        )
    except Exception:  # noqa: BLE001 - already exists
        pass
    for row in rows:
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                id="qa-" + hashlib.sha1(row["user_input"].encode("utf-8")).hexdigest()[:24],
                input={"question": row["user_input"]},
                expected_output=row["reference"],
            )
        except Exception:  # noqa: BLE001 - item already exists
            pass


def _attach_scores(client, rows: list[dict], per_sample: list[dict]) -> int:
    """Attach the Ragas scores to the traces the generation run already created.

    The trace per question was emitted live by _run_pipeline (retriever +
    generation). Creating a second, dataset-run trace for the same question
    would duplicate it in the UI, so the scores are written onto the existing
    trace ids instead.
    """
    attached = 0
    for row, sample in zip(rows, per_sample):
        trace_id = row.get("trace_id")
        if not trace_id:
            continue
        for name, value in sample.items():
            if not _finite(value):
                continue
            try:
                client.create_score(
                    trace_id=trace_id, name=name, value=float(value), data_type="NUMERIC"
                )
                attached += 1
            except Exception as error:  # noqa: BLE001 - scoring is best-effort
                print(f"langfuse score '{name}' failed (best-effort): {error}")
    return attached


def run_ragas_eval(
    rows: list[dict], params: dict | None = None, client=None, metric_names=None
) -> dict:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.run_config import RunConfig

    metrics_to_run = list(metric_names or DEFAULT_METRICS)
    if client is None:
        client = _langfuse_client()
    print(
        f"ragas: scoring {len(rows)} answers with {len(metrics_to_run)} metrics "
        f"({', '.join(metrics_to_run)}) - judge calls, can take a while ...",
        flush=True,
    )
    # trace_id is ours, not a ragas field; keep it out of the dataset.
    ragas_rows = [
        {key: value for key, value in row.items() if key != "trace_id"} for row in rows
    ]
    result = evaluate(
        dataset=EvaluationDataset.from_list(ragas_rows),
        metrics=_build_metrics(metrics_to_run),
        llm=_judge_llm(),
        embeddings=_eval_embeddings(),
        # Two judge requests at a time, not eight. The judge is the same eager
        # vLLM engine as generation (~15-20 tok/s), so eight concurrent requests
        # starve each other and every one of them can blow a 300 s timeout.
        run_config=RunConfig(max_workers=2, timeout=900),
        raise_exceptions=False,
    )
    per_sample = [{str(k): v for k, v in sample.items()} for sample in result.scores]
    scores = _aggregate(per_sample)
    if client is not None:
        try:
            _ensure_dataset(client, rows)
            attached = _attach_scores(client, rows, per_sample)
            print(f"langfuse: attached {attached} scores to the per-question traces")
        except Exception as error:  # noqa: BLE001 - tracing is best-effort
            print(f"langfuse scoring failed (best-effort): {error}")
        finally:
            client.flush()
    _log_mlflow(scores, params or {}, rows, per_sample)
    _publish_scores(scores, params or {}, rows)
    return scores


RAGAS_EXPERIMENT = "legal-rag-ragas-eval"


def _publish_scores(scores: dict, params: dict, rows: list[dict]) -> None:
    """Write the newest scores where the app's Prometheus bridge will read them.

    Ragas lives in MLflow, which Prometheus cannot scrape, so without this file
    an alert on `rag_ragas_faithfulness < 0.8` has no metric behind it.
    """
    try:
        path = settings.eval_scores_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scores": scores,
            "params": {key: str(value) for key, value in (params or {}).items()},
            "questions": len(rows),
            "written_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"scores published to {path} for the Prometheus alert bridge")
    except Exception as error:  # noqa: BLE001 - publishing is best-effort
        print(f"could not publish scores for alerting (best-effort): {error}")


def _safe_name(name: str) -> str:
    # Ragas names like "factual_correctness(mode=f1)" are invalid MLflow metric keys.
    return re.sub(r"[^0-9A-Za-z_\-\. :/]", "_", name)


def _log_mlflow(scores: dict, params: dict, rows: list[dict], per_sample: list[dict]) -> None:
    try:
        import mlflow
        import pandas as pd
        from mlflow.entities import AssessmentSource, AssessmentSourceType

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(RAGAS_EXPERIMENT)
        # This tag is what files the experiment under the UI's GenAI section.
        mlflow.MlflowClient().set_experiment_tag(
            experiment.experiment_id, "mlflow.experimentKind", "genai_development"
        )
        judge = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id=settings.eval_judge_model
        )
        with mlflow.start_run(run_name=f"ragas-{params.get('generation_model', 'model').split('/')[-1]}"):
            mlflow.set_tag("eval_phase", "ragas")
            mlflow.set_tag("mlflow.runNotes", "Ragas quality of the GENERATION model (judge scores only)")
            mlflow.log_params(params)
            mlflow.log_metrics({_safe_name(name): value for name, value in scores.items()})
            table = pd.DataFrame(
                [
                    {
                        "question": row["user_input"],
                        "answer": row["response"],
                        "reference": row["reference"],
                        "contexts": "\n---\n".join(row["retrieved_contexts"]),
                        **{_safe_name(k): v for k, v in sample.items()},
                    }
                    for row, sample in zip(rows, per_sample)
                ]
            )
            mlflow.log_table(table, artifact_file="eval_results_table.json")
            trace_ids = []
            for row in rows:
                with mlflow.start_span(name="qa") as span:
                    span.set_inputs({"question": row["user_input"], "contexts": row["retrieved_contexts"]})
                    span.set_outputs({"answer": row["response"], "reference": row["reference"]})
                trace_ids.append(span.trace_id)
            # Traces are written asynchronously; feedback needs them to exist first.
            mlflow.flush_trace_async_logging()
            for trace_id, sample in zip(trace_ids, per_sample):
                for name, value in sample.items():
                    if _finite(value):
                        mlflow.log_feedback(
                            trace_id=trace_id, name=_safe_name(name), value=float(value), source=judge
                        )
    except Exception as error:  # noqa: BLE001 - tracking is best-effort
        print(f"mlflow logging failed (best-effort): {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--model", default=None, help="Embedding model for retrieval (default: settings.embedding_name)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N questions (quick test)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=list(METRIC_NAMES),
        help=(
            "Ragas metrics to run. Defaults to the four canonical quality "
            "metrics; pass --limit while iterating, the full set on 56 "
            "questions takes hours because the judge is the generation engine."
        ),
    )
    args = parser.parse_args()

    dataset = _load_jsonl(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    if not dataset:
        parser.error(f"Dataset is empty or missing: {args.dataset} — run `python -m eval.dataset` first")
    chunks = _load_jsonl(args.chunks)
    if not chunks:
        parser.error(f"Chunks file is empty or missing: {args.chunks}")

    model_name = args.model or settings.embedding_name
    print(
        f"evaluating generation model={settings.eval_generation_model} "
        f"(judge={settings.eval_judge_model}) embedding={model_name} top_k={args.top_k} ..."
    )
    # The client is created BEFORE generation so each question is traced live,
    # and reused afterwards so the Ragas scores land on those same traces.
    client = _langfuse_client()
    if client is None:
        print("langfuse off - scores will only go to MLflow")
    rows = asyncio.run(
        _run_pipeline(dataset, chunks, model_name, args.top_k, langfuse_client=client)
    )
    if not rows:
        parser.error("Pipeline produced no answers — are the vLLM servers running?")

    params = {
        "generation_model": settings.eval_generation_model,
        "judge_model": settings.eval_judge_model,
        "embedding_model": model_name,
        "top_k": args.top_k,
        "questions": len(rows),
    }
    scores = run_ragas_eval(rows, params, client=client, metric_names=args.metrics)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
