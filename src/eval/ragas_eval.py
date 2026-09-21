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
"""

import asyncio
import json
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


async def _run_pipeline(dataset: list[dict], chunks: list[dict], model_name: str, top_k: int) -> list[dict]:
    """Retrieval + generation over the dataset; returns ragas sample rows."""
    client, _ = await _build_index(chunks, model_name, "ragas_eval")
    provider = EmbeddingProviderFactory.create(model_name)
    generation = AsyncOpenAI(api_key="not-needed", base_url=settings.eval_generation_base_url)
    rows: list[dict] = []
    try:
        for item in dataset:
            question = item["question"]
            query_vector = (await provider.encode([question]))[0]
            response = await client.query_points(
                collection_name="ragas_eval", query=query_vector, limit=top_k
            )
            contexts = [point.payload.get("chunk_text", "") for point in response.points]
            context = "\n\n".join(contexts)
            answer = (
                await generation.chat.completions.create(
                    model=settings.eval_generation_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Articles:\n{context}\n\nQuestion: {question}\nAnswer:"},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
            ).choices[0].message.content or ""
            rows.append(
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": item["ground_truth_answer"],
                }
            )
    finally:
        await client.close()
    return rows


def _judge_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.eval_judge_model,
            base_url=settings.eval_judge_base_url,
            api_key="not-needed",
            temperature=0,
        )
    )


def _eval_embeddings():
    from ragas.embeddings import HuggingfaceEmbeddings

    return HuggingfaceEmbeddings(model_name="BAAI/bge-m3")


def run_ragas_eval(rows: list[dict]) -> dict:
    from ragas import evaluate
    from ragas import metrics as M
    from ragas.dataset_schema import EvaluationDataset

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[
            M.Faithfulness(),
            M.AnswerRelevancy(),
            M.LLMContextPrecisionWithReference(),
            M.LLMContextRecall(),
            M.FactualCorrectness(),
            M.NoiseSensitivity(),
        ],
        llm=_judge_llm(),
        embeddings=_eval_embeddings(),
        raise_exceptions=False,
    )
    return {str(key): float(value) for key, value in dict(result).items() if isinstance(value, (int, float))}


def _log_mlflow(scores: dict, params: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"ragas-{params['generation_model'].split('/')[-1]}"):
            mlflow.log_params(params)
            mlflow.log_metrics(scores)
    except Exception as error:
        print(f"mlflow logging failed (best-effort): {error}")


def _push_langfuse(scores: dict, params: dict) -> None:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        print("langfuse keys not configured — skipping (best-effort)")
        return
    try:
        from langfuse import Langfuse

        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        with lf.start_as_current_observation(name="ragas-eval", input=params) as observation:
            for name, score in scores.items():
                observation.score(name=name, value=score)
            observation.update(output=scores)
        lf.flush()
    except Exception as error:
        print(f"langfuse push failed (best-effort): {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--model", default=None, help="Embedding model for retrieval (default: settings.embedding_name)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = _load_jsonl(args.dataset)
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
    rows = asyncio.run(_run_pipeline(dataset, chunks, model_name, args.top_k))
    if not rows:
        parser.error("Pipeline produced no answers — are the vLLM servers running?")

    scores = run_ragas_eval(rows)
    params = {
        "generation_model": settings.eval_generation_model,
        "judge_model": settings.eval_judge_model,
        "embedding_model": model_name,
        "top_k": args.top_k,
        "questions": len(rows),
    }
    _log_mlflow(scores, params)
    _push_langfuse(scores, params)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
