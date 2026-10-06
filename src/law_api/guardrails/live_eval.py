"""Live evaluation of a served answer.

The batch pipeline (`eval.ragas_eval`) evaluates the retrieval model and the
generation model offline. This module does the same thing for a single served
request, so a regression can be seen on the request that caused it instead of
hours later in a batch run.

Two halves, deliberately separated by cost:

* **Retrieval scoring** is arithmetic — hit@k and reciprocal rank against a gold
  article number. No LLM, no GPU, sub-millisecond. When the caller supplies
  `expected_article` this is the only thing needed to evaluate the *retrieval*
  model in production.
* **Answer scoring** is LLM-as-judge. It costs a real generation call, so it is
  opt-in per request (`evaluate=true`) and defaults to faithfulness plus answer
  relevancy. `faithfulness` is the graded metric.

Why this runs inline rather than in the background: BentoML mounts the FastAPI
app as an ASGI sub-app on a per-request event loop, so a background task started
during a request never survives to run again. Scores are therefore computed
before the response is returned, and the latency is reported in
`live_eval.duration_seconds` so the cost is visible rather than hidden.
"""

import asyncio
import json
import re
from time import perf_counter

# Answer-side metrics this module can run, with the ragas classes behind them.
SUPPORTED_METRICS = ("faithfulness", "answer_relevancy", "context_relevancy")
DEFAULT_METRICS = ("faithfulness", "answer_relevancy")

_ARTICLE_IN_TEXT = re.compile(r"Article\s+(\d+)")


def article_numbers(text: str) -> set[int]:
    """Article numbers mentioned in free text, e.g. "Article 147" or "Articles 5 and 7"."""
    found: set[int] = set()
    for match in _ARTICLE_IN_TEXT.finditer(text or ""):
        try:
            found.add(int(match.group(1)))
        except ValueError:
            continue
    # "Articles 5 and 7" / "Articles 5, 6 and 7" -> catch the bare numbers too.
    for clause in re.split(r"\band\b|,", text or ""):
        stripped = clause.strip()
        if not stripped.lower().startswith(("article", "articles")):
            continue
        for number in re.findall(r"\d+", stripped):
            try:
                found.add(int(number))
            except ValueError:
                continue
    return found


def score_retrieval(expected_article: int, results: list[dict], top_k: int) -> dict:
    """hit@k and reciprocal rank for the gold article. No LLM involved."""
    limit = top_k or len(results) or 1
    ranks: list[int] = []
    for position, chunk in enumerate(results[:limit], start=1):
        numbers = article_numbers(chunk.get("citation", ""))
        if not numbers:
            numbers = {chunk["article_number"]} if chunk.get("article_number") is not None else set()
        if expected_article in numbers:
            ranks.append(position)
    reciprocal = 1.0 / min(ranks) if ranks else 0.0
    return {
        "hit_at_k": 1.0 if ranks else 0.0,
        "reciprocal_rank": round(reciprocal, 6),
        "retrieved_ranks": ranks,
    }


def _judge_prompt(question: str, answer: str, contexts: list[str], metric: str) -> str:
    context_block = "\n\n".join(contexts[:8])
    if metric == "faithfulness":
        return (
            "You are grading a RAG answer against the law articles it was given.\n\n"
            f"ARTICLES:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            "Decide whether EVERY claim in the answer is supported by the articles. "
            "Ignore stylistic issues and any correct legal background.\n"
            'Reply with JSON only: {"verdict": "yes"|"no", "reason": "<one sentence>"}'
        )
    if metric == "answer_relevancy":
        return (
            "You are grading how well an answer addresses the question asked.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
            "Reply with JSON only: {\"verdict\": \"yes\"|\"no\", \"reason\": \"<one sentence>\"} "
            "where yes means the answer actually addresses the question."
        )
    return (
        "You are grading whether the retrieved law articles are relevant to the question.\n\n"
        f"QUESTION: {question}\n\nARTICLES:\n{context_block}\n\n"
        'Reply with JSON only: {"verdict": "yes"|"no", "reason": "<one sentence>"}'
    )


def _parse_verdict(raw: str) -> float | None:
    """Turn a judge reply into 1.0/0.0. Small models are chatty, so be tolerant."""
    if not raw:
        return None
    text = raw.strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            verdict = str(payload.get("verdict", "")).strip().lower()
            if verdict in ("yes", "true", "1"):
                return 1.0
            if verdict in ("no", "false", "0"):
                return 0.0
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    if lowered.startswith("yes") or '"yes"' in lowered:
        return 1.0
    if lowered.startswith("no") or '"no"' in lowered:
        return 0.0
    return None


async def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    metrics: tuple[str, ...],
    base_url: str,
    model: str,
    timeout: float = 120.0,
) -> tuple[dict[str, float], str | None]:
    """Score one answer with the LLM judge. Returns (scores, degraded_reason)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="not-needed", base_url=base_url, timeout=timeout)
    scores: dict[str, float] = {}
    degraded: str | None = None
    try:
        for metric in metrics:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict grader. Reply with JSON only.",
                        },
                        {
                            "role": "user",
                            "content": _judge_prompt(question, answer, contexts, metric),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                value = _parse_verdict(response.choices[0].message.content or "")
                if value is None:
                    degraded = f"judge output unparsable for {metric}"
                else:
                    scores[metric] = value
            except Exception as error:
                degraded = f"{metric}: {type(error).__name__}"
    finally:
        await client.close()
    return scores, degraded


async def evaluate_live(
    *,
    question: str,
    answer: str,
    results: list[dict],
    top_k: int,
    expected_article: int | None,
    metrics: tuple[str, ...] | None,
    judge_base_url: str,
    judge_model: str,
) -> dict:
    """Full live evaluation of one served request."""
    started = perf_counter()
    payload: dict = {"hit_at_k": None, "reciprocal_rank": None, "retrieved_ranks": []}

    if expected_article is not None:
        payload.update(score_retrieval(expected_article, results, top_k))

    if metrics:
        contexts = [chunk.get("chunk_text", "") for chunk in results]
        scores, degraded = await judge_answer(
            question, answer, contexts, metrics, judge_base_url, judge_model
        )
        payload.update(scores)
        if degraded:
            payload["degraded"] = degraded

    payload["judge_model"] = judge_model if metrics else None
    payload["duration_seconds"] = round(perf_counter() - started, 3)
    return payload
