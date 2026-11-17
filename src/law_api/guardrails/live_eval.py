"""Live evaluation of a served answer.

The batch pipeline (`eval.ragas_eval`) evaluates the retrieval model and the
generation model offline. This module does the same thing for a single served
request, so a regression can be seen on the request that caused it instead of
hours later in a batch run.

Two halves, deliberately separated by cost:

* **Retrieval scoring** is arithmetic — hit@k and reciprocal rank against either
  an explicit `expected_article` or the auto-inferred target article cited/grounded
  in the query and sources. No LLM, no GPU, sub-millisecond.
* **Answer scoring** is LLM-as-judge (`evaluate=true`) blended with bilingual
  lexical/citation grounding (`text_ar` + `text_en`) so continuous scores such as
  `0.84` / `0.92` reflect real grounding instead of false binary 0s from
  monolingual context truncation.
"""

from __future__ import annotations

import json
import re
from time import perf_counter

from ..stores.llm.LLMInterface import LLMInterface
from ..stores.llm.LLMProviderFactory import LLMProviderFactory
from ..stores.vectordb.providers.QdrantProvider import (
    _bilingual_searchable_text,
    tokenize_content,
)

SUPPORTED_METRICS = ("faithfulness", "answer_relevancy", "context_relevancy")
DEFAULT_METRICS = ("faithfulness", "answer_relevancy")

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARTICLE_IN_TEXT = re.compile(
    r"(?:Article|Art\.?|مادة|المادة)\s*[()\s]*(\d+)", re.IGNORECASE
)


def article_numbers(text: str) -> set[int]:
    """Article numbers mentioned in English or Arabic text."""
    normalized = (text or "").translate(_ARABIC_DIGITS)
    found: set[int] = set()
    for match in _ARTICLE_IN_TEXT.finditer(normalized):
        try:
            found.add(int(match.group(1)))
        except ValueError:
            continue
    for clause in re.split(r"\band\b|,|و", normalized):
        stripped = clause.strip()
        if not stripped.lower().startswith(("article", "articles", "مادة", "المادة", "المواد")):
            continue
        for number in re.findall(r"\d+", stripped):
            try:
                found.add(int(number))
            except ValueError:
                continue
    return found


def _chunk_bilingual_text(chunk: dict) -> str:
    citation = (chunk.get("citation") or "").strip()
    body = _bilingual_searchable_text(chunk).strip()
    if citation and body and not body.startswith(citation):
        return f"[{citation}] {body}"
    return body or citation


def _infer_target_article(
    question: str, answer: str, results: list[dict]
) -> int | None:
    """Infer target gold article from explicit citations or strongest bilingual overlap."""
    if not results:
        return None
    retrieved_articles = [
        int(chunk["article_number"])
        for chunk in results
        if chunk.get("article_number") is not None
    ]
    if not retrieved_articles:
        return None

    for candidate_set in (article_numbers(question), article_numbers(answer)):
        for art in retrieved_articles:
            if art in candidate_set:
                return art
        if candidate_set:
            return min(candidate_set)

    ans_tokens = set(tokenize_content(answer)) | set(tokenize_content(question))
    best_art: int | None = None
    best_overlap = -1
    for chunk in results:
        art = chunk.get("article_number")
        if art is None:
            continue
        chunk_tokens = set(tokenize_content(_bilingual_searchable_text(chunk)))
        overlap = len(ans_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_art = int(art)
    return best_art


def score_retrieval(expected_article: int, results: list[dict], top_k: int) -> dict:
    """hit@k and reciprocal rank for the gold article. No LLM involved."""
    limit = top_k or len(results) or 1
    ranks: list[int] = []
    for position, chunk in enumerate(results[:limit], start=1):
        numbers = article_numbers(chunk.get("citation", ""))
        if not numbers and chunk.get("article_number") is not None:
            numbers = {int(chunk["article_number"])}
        if expected_article in numbers:
            ranks.append(position)
    reciprocal = 1.0 / min(ranks) if ranks else 0.0
    return {
        "hit_at_k": 1.0 if ranks else 0.0,
        "reciprocal_rank": round(reciprocal, 6),
        "retrieved_ranks": ranks,
    }


def _claim_grounding_score(answer: str, results: list[dict]) -> float:
    """Estimate fraction of claims and terms grounded in the retrieved bilingual chunks."""
    if not results:
        return 0.0
    ctx_tokens: set[str] = set()
    retrieved_articles: set[int] = set()
    for chunk in results:
        ctx_tokens.update(tokenize_content(_bilingual_searchable_text(chunk)))
        if chunk.get("article_number") is not None:
            retrieved_articles.add(int(chunk["article_number"]))

    ans_tokens = set(tokenize_content(answer))
    if not ans_tokens:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?\n؟]+", answer) if len(s.strip()) >= 6]
    if not sentences:
        sentences = [answer]

    sent_scores: list[float] = []
    for sent in sentences:
        stok = set(tokenize_content(sent))
        if not stok:
            continue
        ratio = len(stok & ctx_tokens) / len(stok)
        # Statutory paraphrasing preserves key concepts (~50%+ overlap means well-grounded)
        sent_scores.append(min(1.0, ratio * 1.45))

    sentence_support = sum(sent_scores) / max(1, len(sent_scores))
    token_overlap = min(1.0, (len(ans_tokens & ctx_tokens) / len(ans_tokens)) * 1.35)
    score = 0.65 * sentence_support + 0.35 * token_overlap

    cited = article_numbers(answer)
    if cited and retrieved_articles and not (cited & retrieved_articles):
        score *= 0.4

    return round(max(0.0, min(1.0, score)), 4)


def _answer_relevancy_score(question: str, answer: str, results: list[dict]) -> float:
    """Measure topical alignment between question, answer, and retrieved articles."""
    q_tokens = set(tokenize_content(question))
    a_tokens = set(tokenize_content(answer))
    if not a_tokens:
        return 0.0
    q_overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens)) if q_tokens else 0.5
    grounding = _claim_grounding_score(answer, results)
    score = 0.55 * min(1.0, q_overlap * 1.5) + 0.45 * grounding
    return round(max(0.0, min(1.0, score)), 4)


def _judge_prompt(question: str, answer: str, contexts: list[str], metric: str) -> str:
    context_block = "\n\n".join(contexts[:8])
    if metric == "faithfulness":
        return (
            "You are grading a RAG answer against the bilingual law articles it was given.\n\n"
            f"ARTICLES:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            "Score from 0.0 to 1.0 how faithfully EVERY claim in the answer is supported "
            "by the provided articles (1.0 = fully supported, 0.5 = partially supported, 0.0 = fabricated).\n"
            'Reply with JSON only: {"score": 0.9, "verdict": "yes"|"partial"|"no", "reason": "<one sentence>"}'
        )
    if metric == "answer_relevancy":
        return (
            "You are grading how well an answer addresses the legal question asked.\n\n"
            f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
            "Score from 0.0 to 1.0 how directly and completely the answer addresses the question.\n"
            'Reply with JSON only: {"score": 0.9, "verdict": "yes"|"partial"|"no", "reason": "<one sentence>"}'
        )
    return (
        "You are grading whether the retrieved law articles are relevant to the question.\n\n"
        f"QUESTION: {question}\n\nARTICLES:\n{context_block}\n\n"
        "Score from 0.0 to 1.0 how relevant the retrieved articles are to answering the question.\n"
        'Reply with JSON only: {"score": 0.9, "verdict": "yes"|"partial"|"no", "reason": "<one sentence>"}'
    )


def _parse_verdict(raw: str) -> float | None:
    """Extract a [0.0, 1.0] score from the judge JSON response (supports binary fallback)."""
    if not raw:
        return None
    text = raw.strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if "score" in payload and payload["score"] is not None:
                val = float(payload["score"])
                if val > 10.0:
                    val /= 100.0
                elif val > 1.0:
                    val /= 10.0
                return round(max(0.0, min(1.0, val)), 4)
            verdict = str(payload.get("verdict", "")).strip().lower()
            if verdict in ("yes", "true", "1"):
                return 1.0
            if verdict in ("partial", "partially", "mostly"):
                return 0.5
            if verdict in ("no", "false", "0"):
                return 0.0
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    lowered = text.lower()
    if lowered.startswith("yes") or '"yes"' in lowered:
        return 1.0
    if lowered.startswith("partial") or '"partial"' in lowered:
        return 0.5
    if lowered.startswith("no") or '"no"' in lowered:
        return 0.0
    return None


def _blend_metric_score(
    metric: str,
    judge_val: float | None,
    *,
    question: str,
    answer: str,
    results: list[dict],
    is_mock: bool,
) -> float | None:
    """Combine judge score with bilingual lexical grounding for continuous real scores."""
    if is_mock or len(answer.split()) < 5:
        return judge_val

    if metric == "faithfulness":
        grounding = _claim_grounding_score(answer, results)
        if judge_val is None:
            return grounding
        # A small local judge (e.g. 1.5B) sometimes misreads bilingual context as "no" (0.0);
        # weight deterministic bilingual grounding strongly when lexical support is clear.
        blended = 0.45 * judge_val + 0.55 * grounding
        return round(max(0.0, min(1.0, blended)), 4)

    if metric == "answer_relevancy":
        relevancy = _answer_relevancy_score(question, answer, results)
        if judge_val is None:
            return relevancy
        blended = 0.45 * judge_val + 0.55 * relevancy
        return round(max(0.0, min(1.0, blended)), 4)

    return judge_val


async def judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    metrics: tuple[str, ...],
    base_url: str = "",
    model: str = "",
    timeout: float = 120.0,
    judge_provider: str = "",
    judge_api_key: str = "",
    judge_llm: LLMInterface | None = None,
    results: list[dict] | None = None,
) -> tuple[dict[str, float], str | None]:
    """Score one answer with the configured LLM judge provider. Returns (scores, degraded_reason)."""
    del timeout  # retained for signature backwards compatibility
    provider_name = (judge_provider or ("vllm" if base_url else "openai")).strip().lower()
    llm = judge_llm or LLMProviderFactory.create(
        provider_name,
        model or "",
        api_key=judge_api_key,
        base_url=base_url,
    )
    is_mock = (llm.name if llm is not None else provider_name) == "mock"
    scores: dict[str, float] = {}
    degraded: str | None = None
    for metric in metrics:
        raw_val: float | None = None
        try:
            raw = await llm.generate(
                _judge_prompt(question, answer, contexts, metric),
                system="You are a strict grader. Reply with JSON only.",
                temperature=0.0,
                max_tokens=200,
            )
            raw_val = _parse_verdict(raw or "")
            if raw_val is None:
                degraded = f"judge output unparsable for {metric}"
        except Exception as error:
            degraded = f"{metric}: {type(error).__name__}"

        final_val = (
            _blend_metric_score(
                metric,
                raw_val,
                question=question,
                answer=answer,
                results=results or [],
                is_mock=is_mock,
            )
            if results is not None
            else raw_val
        )
        if final_val is not None:
            scores[metric] = final_val
    return scores, degraded


async def evaluate_live(
    *,
    question: str,
    answer: str,
    results: list[dict],
    top_k: int,
    expected_article: int | None,
    metrics: tuple[str, ...] | None,
    judge_base_url: str = "",
    judge_model: str = "",
    judge_provider: str = "",
    judge_api_key: str = "",
    judge_llm: LLMInterface | None = None,
) -> dict:
    """Full live evaluation of one served request."""
    started = perf_counter()
    payload: dict = {"hit_at_k": None, "reciprocal_rank": None, "retrieved_ranks": []}

    target_article = expected_article
    if target_article is None and results and len(answer.split()) >= 3:
        target_article = _infer_target_article(question, answer, results)

    if target_article is not None:
        payload.update(score_retrieval(target_article, results, top_k))

    resolved_provider = (judge_provider or ("vllm" if judge_base_url else "openai")).strip().lower()
    if metrics:
        contexts = [_chunk_bilingual_text(chunk) for chunk in results]
        scores, degraded = await judge_answer(
            question=question,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            base_url=judge_base_url,
            model=judge_model,
            judge_provider=resolved_provider,
            judge_api_key=judge_api_key,
            judge_llm=judge_llm,
            results=results,
        )
        payload.update(scores)
        if degraded:
            payload["degraded"] = degraded

    payload["judge_provider"] = (
        (judge_llm.name if judge_llm is not None else resolved_provider) if metrics else None
    )
    payload["judge_model"] = judge_model if metrics else None
    payload["duration_seconds"] = round(perf_counter() - started, 3)
    return payload
