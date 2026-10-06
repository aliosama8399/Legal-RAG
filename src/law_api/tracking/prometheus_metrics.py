"""Prometheus instrumentation for the RAG pipeline.

`/api/v1/metrics` previously called `generate_latest()` with nothing registered,
so it only exposed the Python interpreter defaults (`python_gc_*`, `process_*`)
and a Grafana cost/hour panel was impossible. Every custom metric the
application records is defined here, once, so the /metrics endpoint, the
Prometheus alert rules and the Grafana dashboards all agree on the names.

Token and cost counters are what make "cost per hour" derivable in Grafana:
`rate(rag_llm_tokens_total{direction="completion"}[5m])` gives tokens/second, and
the cost gauge below converts that into USD/hour using the configured prices.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- request level -------------------------------------------------------
REQUESTS = Counter(
    "rag_requests_total",
    "RAG requests by endpoint and mode.",
    ["endpoint", "mode"],
)

ASK_LATENCY = Histogram(
    "rag_ask_latency_seconds",
    "End-to-end latency of POST /api/v1/ask.",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)

STREAM_LATENCY = Histogram(
    "rag_stream_latency_seconds",
    "End-to-end latency of POST /api/v1/ask/stream.",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)

FIRST_TOKEN_LATENCY = Histogram(
    "rag_first_token_latency_seconds",
    "Time from request start to the first streamed token (TTFT).",
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

# --- generation level ----------------------------------------------------
# `direction` is prompt|completion. Grafana derives cost/hour from this.
LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "LLM tokens consumed, by direction.",
    ["direction", "model"],
)

LLM_CALLS = Counter(
    "rag_llm_calls_total",
    "LLM completions requested, by model.",
    ["model"],
)

LLM_COST = Counter(
    "rag_llm_cost_usd_total",
    "Cumulative LLM spend in USD, derived from token counts and configured prices.",
    ["model"],
)

# USD per million tokens. Qwen2.5-1.5B-Instruct is effectively free on a local
# engine; override these to model a hosted provider.
PRICE_PROMPT_PER_MTOK = 0.0
PRICE_COMPLETION_PER_MTOK = 0.0


# --- retrieval level -----------------------------------------------------
RETRIEVAL_SOURCES = Histogram(
    "rag_retrieval_sources",
    "Number of source chunks returned per retrieval.",
    buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20),
)

RETRIEVAL_RERANKED = Counter(
    "rag_retrieval_reranked_total",
    "Retrievals that went through the cross-encoder reranker.",
)

EMPTY_ANSWERS = Counter(
    "rag_empty_answers_total",
    "Questions that retrieved no usable article and got the fallback answer.",
)

# --- guardrails ----------------------------------------------------------
PII_REDACTIONS = Counter(
    "rag_pii_redactions_total",
    "PII entities detected and redacted, by entity type.",
    ["entity_type"],
)

PII_SCANNED = Counter(
    "rag_pii_scanned_total",
    "Answers passed through PII detection.",
)

# --- drift ---------------------------------------------------------------
EMBEDDING_DRIFT = Gauge(
    "rag_embedding_drift_cosine",
    "Mean cosine similarity of recent query embeddings against the stored baseline. "
    "Falls as the query distribution moves away from the indexed corpus.",
)

# How many questions the current drift reading covers. The gauge above starts at
# 0, which is indistinguishable from "measured and terrible" — this lets a
# dashboard tell "no baseline yet" apart from a real number.
EMBEDDING_DRIFT_QUESTIONS = Gauge(
    "rag_embedding_drift_questions",
    "Questions covered by the current embedding-drift reading (0 = no baseline).",
)

# Drift is a DROP against the baseline's own mean, never an absolute similarity:
# real questions sit ~0.85 cosine from their own centroid with zero drift, so an
# absolute "< 0.90" rule alarms forever. Both series are needed to judge it.
EMBEDDING_DRIFT_BASELINE = Gauge(
    "rag_embedding_drift_baseline_cosine",
    "Mean cosine the baseline questions achieved against their own centroid.",
)

EMBEDDING_DRIFT_DELTA = Gauge(
    "rag_embedding_drift_delta",
    "Current mean cosine minus the baseline mean. Negative means drift.",
)

# --- live (per-request) evaluation ---------------------------------------
# Scored inline on the request path, so these move as traffic arrives rather
# than when a batch eval runs. `metric` is e.g. faithfulness / answer_relevancy /
# hit_at_k. The judge-backed ones cost a generation call per request, so they
# are opt-in; hit_at_k and reciprocal_rank are arithmetic and effectively free.
LIVE_EVAL_SCORE = Gauge(
    "rag_live_eval_score",
    "Most recent live score for one served request.",
    ["metric"],
)

LIVE_EVAL_DURATION = Histogram(
    "rag_live_eval_duration_seconds",
    "Wall-clock cost of live evaluation on the request path.",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

LIVE_EVAL_REQUESTS = Counter(
    "rag_live_eval_requests_total",
    "Live evaluations performed, by outcome.",
    ["outcome"],
)


def publish_live_eval(payload: dict) -> int:
    """Republish a live-eval payload into the gauges. Returns metrics published."""
    count = 0
    for key in ("hit_at_k", "reciprocal_rank", "faithfulness", "answer_relevancy", "context_relevancy"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            LIVE_EVAL_SCORE.labels(metric=key).set(float(value))
            count += 1
    return count


def record_live_eval(payload: dict, duration: float) -> None:
    """Publish one live evaluation to the gauges."""
    publish_live_eval(payload)
    LIVE_EVAL_DURATION.observe(duration)
    LIVE_EVAL_REQUESTS.labels(outcome="degraded" if payload.get("degraded") else "ok").inc()


# --- evaluation bridge ---------------------------------------------------
# Ragas scores live in MLflow, which Prometheus cannot scrape. The eval job
# writes data/eval/latest_scores.json; the bridge below republishes it as
# gauges so `rag_ragas_faithfulness < 0.8` can be a real alert rule.
RAGAS_METRIC = Gauge(
    "rag_ragas_metric",
    "Most recent Ragas score per metric, bridged from the eval job.",
    ["metric"],
)

RAGAS_QUESTIONS = Gauge(
    "rag_ragas_questions",
    "Number of questions in the most recent Ragas run.",
)

RAGAS_RUN_AGE = Gauge(
    "rag_ragas_run_age_seconds",
    "Seconds since the most recent Ragas run wrote its scores.",
)


def record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Record token usage for one LLM call and return its cost in USD."""
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    completion_tokens = max(int(completion_tokens or 0), 0)
    if not (prompt_tokens or completion_tokens):
        return 0.0
    LLM_TOKENS.labels(direction="prompt", model=model).inc(prompt_tokens)
    LLM_TOKENS.labels(direction="completion", model=model).inc(completion_tokens)
    cost = (
        prompt_tokens * PRICE_PROMPT_PER_MTOK + completion_tokens * PRICE_COMPLETION_PER_MTOK
    ) / 1_000_000
    # Always increment, even at zero. A local vLLM run has a real cost of $0, and
    # skipping the increment left the series non-existent, so the cost panel read
    # "No data" instead of the truthful "$0.00/hour".
    LLM_COST.labels(model=model).inc(cost)
    return cost


def publish_ragas_scores(payload: dict) -> set[str]:
    """Republish a `latest_scores.json` payload as gauges.

    Returns the metric names it published. Prometheus gauges keep their last
    value forever, so a deleted or stale scores file would otherwise leave a
    fabricated number on the dashboard and keep the alert firing.
    """
    scores = payload.get("scores") or {}
    for name, value in scores.items():
        if isinstance(value, (int, float)):
            RAGAS_METRIC.labels(metric=name).set(float(value))
    questions = payload.get("questions")
    if isinstance(questions, (int, float)):
        RAGAS_QUESTIONS.set(float(questions))
    return set(scores.keys())


def clear_ragas_scores(names: set[str]) -> None:
    """Drop previously published gauges by setting them to NaN.

    NaN renders as "No data" in Grafana and makes comparisons false, which is
    the honest state for "no evaluation result exists" — strictly better than a
    leftover value that looks like a measurement.
    """
    for name in names:
        RAGAS_METRIC.labels(metric=name).set(float("nan"))
    RAGAS_QUESTIONS.set(float("nan"))
