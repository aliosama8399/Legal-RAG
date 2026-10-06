"""Bridges the Ragas evaluation result into Prometheus gauges.

Ragas scores are produced by the one-shot eval job and land in MLflow, which
Prometheus cannot scrape. Without a bridge, the requirement "alert when
faithfulness < 0.80" is unimplementable as a Prometheus rule.

The eval job writes ``data/eval/latest_scores.json``; this module republishes it
as gauges so ``rag_ragas_metric{metric="faithfulness"} < 0.8`` is a real,
alertable expression. A file is used rather than a Pushgateway because it needs
no extra container.

Refresh happens **synchronously inside the /metrics handler**, not in a
background task. BentoML mounts the FastAPI app as an ASGI sub-app and appears
to drive it on a per-request event loop, so an ``asyncio.create_task`` started
during a request never survives to run again — the task-based version silently
published nothing. Reading one small file per scrape is a few hundred
microseconds and is deterministic instead.
"""

import json
from pathlib import Path

from .prometheus_metrics import (
    EMBEDDING_DRIFT,
    EMBEDDING_DRIFT_BASELINE,
    EMBEDDING_DRIFT_DELTA,
    EMBEDDING_DRIFT_QUESTIONS,
    clear_ragas_scores,
    publish_live_eval,
    publish_ragas_scores,
)

_warned: set[str] = set()
# Metric names published by the last successful read, so they can be reset to
# NaN once the scores file goes away.
_published: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(message)


def _read_json(path: Path) -> dict | None:
    """Read JSON, tolerating a UTF-8 BOM.

    utf-8-sig strips the BOM when present and behaves as plain utf-8 otherwise.
    A BOM makes json.loads raise outright, and when that error was swallowed the
    gauges sat at zero while looking perfectly healthy — the exact failure this
    bridge exists to prevent.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        _warn_once(f"json:{path.name}", f"eval bridge: {path.name} is not valid JSON ({error})")
        return None
    except OSError:
        return None


def refresh_eval_gauges(
    scores_path: Path,
    drift_path: Path | None = None,
    live_path: Path | None = None,
) -> dict[str, str]:
    """Publish the newest scores/drift/live-eval into the Prometheus gauges.

    Safe to call on every scrape: a missing or unreadable file leaves the
    previous gauge values in place. Returns a small status dict, surfaced as a
    response header by /metrics, so "gauges missing" can be told apart from
    "bridge never ran" without attaching a debugger.

    The live-eval file is what makes the Grafana live panels survive a restart:
    every counter and gauge inside the process is lost when the container is
    recreated, so without it those panels read zero until the next request.
    """
    status = {
        "scores": "missing",
        "published": "0",
        "live": "missing",
        # Reported separately because it is the usual culprit: BentoML serves
        # from a worker whose CWD is the bento's src dir, so a relative path
        # resolves somewhere that does not exist.
        "abs_exists": str(scores_path.is_file()),
    }

    global _published
    published: list[str] = []
    try:
        if scores_path.is_file():
            status["scores"] = "found"
            payload = _read_json(scores_path)
            if payload is not None:
                _published = publish_ragas_scores(payload)
                published = sorted(_published)
                status["published"] = str(len(published))
        elif _published:
            # No scores file: blank the gauges rather than leaving a stale
            # number that looks like a real measurement.
            clear_ragas_scores(_published)
            _published = set()
            status["published"] = "0"
    except Exception as error:
        status["scores"] = f"error:{type(error).__name__}"
        _warn_once("scores", f"eval bridge: could not read {scores_path} ({error})")

    if drift_path is not None:
        try:
            if drift_path.is_file():
                drift = _read_json(drift_path)
                if drift is not None:
                    value = drift.get("mean_cosine")
                    if isinstance(value, (int, float)):
                        EMBEDDING_DRIFT.set(float(value))
                    covered = drift.get("questions")
                    if isinstance(covered, (int, float)):
                        EMBEDDING_DRIFT_QUESTIONS.set(float(covered))
                reference = drift.get("baseline_mean_cosine")
                if isinstance(reference, (int, float)):
                    EMBEDDING_DRIFT_BASELINE.set(float(reference))
                delta = drift.get("delta_vs_baseline")
                if isinstance(delta, (int, float)):
                    EMBEDDING_DRIFT_DELTA.set(float(delta))
        except Exception as error:
            _warn_once("drift", f"eval bridge: could not read {drift_path} ({error})")

    if live_path is not None:
        try:
            if live_path.is_file():
                live = _read_json(live_path)
                if live is not None:
                    status["live"] = f"found:{publish_live_eval(live)}"
        except Exception as error:
            _warn_once("live", f"eval bridge: could not read {live_path} ({error})")

    # One line per distinct set of published metrics. Silent success is
    # indistinguishable from "the bridge never runs", which is exactly how the
    # task-based version failed unnoticed.
    if published:
        _warn_once(
            "published:" + ",".join(published),
            f"eval bridge: published {len(published)} ragas metric(s): {', '.join(published)}",
        )
    return status
