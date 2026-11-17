"""Apache Airflow DAG for automated monthly and manual Legal-RAG evaluation.

Schedule:
    Runs automatically on the 1st of every month at 00:00 UTC (``0 0 1 * *``),
    or manually via the Airflow UI ("Trigger DAG"), Airflow CLI
    (``airflow dags trigger legal_rag_monthly_eval``), or locally via:

        python dags/legal_rag_monthly_evaluation_dag.py --manual
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DAG_ID = "legal_rag_monthly_eval"
DEFAULT_SCHEDULE = os.getenv("AIRFLOW_EVAL_SCHEDULE", "0 0 1 * *")


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[1], Path("/app"), Path.cwd()]
    for root in candidates:
        if (root / "src").is_dir() and (root / "config").is_dir():
            return root
    return here.parents[1]


PROJECT_ROOT = _resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


try:  # pragma: no cover - used inside Apache Airflow container / runtime
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_INSTALLED = True
except ImportError:
    AIRFLOW_INSTALLED = False

    _ACTIVE_DAG: DAG | None = None

    class _LocalTask:
        """Lightweight task node mirroring Airflow's PythonOperator API for host tests."""

        def __init__(
            self,
            *,
            task_id: str,
            python_callable: Callable[..., Any],
            dag: DAG | None = None,
            op_kwargs: dict[str, Any] | None = None,
            **_: Any,
        ) -> None:
            resolved_dag = dag or _ACTIVE_DAG
            if resolved_dag is None:
                raise ValueError("PythonOperator requires an active DAG context or explicit dag=")
            self.task_id = task_id
            self.python_callable = python_callable
            self.dag = resolved_dag
            self.op_kwargs = dict(op_kwargs or {})
            self.upstream_task_ids: set[str] = set()
            self.downstream_task_ids: set[str] = set()
            resolved_dag._register_task(self)

        def __rshift__(self, other: _LocalTask) -> _LocalTask:
            self.downstream_task_ids.add(other.task_id)
            other.upstream_task_ids.add(self.task_id)
            return other

        def execute(self, context: dict[str, Any] | None = None) -> Any:
            kwargs = {**self.op_kwargs, **(context or {})}
            return self.python_callable(**kwargs)

    class PythonOperator(_LocalTask):
        """Fallback PythonOperator when Apache Airflow is not installed locally."""

    class DAG:
        """Lightweight DAG shim exposing Airflow-compatible metadata and topology."""

        def __init__(
            self,
            dag_id: str,
            *,
            schedule: str | None = None,
            schedule_interval: str | None = None,
            start_date: datetime | None = None,
            catchup: bool = False,
            max_active_runs: int = 1,
            default_args: dict[str, Any] | None = None,
            tags: list[str] | None = None,
            description: str = "",
            is_paused_upon_creation: bool = False,
            **_: Any,
        ) -> None:
            self.dag_id = dag_id
            self.schedule = schedule if schedule is not None else schedule_interval
            self.schedule_interval = self.schedule
            self.start_date = start_date
            self.catchup = catchup
            self.max_active_runs = max_active_runs
            self.default_args = dict(default_args or {})
            self.tags = list(tags or [])
            self.description = description
            self.is_paused_upon_creation = is_paused_upon_creation
            self.task_dict: dict[str, PythonOperator] = {}

        def _register_task(self, task: PythonOperator) -> None:
            self.task_dict[task.task_id] = task

        @property
        def tasks(self) -> list[PythonOperator]:
            return list(self.task_dict.values())

        @property
        def task_ids(self) -> list[str]:
            return list(self.task_dict.keys())

        def get_task(self, task_id: str) -> PythonOperator:
            return self.task_dict[task_id]

        def __enter__(self) -> DAG:
            global _ACTIVE_DAG
            self._previous_dag = _ACTIVE_DAG
            _ACTIVE_DAG = self
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            global _ACTIVE_DAG
            _ACTIVE_DAG = getattr(self, "_previous_dag", None)
            return None


def _eval_python() -> str:
    configured = os.getenv("EVAL_PYTHON_BIN", "").strip()
    if configured and Path(configured).exists():
        return configured
    for candidate in [
        Path("/opt/eval-venv/bin/python"),
        Path("/opt/airflow/eval-venv/bin/python"),
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    roots = [str(SRC_ROOT), str(PROJECT_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join([*roots, existing] if existing else roots)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Inside container / Airflow environment, loopback 127.0.0.1:8001 cannot reach vLLM service.
    # Route container calls to internal network service endpoint `http://vllm:8000/v1`.
    is_container = Path("/.dockerenv").exists() or bool(os.getenv("AIRFLOW_HOME"))
    if is_container:
        gen_url = env.get("LAW_API_EVAL_GENERATION_BASE_URL", "")
        if not gen_url or "127.0.0.1:8001" in gen_url or "localhost:8001" in gen_url:
            env["LAW_API_EVAL_GENERATION_BASE_URL"] = "http://vllm:8000/v1"

        judge_url = env.get("LAW_API_EVAL_JUDGE_BASE_URL", "")
        if not judge_url or "127.0.0.1:8001" in judge_url or "localhost:8001" in judge_url:
            env["LAW_API_EVAL_JUDGE_BASE_URL"] = "http://vllm:8000/v1"
    return env


def verify_corpus_and_dataset(**context: Any) -> dict[str, Any]:
    """Ensure the canonical Civil Code chunks and curated QA dataset exist before eval."""
    del context
    pdf_path = PROJECT_ROOT / "src" / "data" / "raw" / "egyptian_civil_code.pdf"
    processed_dir = PROJECT_ROOT / "src" / "data" / "processed"
    chunks_path = processed_dir / "law_chunks.jsonl"
    qa_path = PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl"

    if not chunks_path.is_file():
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Neither chunks ({chunks_path}) nor source PDF ({pdf_path}) exist."
            )
        subprocess.run(
            [
                _eval_python(),
                str(PROJECT_ROOT / "src" / "data" / "prepare_law.py"),
                str(pdf_path),
                "--output-dir",
                str(processed_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=_subprocess_env(),
            check=True,
        )

    if not qa_path.is_file():
        raise FileNotFoundError(f"Curated QA dataset not found at {qa_path}")

    chunks_count = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
    qa_count = sum(1 for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if chunks_count == 0 or qa_count == 0:
        raise ValueError(f"Empty corpus/dataset (chunks={chunks_count}, qa={qa_count})")

    return {
        "chunks_path": str(chunks_path),
        "chunks_count": chunks_count,
        "qa_path": str(qa_path),
        "qa_count": qa_count,
    }


def compute_embedding_drift(**context: Any) -> dict[str, Any]:
    """Measure cosine embedding drift on qa_dataset.jsonl against the reference baseline."""
    del context
    qa_path = PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl"
    baseline_path = PROJECT_ROOT / "data" / "eval" / "embedding_baseline.json"
    drift_path = PROJECT_ROOT / "data" / "eval" / "latest_drift.json"

    cmd = [
        _eval_python(),
        "-m",
        "scripts.embedding_drift",
        "--questions",
        str(qa_path),
    ]
    if baseline_path.is_file():
        try:
            baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_data.get("model"):
                cmd.extend(["--model", str(baseline_data["model"])])
        except Exception:
            pass
    else:
        cmd.append("--set-baseline")

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_subprocess_env(), check=True)
    if not drift_path.is_file():
        raise FileNotFoundError(f"Drift output missing after run: {drift_path}")
    return json.loads(drift_path.read_text(encoding="utf-8"))


def run_monthly_rag_evaluation(**context: Any) -> dict[str, Any]:
    """Run the monthly retrieval + Ragas evaluation suite (supports manual dag_run.conf overrides)."""
    dag_run = context.get("dag_run")
    conf: dict[str, Any] = (getattr(dag_run, "conf", None) or {}) if dag_run else {}
    conf = {**conf, **(context.get("manual_conf") or {})}

    cmd = [_eval_python(), "-m", "eval.run_evaluation", "--skip-dataset"]
    # In monthly evaluation, evaluate canonical chunks rather than running expensive 12-config sweeps.
    if conf.get("skip_chunks", True):
        cmd.append("--skip-chunks")
    if conf.get("skip_retrieval"):
        cmd.append("--skip-retrieval")
    if conf.get("skip_ragas"):
        cmd.append("--skip-ragas")
    limit = conf.get("limit") or os.getenv("AIRFLOW_EVAL_LIMIT")
    if limit:
        cmd.extend(["--limit", str(int(limit))])
    if conf.get("models"):
        cmd.extend(["--models", *[str(m) for m in conf["models"]]])
    else:
        try:
            from law_api.config import settings as app_settings
            cmd.extend(["--models", app_settings.embedding_name])
        except Exception:
            pass
    if conf.get("top_k"):
        cmd.extend(["--top-k", *[str(int(k)) for k in conf["top_k"]]])
    ragas_metrics = conf.get("ragas_metrics") or (
        os.getenv("AIRFLOW_RAGAS_METRICS", "").split() if os.getenv("AIRFLOW_RAGAS_METRICS") else None
    )
    if ragas_metrics:
        cmd.extend(["--ragas-metrics", *[str(m) for m in ragas_metrics]])

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_subprocess_env(), check=True)

    scores_path = PROJECT_ROOT / "data" / "eval" / "latest_scores.json"
    report_path = PROJECT_ROOT / "data" / "eval" / "evaluation_report.md"
    return {
        "scores_exists": scores_path.is_file(),
        "report_exists": report_path.is_file(),
        "report_path": str(report_path),
    }


def verify_quality_gate(**context: Any) -> dict[str, Any]:
    """Enforce production faithfulness, retrieval Hit@5, and embedding drift thresholds."""
    dag_run = context.get("dag_run")
    conf: dict[str, Any] = (getattr(dag_run, "conf", None) or {}) if dag_run else {}
    conf = {**conf, **(context.get("manual_conf") or {})}

    scores_path = PROJECT_ROOT / "data" / "eval" / "latest_scores.json"
    drift_path = PROJECT_ROOT / "data" / "eval" / "latest_drift.json"

    if not scores_path.is_file():
        raise FileNotFoundError(f"Evaluation scores missing at {scores_path}")

    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    scores = scores_payload.get("scores", {})
    drift_payload = (
        json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.is_file() else {}
    )

    min_faithfulness = float(conf.get("min_faithfulness", os.getenv("EVAL_MIN_FAITHFULNESS", "0.80")))
    min_hit_rate = float(conf.get("min_hit_rate", os.getenv("EVAL_MIN_HIT_RATE_AT_5", "0.70")))

    failures: list[str] = []
    faithfulness = scores.get("faithfulness")
    if isinstance(faithfulness, int | float) and float(faithfulness) < min_faithfulness:
        failures.append(f"faithfulness={faithfulness:.4f} < {min_faithfulness:.2f}")

    hit_rate = scores.get("hybrid_hit_at_5", scores.get("hit_rate@5"))
    if isinstance(hit_rate, int | float) and float(hit_rate) < min_hit_rate:
        failures.append(f"hit_at_5={hit_rate:.4f} < {min_hit_rate:.2f}")

    drift_detected = bool(drift_payload.get("drift_detected", False))
    if drift_detected and conf.get("check_drift", True):
        failures.append(
            f"embedding_drift_detected (delta={drift_payload.get('delta_vs_baseline')})"
        )

    summary = {
        "passed": len(failures) == 0,
        "failures": failures,
        "questions": scores_payload.get("questions", 0),
        "scores": scores,
        "drift_detected": drift_detected,
    }
    if failures and conf.get("strict_gate", True):
        raise RuntimeError(f"Monthly Legal-RAG quality gate failed: {'; '.join(failures)}")
    return summary


with DAG(
    dag_id=DAG_ID,
    description="Automated monthly (and manual) Legal-RAG drift check, evaluation sweep, and quality gate",
    schedule=DEFAULT_SCHEDULE,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["legal-rag", "evaluation", "ragas", "mlops"],
) as dag:
    t_verify_corpus = PythonOperator(
        task_id="verify_corpus_and_dataset",
        python_callable=verify_corpus_and_dataset,
    )
    t_compute_drift = PythonOperator(
        task_id="compute_embedding_drift",
        python_callable=compute_embedding_drift,
    )
    t_run_eval = PythonOperator(
        task_id="run_monthly_rag_evaluation",
        python_callable=run_monthly_rag_evaluation,
    )
    t_quality_gate = PythonOperator(
        task_id="verify_quality_gate",
        python_callable=verify_quality_gate,
    )

    t_verify_corpus >> t_compute_drift >> t_run_eval >> t_quality_gate


def _main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or manually run the monthly Airflow eval DAG.")
    parser.add_argument("--dry-run", action="store_true", help="Print DAG schedule and topological task graph.")
    parser.add_argument("--manual", action="store_true", help="Execute the DAG tasks sequentially on the host.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    if args.dry_run or not args.manual:
        print(
            json.dumps(
                {
                    "dag_id": dag.dag_id,
                    "schedule": getattr(dag, "schedule_interval", None)
                    or getattr(dag, "schedule", None)
                    or str(getattr(getattr(dag, "timetable", None), "summary", "")),
                    "catchup": dag.catchup,
                    "max_active_runs": dag.max_active_runs,
                    "airflow_installed": AIRFLOW_INSTALLED,
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "downstream": sorted(t.downstream_task_ids),
                        }
                        for t in dag.tasks
                    ],
                },
                indent=2,
            )
        )
        return 0

    manual_conf = {
        "limit": args.limit or None,
        "skip_chunks": args.skip_chunks,
        "skip_ragas": args.skip_ragas,
    }
    print("Running verify_corpus_and_dataset ...")
    print(json.dumps(verify_corpus_and_dataset(), indent=2))
    print("Running compute_embedding_drift ...")
    print(json.dumps(compute_embedding_drift(), indent=2))
    print("Running run_monthly_rag_evaluation ...")
    print(json.dumps(run_monthly_rag_evaluation(manual_conf=manual_conf), indent=2))
    print("Running verify_quality_gate ...")
    print(json.dumps(verify_quality_gate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
