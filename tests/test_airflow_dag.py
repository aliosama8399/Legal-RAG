"""Unit tests for the monthly + manual Apache Airflow evaluation DAG."""

from __future__ import annotations

import importlib.util
from pathlib import Path

DAG_PATH = Path(__file__).resolve().parents[1] / "dags" / "legal_rag_monthly_evaluation_dag.py"


def _load_dag_module():
    spec = importlib.util.spec_from_file_location("legal_rag_monthly_evaluation_dag", DAG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monthly_eval_dag_metadata_and_schedule():
    mod = _load_dag_module()
    dag = mod.dag
    assert dag.dag_id == "legal_rag_monthly_eval"
    assert dag.schedule == "0 0 1 * *"
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert set(dag.tags) >= {"legal-rag", "evaluation", "ragas", "mlops"}


def test_monthly_eval_dag_task_topology():
    mod = _load_dag_module()
    dag = mod.dag
    expected_order = [
        "verify_corpus_and_dataset",
        "compute_embedding_drift",
        "run_monthly_rag_evaluation",
        "verify_quality_gate",
    ]
    assert dag.task_ids == expected_order

    t1 = dag.get_task("verify_corpus_and_dataset")
    t2 = dag.get_task("compute_embedding_drift")
    t3 = dag.get_task("run_monthly_rag_evaluation")
    t4 = dag.get_task("verify_quality_gate")

    assert t2.task_id in t1.downstream_task_ids
    assert t3.task_id in t2.downstream_task_ids
    assert t4.task_id in t3.downstream_task_ids
    assert t4.downstream_task_ids == set()


def test_verify_corpus_and_quality_gate_pass_on_repo_artifacts():
    mod = _load_dag_module()
    corpus_info = mod.verify_corpus_and_dataset()
    assert corpus_info["chunks_count"] >= 1000
    assert corpus_info["qa_count"] >= 50

    gate_summary = mod.verify_quality_gate()
    assert gate_summary["passed"] is True
    assert gate_summary["failures"] == []
    assert gate_summary["drift_detected"] is False
