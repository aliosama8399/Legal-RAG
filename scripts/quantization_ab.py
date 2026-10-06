"""A/B the Ragas faithfulness of two generative-model configurations.

The requirement is "RAGAS scores: original vs quantized — <= 0.03 faithfulness
drop". Nothing compared two generation models before; `_log_mlflow` could record
`generation_model` per run but no harness drove two runs or compared them.

This compares two score files produced by two separate `eval.ragas_eval` runs
(one per configuration), so the expensive judge work happens in two ordinary
evaluation runs and this script only does the comparison and the gate:

    # terminal 1 - FP16
    python -m eval.ragas_eval --metrics faithfulness

    # switch .env to the 4-bit settings, restart vllm, then:
    python -m eval.ragas_eval --metrics faithfulness

    python -m scripts.quantization_ab --baseline data/eval/scores_fp16.json \
        --candidate data/eval/scores_nf4.json
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "quantization_ab.json"

# The graded gate: quantization may not cost more than this much faithfulness.
MAX_FAITHFULNESS_DROP = 0.03


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"score file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "scores" not in payload:
        raise SystemExit(f"{path} has no 'scores' object — pass latest_scores.json format")
    return payload


def faithfulness(payload: dict) -> float:
    scores = payload["scores"]
    for key in ("faithfulness", "faithfulness(mode=f1)"):
        if key in scores:
            return float(scores[key])
    raise SystemExit(f"no faithfulness metric in {sorted(scores)}")


def log_to_mlflow(baseline: float, candidate: float, drop: float, labels: dict) -> None:
    try:
        import mlflow

        from law_api.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name="quantization-ab"):
            mlflow.set_tag("eval_phase", "quantization_ab")
            mlflow.log_params({key: str(value) for key, value in labels.items()})
            mlflow.log_metric("faithfulness_baseline", baseline)
            mlflow.log_metric("faithfulness_candidate", candidate)
            mlflow.log_metric("faithfulness_drop", drop)
        print("logged comparison to MLflow")
    except Exception as error:  # noqa: BLE001 - logging is best-effort
        print(f"mlflow logging skipped (best-effort): {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="scores from the FP16 run")
    parser.add_argument("--candidate", type=Path, required=True, help="scores from the quantized run")
    parser.add_argument("--max-drop", type=float, default=MAX_FAITHFULNESS_DROP)
    args = parser.parse_args()

    baseline_payload = load(args.baseline)
    candidate_payload = load(args.candidate)
    baseline = faithfulness(baseline_payload)
    candidate = faithfulness(candidate_payload)

    # Drop is baseline - candidate: positive means quantization hurt.
    drop = baseline - candidate
    passed = drop <= args.max_drop

    labels = {
        "baseline_model": baseline_payload.get("params", {}).get("generation_model", "unknown"),
        "candidate_model": candidate_payload.get("params", {}).get("generation_model", "unknown"),
        "baseline_questions": baseline_payload.get("questions"),
        "candidate_questions": candidate_payload.get("questions"),
        "max_drop": args.max_drop,
    }

    print(json.dumps({**labels, "faithfulness_baseline": baseline,
                      "faithfulness_candidate": candidate, "faithfulness_drop": round(drop, 4),
                      "verdict": "PASS" if passed else "FAIL"}, indent=2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                **labels,
                "faithfulness_baseline": baseline,
                "faithfulness_candidate": candidate,
                "faithfulness_drop": round(drop, 4),
                "verdict": "PASS" if passed else "FAIL",
                "compared_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_to_mlflow(baseline, candidate, drop, labels)

    if not passed:
        print(
            f"FAIL: faithfulness dropped {drop:.4f}, above the {args.max_drop} budget. "
            "Either raise the tolerance with evidence or keep the FP16 model."
        )
        return 1
    print(f"PASS: faithfulness drop {drop:.4f} is within the {args.max_drop} budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
