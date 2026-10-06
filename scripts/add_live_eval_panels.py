"""Add the live-evaluation panels to the Grafana dashboard.

Run after changing monitoring/grafana/dashboards/rag-observability.json by hand:

    python scripts/add_live_eval_panels.py

Kept as a script because editing the dashboard JSON by hand is error-prone, and
the file is a single large object that Grafana rewrites on save.
"""

import json
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards" / "rag-observability.json"

LIVE_PANELS = [
    {
        "type": "stat",
        "title": "Live faithfulness (last request)",
        "description": "Per-request judge score from `evaluate: true` on /ask. Empty until a request asks for live evaluation.",
        "gridPos": {"h": 6, "w": 6, "x": 0, "y": 30},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "fieldConfig": {
            "defaults": {
                "unit": "none",
                "decimals": 3,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "orange", "value": 0.8},
                        {"color": "green", "value": 1},
                    ],
                },
            },
            "overrides": [],
        },
        "targets": [
            {"refId": "A", "expr": 'rag_live_eval_score{metric="faithfulness"}', "legendFormat": "faithfulness"}
        ],
    },
    {
        "type": "stat",
        "title": "Live retrieval hit@k (last request)",
        "description": "Arithmetic, no judge call: did the gold article appear in the retrieved set? Empty until a request supplies expected_article.",
        "gridPos": {"h": 6, "w": 6, "x": 6, "y": 30},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "fieldConfig": {
            "defaults": {
                "unit": "none",
                "decimals": 3,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": 1},
                    ],
                },
            },
            "overrides": [],
        },
        "targets": [
            {"refId": "A", "expr": 'rag_live_eval_score{metric="hit_at_k"}', "legendFormat": "hit@k"}
        ],
    },
    {
        "type": "timeseries",
        "title": "Live evaluation cost",
        "description": "Wall-clock added to the request path by live evaluation. Non-zero only when evaluate=true.",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 30},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
        "targets": [
            {
                "refId": "A",
                "expr": "histogram_quantile(0.50, sum(rate(rag_live_eval_duration_seconds_bucket[5m])) by (le))",
                "legendFormat": "p50",
            },
            {
                "refId": "B",
                "expr": "sum(rate(rag_live_eval_requests_total[5m])) * 60",
                "legendFormat": "evals/min",
            },
        ],
    },
]


def main() -> int:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    panels = dashboard["panels"]
    existing = {panel.get("title") for panel in panels}
    added = [panel for panel in LIVE_PANELS if panel["title"] not in existing]
    if not added:
        print("panels already present, nothing to do")
        return 0
    # Park live-eval panels below whatever is already there.
    bottom = max((panel["gridPos"]["y"] + panel["gridPos"]["h"]) for panel in panels)
    for offset, panel in enumerate(added):
        panel["gridPos"]["y"] = bottom + (offset // 2) * 8
    panels.extend(added)
    dashboard["panels"] = panels
    DASHBOARD.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"added {len(added)} live-eval panel(s); dashboard now has {len(panels)} panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
