#!/usr/bin/env python3
"""MLflow metric scraper — reads a trial's objective metric from the tracking server.

Uses the kubeflow-llm-training MLflow workspace auth (env-driven), matching
training/common/mlflow_utils.py. stdlib urllib only.

  MLFLOW_TRACKING_URL   (or --tracking-url)
  MLFLOW_TRACKING_TOKEN -> Authorization: Bearer
  MLFLOW_WORKSPACE_NAME -> X-MLflow-Workspace

--history uses metrics/get-history to pick the best (min/max) over the eval
curve; default reads the latest value from runs/get.
Exit 0 ok, 2 unreachable/missing (graceful — loop treats as failed trial), 1 usage.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _headers():
    h = {"Content-Type": "application/json"}
    tok = os.environ.get("MLFLOW_TRACKING_TOKEN")
    ws = os.environ.get("MLFLOW_WORKSPACE_NAME")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if ws:
        h["X-MLflow-Workspace"] = ws
    return h


def _get(url, timeout):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.load(r)


def scrape(base, run_id, metric, history, direction, timeout):
    base = base.rstrip("/")
    if history:
        url = f"{base}/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric}"
        data = _get(url, timeout)
        pts = [m.get("value") for m in data.get("metrics", []) if "value" in m]
        pts = [v for v in pts if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not pts:
            return None
        return min(pts) if direction == "minimize" else max(pts)
    url = f"{base}/api/2.0/mlflow/runs/get?run_id={run_id}"
    data = _get(url, timeout)
    metrics = data.get("run", {}).get("data", {}).get("metrics", [])
    for m in metrics:
        if m.get("key") == metric:
            return m.get("value")
    return None


def main():
    ap = argparse.ArgumentParser(description="Scrape a trial's objective metric from MLflow.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--tracking-url", default=os.environ.get("MLFLOW_TRACKING_URL"))
    ap.add_argument("--direction", choices=["minimize", "maximize"], default="minimize",
                    help="only used with --history to pick best point")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--trial-id", default=None)
    a = ap.parse_args()
    if not a.tracking_url:
        print(json.dumps({"status": "error", "error": "MLFLOW_TRACKING_URL not set (or --tracking-url)"}))
        return 1
    try:
        val = scrape(a.tracking_url, a.run_id, a.metric, a.history, a.direction, a.timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(json.dumps({"status": "unreachable", "run_id": a.run_id, "error": str(e)}))
        return 2
    if val is None:
        print(json.dumps({"status": "missing", "run_id": a.run_id, "metric": a.metric}))
        return 2
    out = {"status": "ok", "run_id": a.run_id, "metric": a.metric, "value": val}
    if a.trial_id:
        out["trial_id"] = a.trial_id
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
