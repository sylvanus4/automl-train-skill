#!/usr/bin/env python3
"""Structured final report (adapted from tao-run-automl's final summary).

Ties the loop together honestly: search space, baseline, every trial metric,
the winner, whether it actually beat the baseline, and the root cause of every
failure. This is the artifact that makes "the search found no improvement" a
first-class, auditable outcome rather than a silently dropped run.

Deterministic (stdlib-only). Exit 0.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from search_space_schema import validate  # noqa: E402


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def build(schema, baseline, results, direction):
    valid = [r for r in results if _num(r.get("value"))]
    failed = [r for r in results if not _num(r.get("value"))]
    best = None
    promoted = False
    if valid:
        best = sorted(valid, key=lambda r: r["value"], reverse=(direction == "maximize"))[0]
        b = baseline.get("value") if baseline else None
        if _num(b):
            promoted = (best["value"] <= b) if direction == "minimize" else (best["value"] >= b)
    return {
        "objective": (schema or {}).get("objective"),
        "algorithm": (schema or {}).get("strategy") or "random/asha",
        "budget": (schema or {}).get("budget"),
        "search_space": (schema or {}).get("search_space"),
        "baseline": baseline,
        "n_trials": len(results), "n_valid": len(valid), "n_failed": len(failed),
        "trials": [{"trial_id": r.get("trial_id"), "value": r.get("value"),
                    "status": r.get("status"), "category": r.get("category")} for r in results],
        "best": ({"trial_id": best.get("trial_id"), "value": best["value"],
                  "checkpoint": best.get("checkpoint")} if best else None),
        "promoted": promoted,
        "verdict": ("best beats baseline — promote" if promoted else
                    "no trial beat baseline — nothing promoted"),
        "failures": [{"trial_id": r.get("trial_id"), "category": r.get("category"),
                      "status": r.get("status")} for r in failed],
    }


def main():
    ap = argparse.ArgumentParser(description="Emit a structured final HPO report.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--schema")
    ap.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    results = json.load(open(a.results))
    baseline = json.load(open(a.baseline)) if a.baseline else None
    schema = None
    if a.schema:
        schema = json.load(open(a.schema))
        errors, _, _ = validate(schema)
        if errors:
            schema = None  # report still useful without a schema
    rep = build(schema, baseline, results, a.direction)
    rep["status"] = "ok"
    if a.json:
        print(json.dumps(rep))
    else:
        print(f"baseline: {rep['baseline']}")
        print(f"trials: {rep['n_trials']} ({rep['n_valid']} valid, {rep['n_failed']} failed)")
        for t in rep["trials"]:
            tag = f" [{t['category']}]" if t.get("category") else ""
            print(f"  {t['trial_id']}: {t.get('value', t.get('status'))}{tag}")
        print(f"best: {rep['best']}")
        print(f"verdict: {rep['verdict']}")
        if rep["failures"]:
            print("failures:", json.dumps(rep["failures"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
