#!/usr/bin/env python3
"""Best-trial selector — ranks trial results by the objective metric.

Input results.json = list of {trial_id, value, [env], [checkpoint], [run_id]}.
Emits the best trial + full ranking. The winner still MUST pass hpo_gate.py
--check final (independent re-eval) before promotion — selection != validation.
Exit 0 ok, 1 no valid results.
"""
import argparse
import json
import sys


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser(description="Select best trial by objective.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        with open(a.results) as f:
            results = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"cannot read results: {e}"}))
        return 1
    if not isinstance(results, list):
        print(json.dumps({"status": "error", "error": "results must be a JSON list"}))
        return 1
    valid = [r for r in results if isinstance(r, dict) and _num(r.get("value"))]
    dropped = len(results) - len(valid)
    if not valid:
        print(json.dumps({"status": "empty", "error": "no trials with numeric 'value'",
                          "dropped": dropped}))
        return 1
    ranked = sorted(valid, key=lambda r: r["value"], reverse=(a.direction == "maximize"))
    best = ranked[0]
    out = {
        "status": "ok", "direction": a.direction, "n_valid": len(valid), "n_dropped": dropped,
        "best": {"trial_id": best.get("trial_id"), "value": best["value"],
                 "checkpoint": best.get("checkpoint"), "run_id": best.get("run_id"),
                 "env": best.get("env")},
        "ranking": [{"trial_id": r.get("trial_id"), "value": r["value"]} for r in ranked],
    }
    print(json.dumps(out, indent=2 if not a.json else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
