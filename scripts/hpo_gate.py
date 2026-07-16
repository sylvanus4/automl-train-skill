#!/usr/bin/env python3
"""HPO gate — the 4 TAO invariants, owned by deterministic code.

  --check schema   --schema search-space.json     invariant #1 (no schema, no search)
  --check baseline --baseline baseline.json        invariant #2 (baseline recorded first)
  --check budget   --schema search-space.json      invariant #3 (GPU budget within cap)
  --check final    --baseline baseline.json --best best.json   invariant #4 (best genuinely better)

exit 0 = gate PASS (loop may proceed), 1 = gate FAIL (STOP), 2 = cannot read input.
Never let the model self-report these — the loop is closed by this file
([[close-the-agent-loop]]·[[evaluator-must-act]]).
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from search_space_schema import validate  # noqa: E402


def _load(path):
    with open(path) as f:
        return json.load(f)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def gate_schema(a):
    schema = _load(a.schema)
    errors, warnings, meta = validate(schema)
    ok = not errors
    return ok, {"check": "schema", "pass": ok, "errors": errors, "warnings": warnings, **meta}


def gate_baseline(a):
    b = _load(a.baseline)
    metric, val = b.get("metric"), b.get("value")
    ok = isinstance(metric, str) and bool(metric) and _num(val)
    reason = None if ok else "baseline.json must contain {metric: str, value: number}"
    return ok, {"check": "baseline", "pass": ok, "metric": metric, "value": val, "reason": reason}


def gate_budget(a):
    schema = _load(a.schema)
    bud = schema.get("budget", {})
    need = bud.get("max_concurrent", 0) * bud.get("gpus_per_trial", 0)
    cap = bud.get("gpu_cap", 0)
    ok = _num(need) and _num(cap) and need <= cap and bud.get("max_trials", 0) >= 1
    return ok, {"check": "budget", "pass": ok, "gpus_in_flight": need, "gpu_cap": cap,
                "max_trials": bud.get("max_trials"), "max_concurrent": bud.get("max_concurrent")}


def gate_final(a):
    base = _load(a.baseline)
    best = _load(a.best)
    direction = best.get("direction") or a.direction
    bmetric, bval = base.get("metric"), base.get("value")
    xval = best.get("value")
    if not (_num(bval) and _num(xval) and direction in {"minimize", "maximize"}):
        return False, {"check": "final", "pass": False,
                       "reason": "need numeric baseline/best values and a direction"}
    if best.get("metric") and base.get("metric") and best["metric"] != base["metric"]:
        return False, {"check": "final", "pass": False,
                       "reason": f"metric mismatch: baseline={base['metric']} best={best['metric']}"}
    if direction == "minimize":
        improved = xval <= bval * (1.0 - a.min_improve)
    else:
        improved = xval >= bval * (1.0 + a.min_improve)
    return improved, {"check": "final", "pass": improved, "direction": direction,
                      "baseline": bval, "best": xval, "min_improve": a.min_improve,
                      "verdict": "improved" if improved else "no improvement over baseline"}


CHECKS = {"schema": gate_schema, "baseline": gate_baseline, "budget": gate_budget, "final": gate_final}


def main():
    ap = argparse.ArgumentParser(description="HPO gate — 4 TAO invariants.")
    ap.add_argument("--check", required=True, choices=list(CHECKS))
    ap.add_argument("--schema")
    ap.add_argument("--baseline")
    ap.add_argument("--best")
    ap.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    ap.add_argument("--min-improve", type=float, default=0.0,
                    help="fractional improvement required for --check final (0 = any non-worse)")
    a = ap.parse_args()
    need = {"schema": ["schema"], "baseline": ["baseline"], "budget": ["schema"],
            "final": ["baseline", "best"]}[a.check]
    for n in need:
        if not getattr(a, n):
            print(json.dumps({"status": "error", "error": f"--{n} required for --check {a.check}"}))
            return 2
    try:
        ok, out = CHECKS[a.check](a)
    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "error": f"cannot read input: {e}"}))
        return 2
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "error", "error": f"bad input: {e}"}))
        return 2
    out["status"] = "ok"
    print(json.dumps(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
