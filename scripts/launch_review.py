#!/usr/bin/env python3
"""Launch review — the pre-GPU-spend gate (invariant #3, expanded).

Adapted from NVIDIA tao-run-automl's launch review: before spending a single
GPU-hour, show the searchable ranges, the objective, the budget, and an
ESTIMATED wall-clock, and refuse to launch if the estimate blows a stated limit.
Answers "is this search worth starting?" with numbers, not vibes.

wall_time ≈ baseline_min + ceil(max_trials / max_concurrent) * per_trial_min

Deterministic (stdlib-only). Exit 0 = within limit, 1 = exceeds --max-minutes,
2 = cannot read schema.
"""
import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from search_space_schema import validate  # noqa: E402


def _range_str(spec):
    t = spec.get("type")
    if t == "choice":
        return f"choice{spec.get('values')}"
    return f"{t}[{spec.get('low')}, {spec.get('high')}]"


def review(schema, per_trial_min, baseline_min, max_minutes):
    bud = schema["budget"]
    rounds = math.ceil(bud["max_trials"] / bud["max_concurrent"])
    wall = baseline_min + rounds * per_trial_min
    within = (max_minutes is None) or (wall <= max_minutes)
    ss = schema.get("search_space", {})
    return {
        "objective": schema.get("objective"),
        "searchable": {k: _range_str(v) for k, v in ss.items()},
        "budget": bud,
        "sequential_rounds": rounds,
        "per_trial_min": per_trial_min,
        "baseline_min": baseline_min,
        "estimated_wall_minutes": round(wall, 1),
        "max_minutes": max_minutes,
        "within_limit": within,
        "advice": None if within else
        "estimate exceeds limit — reduce max_trials, raise max_concurrent, "
        "shorten MAX_STEPS, or narrow the search space before launching",
    }


def main():
    ap = argparse.ArgumentParser(description="Pre-GPU launch review with wall-time estimate.")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--per-trial-min", type=float, required=True,
                    help="estimated minutes for ONE trial (from a probe run)")
    ap.add_argument("--baseline-min", type=float, default=0.0)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stated wall-clock limit; gate fails if exceeded")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        schema = json.load(open(a.schema))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"cannot read schema: {e}"}))
        return 2
    errors, _, _ = validate(schema)
    if errors:
        print(json.dumps({"status": "error", "error": "schema invalid", "errors": errors}))
        return 2
    r = review(schema, a.per_trial_min, a.baseline_min, a.max_minutes)
    r["status"] = "ok"
    if a.json:
        print(json.dumps(r))
    else:
        print(f"Objective : {r['objective']}")
        print(f"Search    : {json.dumps(r['searchable'], ensure_ascii=False)}")
        print(f"Budget    : {r['budget']['max_trials']} trials, "
              f"{r['budget']['max_concurrent']} concurrent, {r['sequential_rounds']} rounds")
        print(f"Est. wall : {r['estimated_wall_minutes']} min "
              f"(baseline {a.baseline_min} + {r['sequential_rounds']}x{a.per_trial_min})")
        if a.max_minutes is not None:
            print(f"Limit     : {a.max_minutes} min -> {'WITHIN' if r['within_limit'] else 'EXCEEDS'}")
        if r["advice"]:
            print(f"Advice    : {r['advice']}")
    return 0 if r["within_limit"] else 1


if __name__ == "__main__":
    sys.exit(main())
