#!/usr/bin/env python3
"""Trial config generator — samples the validated search space into TrainJob env dicts.

Two strategies:
  random : sample `budget.max_trials` independent configs.
  asha   : successive halving. rung 0 = initial pool; rung N reads prior results
           and promotes the top 1/eta with MAX_STEPS scaled up by eta.

Reproducible via --seed. Deterministic given (schema, seed, rung, results).
The agent recommends WHICH strategy; the sampling itself is code-owned
([[sonnet-format-determinism]]). Exit 0 ok, 1 error, 2 nothing to promote.
"""
import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from search_space_schema import validate  # noqa: E402


def _sample_dim(spec, rng):
    t = spec["type"]
    if t == "choice":
        return rng.choice(spec["values"])
    if t == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    if t == "loguniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    if t == "int_uniform":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    raise ValueError(f"unknown dist type {t}")


def _fmt(v):
    # env vars are strings; keep LR in scientific, ints as ints
    if isinstance(v, float):
        return f"{v:.3e}" if v < 1e-2 else f"{v:.6g}"
    return str(v)


def _env(base, sampled):
    env = {str(k): str(v) for k, v in base.items()}
    for k, v in sampled.items():
        env[k] = _fmt(v)
    return env


def gen_random(schema, seed):
    import random
    rng = random.Random(seed)
    ss = schema["search_space"]
    n = schema["budget"]["max_trials"]
    trials = []
    for i in range(n):
        sampled = {k: _sample_dim(spec, rng) for k, spec in ss.items()}
        trials.append({"trial_id": f"t{seed}-{i:03d}", "env": _env(schema["base"], sampled),
                       "sampled": {k: _fmt(v) for k, v in sampled.items()}})
    return trials


def gen_asha(schema, seed, rung, results, eta):
    if rung == 0:
        return gen_random(schema, seed)
    if not results:
        return []
    direction = schema["objective"]["direction"]
    metric = schema["objective"]["metric"]
    scored = [r for r in results if isinstance(r.get("value"), (int, float))
              and not isinstance(r.get("value"), bool)]
    if not scored:
        return []
    scored.sort(key=lambda r: r["value"], reverse=(direction == "maximize"))
    keep = max(1, math.ceil(len(scored) / eta))
    promoted = []
    for r in scored[:keep]:
        env = dict(r.get("env", {}))
        # scale training length up by eta for the promoted rung
        cur = env.get("MAX_STEPS")
        if cur and str(cur).isdigit():
            env["MAX_STEPS"] = str(int(cur) * eta)
        promoted.append({"trial_id": f"{r['trial_id']}-r{rung}", "env": env,
                         "promoted_from": r["trial_id"], "prev_value": r["value"], "metric": metric})
    return promoted


def main():
    ap = argparse.ArgumentParser(description="Generate TrainJob trial env configs.")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--strategy", choices=["random", "asha"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rung", type=int, default=0)
    ap.add_argument("--results", help="prior results.json (asha rung>0)")
    ap.add_argument("--eta", type=int, default=3, help="asha halving factor")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        with open(a.schema) as f:
            schema = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"cannot read schema: {e}"}))
        return 1
    errors, _, _ = validate(schema)
    if errors:
        print(json.dumps({"status": "error", "error": "schema invalid — run search_space_schema.py",
                          "errors": errors}))
        return 1

    results = None
    if a.results:
        try:
            with open(a.results) as f:
                results = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"status": "error", "error": f"cannot read results: {e}"}))
            return 1

    if a.strategy == "random":
        trials = gen_random(schema, a.seed)
    else:
        trials = gen_asha(schema, a.seed, a.rung, results or [], a.eta)
        if not trials:
            print(json.dumps({"status": "empty", "reason": "no trials to promote", "rung": a.rung}))
            return 2

    out = {"status": "ok", "strategy": a.strategy, "rung": a.rung, "n": len(trials),
           "max_concurrent": schema["budget"]["max_concurrent"], "trials": trials}
    print(json.dumps(out, indent=2 if not a.json else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
