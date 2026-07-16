#!/usr/bin/env python3
"""Search-space schema validator — invariant #1 ("no schema, no search").

Backend-agnostic. Validates an AutoML/HPO search-space JSON structurally so the
loop cannot enter with an ill-formed or over-budget search. The set of tunable
keys is declared BY THE SCHEMA (`tunable_env`), so this works for any trainer
(Kubeflow Trainer v2, Ray Tune, SLURM, a local subprocess) — not just one.

Deterministic (stdlib-only). Exposes validate() for hpo_gate.py reuse.
Exit 0 = ok, 1 = invalid schema, 2 = cannot read input.
"""
import argparse
import json
import sys

# Example default whitelist (HF/TRL-style env names). Override per-trainer by
# declaring "tunable_env": ["LR", "BATCH", ...] in the schema.
DEFAULT_TUNABLE = {
    "LEARNING_RATE", "BATCH_SIZE", "GRAD_ACCUM_STEPS", "MAX_STEPS", "EPOCHS",
    "WARMUP_STEPS", "MAX_LENGTH", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT",
    "LR_SCHEDULER_TYPE", "WEIGHT_DECAY",
}
DIST_TYPES = {"choice", "uniform", "loguniform", "int_uniform"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_dim(key, spec, errors):
    if not isinstance(spec, dict) or "type" not in spec:
        errors.append(f"search_space.{key}: must be an object with a 'type' field")
        return
    t = spec.get("type")
    if t not in DIST_TYPES:
        errors.append(f"search_space.{key}: type '{t}' not in {sorted(DIST_TYPES)}")
        return
    if t == "choice":
        vals = spec.get("values")
        if not isinstance(vals, list) or not vals:
            errors.append(f"search_space.{key}: 'choice' needs a non-empty 'values' list")
    else:  # uniform / loguniform / int_uniform
        lo, hi = spec.get("low"), spec.get("high")
        if not _num(lo) or not _num(hi):
            errors.append(f"search_space.{key}: '{t}' needs numeric 'low' and 'high'")
            return
        if lo >= hi:
            errors.append(f"search_space.{key}: low ({lo}) must be < high ({hi})")
        if t == "loguniform" and lo <= 0:
            errors.append(f"search_space.{key}: 'loguniform' needs low > 0 (got {lo})")
        if t == "int_uniform" and (int(lo) != lo or int(hi) != hi):
            errors.append(f"search_space.{key}: 'int_uniform' needs integer low/high")


def validate(schema):
    """Return (errors, warnings, meta). errors empty => valid."""
    errors, warnings = [], []
    if not isinstance(schema, dict):
        return (["schema must be a JSON object"], [], {})

    # tunable whitelist: schema-declared, else built-in default (with a nudge)
    declared = schema.get("tunable_env")
    if isinstance(declared, list) and declared:
        allowed = set(declared)
    elif declared is not None:
        errors.append("tunable_env: must be a non-empty list of key names")
        allowed = set()
    else:
        allowed = set(DEFAULT_TUNABLE)
        warnings.append("tunable_env not declared — using built-in HF/TRL default set; "
                        "declare tunable_env for your own trainer's parameter names")

    base = schema.get("base")
    if base is None:
        base = {}
        warnings.append("base (fixed params) not declared")
    elif not isinstance(base, dict):
        errors.append("base: must be an object of fixed params")
        base = {}

    obj = schema.get("objective")
    metric, direction = None, None
    if not isinstance(obj, dict):
        errors.append("objective: required object {metric, direction}")
    else:
        metric = obj.get("metric")
        direction = obj.get("direction")
        if not isinstance(metric, str) or not metric:
            errors.append("objective.metric: required non-empty string")
        if direction not in {"minimize", "maximize"}:
            errors.append("objective.direction: must be 'minimize' or 'maximize'")

    ss = schema.get("search_space")
    n_dims = 0
    if not isinstance(ss, dict) or not ss:
        errors.append("search_space: required non-empty object (no schema => no search)")
    else:
        n_dims = len(ss)
        for key, spec in ss.items():
            if key not in allowed:
                errors.append(f"search_space.{key}: not in tunable_env {sorted(allowed)}")
                continue
            if key in base:
                errors.append(f"search_space.{key}: also fixed in base (ambiguous — pick one)")
            _validate_dim(key, spec, errors)

    bud = schema.get("budget")
    if not isinstance(bud, dict):
        errors.append("budget: required object")
    else:
        for k in ("max_trials", "max_concurrent", "gpus_per_trial", "gpu_cap"):
            v = bud.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                errors.append(f"budget.{k}: required int >= 1")
        if all(isinstance(bud.get(k), int) for k in ("max_concurrent", "gpus_per_trial", "gpu_cap")):
            need = bud["max_concurrent"] * bud["gpus_per_trial"]
            if need > bud["gpu_cap"]:
                errors.append(
                    f"budget: max_concurrent*gpus_per_trial ({need}) exceeds gpu_cap ({bud['gpu_cap']})"
                )

    meta = {"method": schema.get("method"), "n_search_dims": n_dims,
            "metric": metric, "direction": direction, "n_tunable": len(allowed)}
    return (errors, warnings, meta)


def main():
    ap = argparse.ArgumentParser(description="Validate an AutoML search-space schema.")
    ap.add_argument("--schema", required=True, help="Path to search-space.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        with open(a.schema) as f:
            schema = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": f"cannot read schema: {e}"}))
        return 2
    errors, warnings, meta = validate(schema)
    status = "ok" if not errors else "invalid"
    out = {"status": status, "errors": errors, "warnings": warnings, **meta}
    if a.json:
        print(json.dumps(out))
    else:
        print(f"[{status}] dims={meta.get('n_search_dims')} "
              f"objective={meta.get('metric')}/{meta.get('direction')}")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  warn:  {w}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
