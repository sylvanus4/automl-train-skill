#!/usr/bin/env python3
"""Stdlib smoke test — proves the 8 scripts compose and the gates actually gate.
Run: python tests/smoke.py   (no pytest, no deps). Exit 0 = all pass.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "scripts")
E = os.path.join(ROOT, "examples")
PY = sys.executable
passed, failed = 0, 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {extra}")


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


# 1. schema gate accepts the example, rejects a bad one
r = run([PY, f"{S}/search_space_schema.py", "--schema", f"{E}/search-space-example.json"])
check("schema: example valid", r.returncode == 0, r.stdout)
bad = "/tmp/_smoke_bad.json"
json.dump({"tunable_env": ["LEARNING_RATE"], "base": {},
           "search_space": {"SECRET": {"type": "choice", "values": [1]}},
           "objective": {"metric": "eval_loss", "direction": "minimize"},
           "budget": {"max_trials": 2, "max_concurrent": 2, "gpus_per_trial": 2, "gpu_cap": 2}},
          open(bad, "w"))
r = run([PY, f"{S}/search_space_schema.py", "--schema", bad])
check("schema: bad rejected (exit 1)", r.returncode == 1, r.stdout)

# 2. zero-step guard
zs = "/tmp/_smoke_zs.json"
json.dump({"tunable_env": ["MAX_STEPS"], "base": {},
           "search_space": {"MAX_STEPS": {"type": "choice", "values": [0, 100]}},
           "objective": {"metric": "eval_loss", "direction": "minimize"},
           "budget": {"max_trials": 2, "max_concurrent": 1, "gpus_per_trial": 1, "gpu_cap": 1}},
          open(zs, "w"))
r = run([PY, f"{S}/search_space_schema.py", "--schema", zs])
check("schema: zero-step rejected", r.returncode == 1 and "zero training" in r.stdout, r.stdout)

# 3. local backend: good lr succeeds, huge lr diverges -> model_code
r = run([PY, f"{E}/local_backend.py"], env={**os.environ, "LEARNING_RATE": "0.3", "MAX_STEPS": "100"})
check("backend: good lr -> eval_loss", r.returncode == 0 and "eval_loss" in r.stdout, r.stderr)
r = run([PY, f"{E}/local_backend.py"], env={**os.environ, "LEARNING_RATE": "5.0", "MAX_STEPS": "100"})
diverged = r.returncode == 1
cr = run([PY, f"{S}/failure_classify.py", "classify"], input=r.stdout + r.stderr)
cat = json.loads(cr.stdout).get("category")
check("failure: diverge -> model_code", diverged and cat == "model_code", f"cat={cat}")

# 4. stop policy: consecutive SYSTEMIC failures stop; model_code (per-config) does not
res = "/tmp/_smoke_res.json"
json.dump([{"trial_id": "a", "status": "Failed", "category": "data"},
           {"trial_id": "b", "status": "Failed", "category": "data"}], open(res, "w"))
r = run([PY, f"{S}/failure_classify.py", "policy", "--results", res, "--consecutive", "2"])
check("policy: systemic (data) -> stop", json.loads(r.stdout).get("stop") is True, r.stdout)
json.dump([{"trial_id": "a", "status": "Failed", "category": "model_code"},
           {"trial_id": "b", "status": "Failed", "category": "model_code"}], open(res, "w"))
r = run([PY, f"{S}/failure_classify.py", "policy", "--results", res, "--consecutive", "2"])
check("policy: model_code (per-config) -> continue", json.loads(r.stdout).get("stop") is False, r.stdout)

# 5. launch_review gates on wall-time
r = run([PY, f"{S}/launch_review.py", "--schema", f"{E}/search-space-example.json",
         "--per-trial-min", "8", "--baseline-min", "5", "--max-minutes", "20"])
check("launch_review: over-limit gates (exit 1)", r.returncode == 1, r.stdout)

# 6. full end-to-end composes and promotes on the toy objective
r = run([PY, f"{E}/run_local_sweep.py"])
check("end-to-end: runs + promotes", r.returncode == 0 and "final_gate=PASS(promote)" in r.stdout,
      r.stdout[-200:])

# 7. second backend (different algorithm) produces its metric
r = run([PY, f"{E}/logreg_backend.py"], env={**os.environ, "LEARNING_RATE": "0.5", "MAX_STEPS": "120"})
check("backend #2 (logreg): accuracy produced", r.returncode == 0 and "accuracy" in r.stdout, r.stderr)

# 8. language-agnostic backend: a NON-Python (bash) trainer the loop can scrape
r = run(["bash", f"{E}/demo_trainer.sh"], env={**os.environ, "LEARNING_RATE": "0.3", "MAX_STEPS": "100"})
check("backend #3 (bash, non-Python): eval_loss produced",
      r.returncode == 0 and "eval_loss" in r.stdout, r.stderr)

# 9. MAXIMIZE objective path (accuracy): select_best + final gate honor direction
mres = "/tmp/_smoke_max.json"
json.dump([{"trial_id": "a", "value": 0.81}, {"trial_id": "b", "value": 0.97},
           {"trial_id": "c", "value": 0.74}], open(mres, "w"))
r = run([PY, f"{S}/select_best.py", "--results", mres, "--direction", "maximize", "--json"])
best_max = json.loads(r.stdout)["best"]["trial_id"]
check("maximize: select_best picks highest", best_max == "b", r.stdout)
json.dump({"metric": "accuracy", "value": 0.80}, open("/tmp/_smoke_maxbl.json", "w"))
json.dump({"metric": "accuracy", "value": 0.97, "direction": "maximize"}, open("/tmp/_smoke_maxbest.json", "w"))
r = run([PY, f"{S}/hpo_gate.py", "--check", "final", "--baseline", "/tmp/_smoke_maxbl.json",
         "--best", "/tmp/_smoke_maxbest.json"])
check("maximize: final gate passes when higher", r.returncode == 0, r.stdout)
json.dump({"metric": "accuracy", "value": 0.70, "direction": "maximize"}, open("/tmp/_smoke_maxworse.json", "w"))
r = run([PY, f"{S}/hpo_gate.py", "--check", "final", "--baseline", "/tmp/_smoke_maxbl.json",
         "--best", "/tmp/_smoke_maxworse.json"])
check("maximize: final gate fails when lower", r.returncode == 1, r.stdout)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
