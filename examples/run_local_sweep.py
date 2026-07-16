#!/usr/bin/env python3
"""End-to-end reference runner — composes ALL the skill's scripts with the local
backend, zero GPU. This is both the integration test ("do the 8 scripts actually
compose?") and the worked example an adopter copies and points at their trainer.

Flow (every step is a real call to a shipped script):
  gate schema -> gate budget -> launch_review -> baseline(local_backend) ->
  gate baseline -> trials(trial_config_gen) -> run each(local_backend) ->
  scrape / failure_classify -> stop policy -> select_best -> gate final -> report

Swap `run_trial()` for your own submit()+scrape() to use a real trainer.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
PY = sys.executable
SCHEMA = os.path.join(HERE, "search-space-local.json")
BACKEND = os.path.join(HERE, "local_backend.py")
WORK = os.path.join(HERE, ".localrun")
os.makedirs(WORK, exist_ok=True)


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def gate(check, **files):
    a = [PY, os.path.join(SCRIPTS, "hpo_gate.py"), "--check", check]
    for k, v in files.items():
        a += [f"--{k}", v]
    r = sh(a)
    ok = r.returncode == 0
    print(f"  gate {check}: {'PASS' if ok else 'FAIL'}  {r.stdout.strip()[:120]}")
    return ok


def run_trial(trial_id, env):
    """submit + scrape for the local backend. Returns {trial_id, status, value?, category?}."""
    r = sh([PY, BACKEND], env={**os.environ, **{k: str(v) for k, v in env.items()}})
    if r.returncode == 0:
        val = None
        for line in r.stdout.strip().splitlines():
            try:
                j = json.loads(line)
                if "eval_loss" in j:
                    val = float(j["eval_loss"])
            except Exception:  # noqa: BLE001
                pass
        if val is not None:
            return {"trial_id": trial_id, "status": "ok", "value": val, "env": env}
    # failed -> classify from the log
    cr = sh([PY, os.path.join(SCRIPTS, "failure_classify.py"), "classify"],
            input=r.stdout + r.stderr)
    cat = json.loads(cr.stdout).get("category", "unknown")
    return {"trial_id": trial_id, "status": "Failed", "category": cat, "env": env}


def main():
    print("== automl-train local end-to-end sweep (zero GPU) ==")
    if not gate("schema", schema=SCHEMA):
        return 1
    if not gate("budget", schema=SCHEMA):
        return 1

    lr = sh([PY, os.path.join(SCRIPTS, "launch_review.py"), "--schema", SCHEMA,
             "--per-trial-min", "0.02", "--baseline-min", "0.02", "--max-minutes", "5", "--json"])
    print(f"  launch_review: {lr.stdout.strip()[:140]}")
    if lr.returncode != 0:
        print("  launch review exceeds limit — aborting"); return 1

    base = json.load(open(SCHEMA))["base"]
    print("baseline:")
    bl = run_trial("baseline", dict(base, LEARNING_RATE="0.01", MAX_STEPS="50"))
    print(f"  baseline -> {bl.get('value', bl['status'])}")
    if "value" not in bl:
        print("  baseline failed — abort (invariant #2)"); return 1
    json.dump({"metric": "eval_loss", "value": bl["value"], "direction": "minimize"},
              open(os.path.join(WORK, "baseline.json"), "w"))
    if not gate("baseline", baseline=os.path.join(WORK, "baseline.json")):
        return 1

    gen = sh([PY, os.path.join(SCRIPTS, "trial_config_gen.py"), "--schema", SCHEMA,
              "--strategy", "random", "--seed", "7", "--json"])
    trials = json.loads(gen.stdout)["trials"]
    results = []
    for t in trials:
        r = run_trial(t["trial_id"], t["env"])
        tag = r.get("value", f"FAILED[{r.get('category')}]")
        print(f"  {t['trial_id']}: lr={t['sampled'].get('LEARNING_RATE')} steps={t['sampled'].get('MAX_STEPS')} -> {tag}")
        results.append(r)
        json.dump(results, open(os.path.join(WORK, "results.json"), "w"))
        # budget-preserving stop on shared systemic cause
        pol = sh([PY, os.path.join(SCRIPTS, "failure_classify.py"), "policy",
                  "--results", os.path.join(WORK, "results.json"), "--consecutive", "3"])
        if json.loads(pol.stdout).get("stop"):
            print(f"  STOP: {json.loads(pol.stdout)['reason']}"); break

    sel = sh([PY, os.path.join(SCRIPTS, "select_best.py"), "--results",
              os.path.join(WORK, "results.json"), "--direction", "minimize", "--json"])
    best = json.loads(sel.stdout).get("best")
    if not best:
        print("no valid trials"); return 1
    json.dump({"metric": "eval_loss", "value": best["value"], "direction": "minimize"},
              open(os.path.join(WORK, "best.json"), "w"))
    print(f"best: {best['trial_id']} -> {best['value']}")
    final_ok = gate("final", baseline=os.path.join(WORK, "baseline.json"),
                    best=os.path.join(WORK, "best.json"))

    print("report:")
    rep = sh([PY, os.path.join(SCRIPTS, "report.py"), "--results", os.path.join(WORK, "results.json"),
              "--baseline", os.path.join(WORK, "baseline.json"), "--schema", SCHEMA])
    print("  " + rep.stdout.strip().replace("\n", "\n  "))
    print(f"== DONE  final_gate={'PASS(promote)' if final_ok else 'no-improvement(hold)'} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
