#!/usr/bin/env python3
"""Failure classifier + budget-preserving stop policy (invariant #5).

Adapted from NVIDIA tao-run-automl's failure taxonomy: classify WHY a trial
failed before spending more GPU, and if consecutive trials fail from the same
systemic cause, STOP and summarize instead of exhausting the budget on
recommendations that cannot succeed.

Two modes:
  classify : one log (file or stdin) -> category
  policy   : a results list -> {stop, reason} based on shared-cause detection

Deterministic (stdlib-only). Exit 0.
"""
import argparse
import json
import re
import sys

# ordered: first match wins (most specific / most actionable first)
TAXONOMY = [
    ("data", [r"neither a.{0,3}Dataset", r"FileNotFoundError", r"No such file",
              r"NoSuchKey", r"dataset .*not found", r"path .*unreadable", r"Directory .* is neither"]),
    ("image_cred", [r"ImagePullBackOff", r"ErrImagePull", r"manifest .*not found",
                    r"401 Unauthorized", r"403 Forbidden", r"AccessDenied",
                    r"authentication failed", r"invalid credentials", r"pull access denied"]),
    ("infra", [r"JobCreationFailed", r"FailedScheduling", r"admission webhook",
               r"exceeded quota", r"Insufficient nvidia\.com/gpu", r"node.*NotReady",
               r"Kueue.*rejected", r"OOMKilled"]),
    ("spec_schema", [r"out of range", r"zero training steps", r"max_steps must",
                     r"invalid .*parameter", r"ValueError: .*(step|batch|lr|learning)"]),
    ("model_code", [r"CUDA out of memory", r"RuntimeError", r"loss is nan", r"NaN",
                    r"Traceback \(most recent call last\)", r"metric .*not found",
                    r"eval_loss.*None", r"KeyError"]),
]
# systemic categories: if repeated, a new trial will also fail -> stop
SYSTEMIC = {"data", "image_cred", "infra", "spec_schema"}


def classify(text):
    for cat, pats in TAXONOMY:
        for p in pats:
            if re.search(p, text, re.IGNORECASE):
                return cat
    return "unknown"


def policy(results, consecutive):
    """results: [{trial_id, status, category?}]. Stop if the last `consecutive`
    entries all FAILED with the same systemic category."""
    failed = [r for r in results if r.get("status") not in ("ok", "Succeeded")]
    counts = {}
    for r in failed:
        c = r.get("category") or "unknown"
        counts[c] = counts.get(c, 0) + 1
    # look at the tail
    tail = [r for r in results if r.get("status") not in ("ok", "Succeeded")][-consecutive:]
    stop, reason, cat = False, None, None
    if len(tail) >= consecutive:
        cats = {(r.get("category") or "unknown") for r in tail}
        if len(cats) == 1:
            cat = next(iter(cats))
            if cat in SYSTEMIC:
                stop = True
                reason = (f"{consecutive} consecutive trials failed with shared "
                          f"systemic cause '{cat}' — stopping to preserve GPU budget")
    return {"stop": stop, "reason": reason, "dominant_category": cat,
            "failed_total": len(failed), "counts": counts}


def main():
    ap = argparse.ArgumentParser(description="Classify trial failures / decide stop policy.")
    sub = ap.add_argument
    ap.add_argument("mode", choices=["classify", "policy"])
    ap.add_argument("--log", help="log file for classify (default: stdin)")
    ap.add_argument("--results", help="results.json for policy mode")
    ap.add_argument("--consecutive", type=int, default=2,
                    help="how many same-cause failures in a row trigger stop")
    a = ap.parse_args()
    if a.mode == "classify":
        text = open(a.log).read() if a.log else sys.stdin.read()
        print(json.dumps({"category": classify(text)}))
        return 0
    # policy
    if not a.results:
        print(json.dumps({"status": "error", "error": "--results required for policy"}))
        return 0
    results = json.load(open(a.results))
    print(json.dumps({"status": "ok", **policy(results, a.consecutive)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
