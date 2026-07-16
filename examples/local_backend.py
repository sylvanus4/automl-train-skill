#!/usr/bin/env python3
"""Local reference training backend — real signal, zero GPU, stdlib only.

A tiny linear model fit by full-batch gradient descent on a synthetic dataset
(y = 3x - 2). LEARNING_RATE and MAX_STEPS genuinely change the held-out
eval_loss: a good LR converges, too-low underfits within the step budget, too-high
diverges to inf. So a search has a real optimum and a trial CAN beat the baseline.

This is the `submit(config) -> metric` adapter an adopter copies for their own
trainer. It reads the trial's params from env, trains, and prints eval_loss the
way HF/most trainers log it, so the same scrape works.

Env: LEARNING_RATE, MAX_STEPS (others ignored). Exit 0 ok, 1 diverged (a real
model-code failure the loop's failure_classify will tag).
"""
import json
import math
import os
import sys


def dataset(n, seed):
    import random
    rng = random.Random(seed)
    xs = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    ys = [3.0 * x - 2.0 + rng.gauss(0, 0.05) for x in xs]
    return xs, ys


def mse(w, b, xs, ys):
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def train(lr, steps):
    xtr, ytr = dataset(200, 1)
    xev, yev = dataset(50, 2)
    w, b = 0.0, 0.0
    n = len(xtr)
    for _ in range(steps):
        pred = [w * x + b for x in xtr]
        gw = sum(2 * (p - y) * x for p, x, y in zip(pred, xtr, ytr)) / n
        gb = sum(2 * (p - y) for p, y in zip(pred, ytr)) / n
        w -= lr * gw
        b -= lr * gb
        if not (math.isfinite(w) and math.isfinite(b)):
            return None  # diverged
    loss = mse(w, b, xev, yev)
    return loss if math.isfinite(loss) and loss < 1e6 else None


def main():
    try:
        lr = float(os.environ.get("LEARNING_RATE", "0.01"))
        steps = int(float(os.environ.get("MAX_STEPS", "50")))
    except (TypeError, ValueError) as e:
        print(f"spec error: bad LEARNING_RATE/MAX_STEPS ({e})", file=sys.stderr)
        return 1
    loss = train(lr, steps)
    if loss is None:
        # a real failure path (model-code): loss diverged to nan/inf
        print("RuntimeError: loss is nan (training diverged)", file=sys.stderr)
        return 1
    # log the way a trainer does (scrapeable), plus a machine-readable line
    print(f"{{'eval_loss': '{loss:.6f}', 'lr': '{lr:.4g}', 'steps': {steps}}}")
    print(json.dumps({"eval_loss": round(loss, 6)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
