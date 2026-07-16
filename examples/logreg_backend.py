#!/usr/bin/env python3
"""Second reference backend — a DIFFERENT algorithm and a MAXIMIZE objective.

Logistic-regression classifier fit by gradient descent on a synthetic 2-class
dataset. The objective is held-out **accuracy** (higher is better), so this
exercises the `maximize` direction end to end — a different code path from the
regression backend's `eval_loss` (minimize). LEARNING_RATE and MAX_STEPS move
accuracy: too low underfits within the budget, a good value converges, too high
diverges (a real model-code failure).

Env: LEARNING_RATE, MAX_STEPS. Prints `accuracy`. Exit 0 ok, 1 diverged.
"""
import json
import math
import os
import sys


def data(n, seed):
    import random
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        # two Gaussian blobs, labels 0/1, mild overlap
        label = rng.randint(0, 1)
        cx = 1.2 if label else -1.2
        xs.append((rng.gauss(cx, 1.0), rng.gauss(cx, 1.0)))
        ys.append(label)
    return xs, ys


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def train(lr, steps):
    xtr, ytr = data(300, 1)
    xev, yev = data(100, 2)
    w0, w1, b = 0.0, 0.0, 0.0
    n = len(xtr)
    for _ in range(steps):
        g0 = g1 = gb = 0.0
        for (x0, x1), y in zip(xtr, ytr):
            p = sigmoid(w0 * x0 + w1 * x1 + b)
            d = p - y
            g0 += d * x0
            g1 += d * x1
            gb += d
        w0 -= lr * g0 / n
        w1 -= lr * g1 / n
        b -= lr * gb / n
        if not all(math.isfinite(v) for v in (w0, w1, b)):
            return None
    correct = sum(1 for (x0, x1), y in zip(xev, yev)
                  if (1 if sigmoid(w0 * x0 + w1 * x1 + b) >= 0.5 else 0) == y)
    return correct / len(xev)


def main():
    try:
        lr = float(os.environ.get("LEARNING_RATE", "0.1"))
        steps = int(float(os.environ.get("MAX_STEPS", "100")))
    except (TypeError, ValueError) as e:
        print(f"spec error: bad LEARNING_RATE/MAX_STEPS ({e})", file=sys.stderr)
        return 1
    acc = train(lr, steps)
    if acc is None:
        print("RuntimeError: weights diverged to nan", file=sys.stderr)
        return 1
    print(f"{{'accuracy': '{acc:.4f}', 'lr': '{lr:.4g}', 'steps': {steps}}}")
    print(json.dumps({"accuracy": round(acc, 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
