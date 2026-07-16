# automl-train

> Language: English (main) · [한국어](README.ko.md)

A schema-gated hyperparameter-search (HPO / AutoML) **agent skill** that works with
any training backend. It turns a single training run into a disciplined search and,
crucially, **refuses to declare a win that isn't one**.

It is a portable `SKILL.md` contract plus eight small stdlib-only Python scripts,
with a runnable zero-GPU end-to-end example. Drop it into Claude Code, Codex, or
Gemini (or just run the scripts by hand).

## Why

Most AutoML demos show you the good number at the end. The interesting question is
what happens when the search *doesn't* find an improvement. This skill answers that
honestly: five gates, owned by deterministic code, decide whether a run may start
and whether a result may be promoted. The model recommends the next hyperparameters
and nothing else.

The gate philosophy is borrowed from NVIDIA's
[`tao-run-automl`](https://github.com/NVIDIA-TAO/tao-skill-bank) skill (Apache-2.0),
which reviews recommendations, metric, search space, and expected runtime before it
spends a GPU. What is different here: it is trainer-agnostic (you wire your own
backend) and the gates are plain Python you can read in a minute.

## The gates

1. **No schema, no search** — an invalid, unbounded, or zero-training-step search space cannot start the loop.
2. **Baseline before tuning** — you must record a reference metric first.
3. **Launch reviewed up front** — `max_concurrent × gpus_per_trial ≤ gpu_cap`, plus an estimated wall-clock that fails closed against a stated limit.
4. **Independent final eval** — the winning trial must beat the baseline on the same eval set, or it is not promoted.
5. **Fail without burning budget** — failures are classified (data / image-cred / infra / spec-schema / model-code); a shared systemic cause stops the sweep instead of exhausting the GPU budget on runs that cannot succeed.

Search strategies: **random**, **ASHA** (successive halving), and **Hyperband** (bracketed successive halving).

## Quickstart

```bash
# 1. validate a search space (fails closed)
python scripts/search_space_schema.py --schema examples/search-space-example.json

# 2. gates
python scripts/hpo_gate.py --check schema --schema examples/search-space-example.json
python scripts/hpo_gate.py --check budget --schema examples/search-space-example.json

# 3. generate trials (random or ASHA successive-halving)
python scripts/trial_config_gen.py --schema examples/search-space-example.json --strategy random --seed 7 --json

# 4. after running trials on YOUR trainer and collecting {trial_id, value}:
python scripts/select_best.py --results results.json --direction minimize --json
python scripts/hpo_gate.py --check final --baseline baseline.json --best best.json
```

## Verify it works (zero GPU)

The repo ships a runnable end-to-end proof so you never take the composition on
faith. A tiny stdlib trainer (`examples/local_backend.py`) fits a linear model by
gradient descent, so `LEARNING_RATE` and `MAX_STEPS` genuinely move the eval loss.

```bash
python tests/smoke.py                  # 9 checks: gates gate, failures classify, loop composes
python examples/run_local_sweep.py     # full loop: baseline -> 6 trials -> select -> final gate -> report
```

`run_local_sweep.py` is also the worked example: copy it and swap `run_trial()`
for your own `submit()` + `scrape()` to point the loop at a real trainer.

### Demonstrated backends

The loop is backend-agnostic, and that is shown, not asserted. The test suite
runs the same scripts against three different backends:

- **`local_backend.py`** — linear regression, objective `eval_loss` (**minimize**), diverges at a bad LR (proves the failure path).
- **`logreg_backend.py`** — a different algorithm (logistic classifier), objective `accuracy` (**maximize** — a different gate path).
- **`demo_trainer.sh`** — a **non-Python (bash)** trainer, proving the loop drives any command that reads params and prints a metric.

A real GPU backend (Kubeflow Trainer v2 + Kueue + MLflow) was validated
separately. Ray Tune / SLURM / Weights & Biases follow the same two-function
adapter shape.

Failure policy note: only **systemic** causes (data / image-cred / infra /
spec-schema) trigger the budget-preserving early stop, because a fresh trial will
also hit them. A **model-code** failure (e.g. a diverging learning rate) is
per-config, so the search keeps exploring rather than halting on one bad sample.

## Wiring your backend

The scripts speak flat key/value trial configs and a numeric metric. You supply two adapters:

| adapter | job | examples |
|---|---|---|
| `submit(config) -> run_id` | launch one training run | Kubeflow Trainer v2 `TrainJob`, Ray Tune, SLURM `sbatch`, local `subprocess` |
| `scrape(run_id) -> value` | read the objective metric | `mlflow_scrape.py` (reference), Weights & Biases, stdout parsing |

`tunable_env` in the schema declares which keys are searchable, so the validator is
not bound to any single trainer's parameter names.

## Files

```
SKILL.md                          # the agent contract
scripts/search_space_schema.py    # gate #1: structural + zero-step validation
scripts/hpo_gate.py               # gates #2-4, by exit code
scripts/launch_review.py          # gate #3: wall-time estimate + limit
scripts/trial_config_gen.py       # random / ASHA / Hyperband samplers
scripts/mlflow_scrape.py          # reference metric scraper (MLflow REST)
scripts/failure_classify.py       # gate #5: failure taxonomy + budget-preserving stop
scripts/select_best.py            # rank trials by objective
scripts/report.py                 # structured final report (incl. failures + root causes)
examples/search-space-example.json
examples/local_backend.py         # backend #1: linear regression, eval_loss (minimize)
examples/logreg_backend.py        # backend #2: logistic classifier, accuracy (maximize)
examples/demo_trainer.sh          # backend #3: non-Python (bash) trainer
examples/run_local_sweep.py       # end-to-end reference runner (copy + swap run_trial)
tests/smoke.py                    # 9 stdlib checks — no pytest, no deps
```

No dependencies beyond the Python standard library.

## Credits

- Gate discipline adapted from NVIDIA's [`tao-run-automl`](https://github.com/NVIDIA-TAO/tao-skill-bank) (Apache-2.0). See also [`NVIDIA/skills`](https://github.com/NVIDIA/skills).

## License

Apache-2.0.
