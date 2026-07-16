---
name: automl-train
description: >-
  Schema-gated hyperparameter search (HPO/AutoML) skill for any training backend.
  Owns the closed loop — baseline eval, a validated search space, N trials, metric
  scrape, next-config recommendation, and an independent final eval of the best
  checkpoint — while deferring the single training run to whatever trainer you
  already have (Kubeflow Trainer v2, Ray Tune, SLURM, a local subprocess).
  Use when "hyperparameter search", "HPO", "AutoML", "tune the fine-tune", "sweep".
  Do NOT use for a single training run (call your trainer directly).
---

# automl-train — Schema-Gated HPO Loop

A thin orchestration layer that turns a single training run into a disciplined
hyperparameter search. The philosophy is borrowed from NVIDIA TAO's
`tao-run-automl` skill (Apache-2.0): **gate before you spend GPU**. What is new
here is that the four gates are owned by small deterministic scripts, so the
model recommends the next hyperparameters and nothing else — schema validation,
enum normalization, metric aggregation, and the promotion decision are code.

## Invariants (owned by code, not the model)

1. **No schema, no search.** `search_space_schema.py` must pass — a valid, bounded space with no zero-training-step configs — or the loop cannot start.
2. **Baseline before tuning.** `hpo_gate.py --check baseline` requires a recorded reference metric.
3. **Launch reviewed up front.** `--check budget` enforces `max_concurrent * gpus_per_trial <= gpu_cap`; `launch_review.py` estimates the wall-clock and refuses to start if it blows a stated limit.
4. **Independent final eval.** The winning trial must beat the baseline on the same eval set, or it is not promoted.
5. **Fail without burning budget.** `failure_classify.py` tags each failed trial (data / image-cred / infra / spec-schema / model-code); if consecutive trials share a systemic cause, the loop stops and reports instead of exhausting the GPU budget.

## Loop

```
schema-gate -> baseline(+gate) -> launch-review(wall-time, +gate) ->
  for each trial from trial_config_gen:
    submit to YOUR trainer -> scrape metric (or classify failure) ->
    if consecutive same-cause failures: stop early ->
  select_best -> final-eval on best -> final-gate -> report -> promote
```

The model owns: writing the schema, choosing the strategy (random / ASHA /
Hyperband), and reading gate results. Everything else is deterministic.

## Search strategies

- **random** — independent samples; the honest baseline searcher.
- **asha** — successive halving; promote the top `1/eta` each rung with more resource.
- **hyperband** — bracketed successive halving; each bracket trades width for min-resource, then promotes with the ASHA rungs.

## Scripts (stdlib-only, JSON I/O)

```bash
python scripts/search_space_schema.py --schema search-space.json          # gate #1
python scripts/hpo_gate.py --check baseline --baseline baseline.json      # gate #2
python scripts/hpo_gate.py --check budget   --schema search-space.json    # gate #3
python scripts/launch_review.py --schema search-space.json --per-trial-min 8 --baseline-min 5 --max-minutes 60
python scripts/trial_config_gen.py --schema search-space.json --strategy random --seed 7 --json
python scripts/trial_config_gen.py --schema search-space.json --strategy hyperband --R 27 --bracket 1 --json
python scripts/mlflow_scrape.py --run-id <id> --metric eval_loss          # or adapt to your tracker
python scripts/failure_classify.py classify --log trial.log               # data|image_cred|infra|spec_schema|model_code
python scripts/failure_classify.py policy --results results.json --consecutive 2   # stop on shared cause?
python scripts/select_best.py --results results.json --direction minimize --json
python scripts/hpo_gate.py --check final --baseline baseline.json --best best.json  # gate #4
python scripts/report.py --results results.json --baseline baseline.json --schema search-space.json
```

## Wiring your backend (the one part you write)

The scripts emit trial configs as flat key/value dicts and read a numeric metric
back. You provide two small adapters:

- **submit(env) -> run_id**: launch one training run with the trial's params.
  Kubeflow Trainer v2 (render a TrainJob with the env), Ray Tune (`tune.run`),
  SLURM (`sbatch` with exported vars), or a local `subprocess`.
- **scrape(run_id) -> value**: read the objective metric. `mlflow_scrape.py` is
  the reference (MLflow REST); Weights & Biases or parsing stdout are analogous.

`tunable_env` in the schema declares which keys are searchable, so the validator
is not tied to any one trainer's parameter names. See `examples/`.

## Not in scope

This is the search loop and the gates. It does not deploy a controller, own your
cluster scheduler, or serve the model — it drives the trainer you already run.
