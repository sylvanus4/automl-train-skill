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

## Four invariants (owned by code, not the model)

1. **No schema, no search.** `search_space_schema.py` must pass or the loop cannot start.
2. **Baseline before tuning.** `hpo_gate.py --check baseline` requires a recorded reference metric.
3. **Budget reviewed up front.** `--check budget`: `max_concurrent * gpus_per_trial <= gpu_cap`.
4. **Independent final eval.** The winning trial must beat the baseline on the same eval set, or it is not promoted.

## Loop

```
schema-gate -> baseline(+gate) -> budget-gate(review) ->
  for each trial from trial_config_gen:  submit to YOUR trainer -> scrape metric ->
  select_best -> final-eval on best -> final-gate -> promote
```

The model owns: writing the schema, choosing the strategy (random / ASHA), and
reading gate results. Everything else is deterministic.

## Scripts (stdlib-only, JSON I/O)

```bash
python scripts/search_space_schema.py --schema search-space.json          # gate #1
python scripts/hpo_gate.py --check schema   --schema search-space.json
python scripts/hpo_gate.py --check baseline --baseline baseline.json      # gate #2
python scripts/hpo_gate.py --check budget   --schema search-space.json    # gate #3
python scripts/trial_config_gen.py --schema search-space.json --strategy random --seed 7 --json
python scripts/trial_config_gen.py --schema search-space.json --strategy asha --rung 1 --results results.json --json
python scripts/mlflow_scrape.py --run-id <id> --metric eval_loss          # or adapt to your tracker
python scripts/select_best.py --results results.json --direction minimize --json
python scripts/hpo_gate.py --check final --baseline baseline.json --best best.json  # gate #4
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
