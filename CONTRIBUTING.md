# Contributing

Thanks for your interest. This is a small, deliberately dependency-free project;
contributions that keep it that way are the easiest to accept.

## Ground rules

- **Standard library only.** No third-party runtime dependencies. If a change
  seems to need one, open an issue first to discuss.
- **The gates are the point.** Changes must not weaken the five invariants
  (schema, baseline, launch review, independent final eval, failure-aware stop).
  A gate that can be silently bypassed is a bug.
- **Deterministic scripts.** Format, validation, and aggregation live in code, not
  in model output. Keep it that way.

## Before you open a PR

```bash
python tests/smoke.py              # must be green (9/9)
python examples/run_local_sweep.py # end-to-end must still compose + promote
python -m py_compile scripts/*.py examples/*.py tests/*.py
```

If you add a behavior, add a check to `tests/smoke.py` that would fail without it.

## Style

- Plain, readable Python. Small functions. Clear error messages on the gate paths.
- JSON in, JSON out for scripts; one-line status where practical.
- No secrets in code, tests, or fixtures.

## Reporting bugs / ideas

Open an issue with a minimal repro (a search-space JSON and the command you ran).
For security issues see [SECURITY.md](SECURITY.md) instead.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
