# Security Policy

## Supported versions

This project ships from `main`. Fixes land on `main`; there are no long-lived
release branches.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

- Preferred: open a private report via GitHub Security Advisories
  ("Report a vulnerability" under the repo's Security tab).
- Or email the maintainer: thakiaiplatform@gmail.com

Include what you found, how to reproduce it, and the impact. You will get an
acknowledgement within a few days. Once a fix is ready we will credit you unless
you prefer to stay anonymous.

## Scope notes

- This repo is **standard-library only** — there are no third-party runtime
  dependencies to pull in a supply-chain vulnerability.
- `mlflow_scrape.py` reads secrets from environment variables
  (`MLFLOW_TRACKING_TOKEN`, `MLFLOW_WORKSPACE_NAME`) and never logs them. If you
  wire your own backend adapter, keep credentials in the environment, not in the
  search-space JSON or committed files.
- Secret scanning and push protection are enabled on this repository.
