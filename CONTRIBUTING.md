# Contributing

## Branches

- Start each task from `main` on a short-lived branch such as `feat/exact-inference` or `feat/gibbs`.

## Commit messages

- Keep each commit focused on one change. Use `type: short description`.
- Use `feat` for new behavior, `fix` for bug fixes, `refactor` for internal changes that preserve behavior, `test` for test-only changes, `docs` for documentation, and `chore` for setup or maintenance.
- Examples include `feat: compute exact evidence probability`, `fix: handle zero total sample weight`, and `docs: add ALARM source information`.

## Project layout



Use Python and record the supported Python version and dependency versions with the environment setup. Document the install, test, and single-case reproduction commands in the README as they become available.

## Python style

- Use four spaces for indentation, `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Group imports by standard library, third-party packages, and project modules, in that order.
- Add parameter and return type hints to shared functions. Use short docstrings for behavior callers need to know, including what a sample budget counts or when a result can be undefined.
- Use consistent names across samplers. For example, use `query`, `evidence`, and `seed` for the shared inputs.

## Shared inference interface

- Use the same ALARM model for all methods. Preserve the BIF variable names, state labels, and CPT parent ordering. Validate evidence assignments and keep the query variable separate from evidence.
- Use Variable Elimination to compute both the reference posterior and the exact evidence probability. Treat zero-probability evidence as undefined for conditional inference.
- Return the posterior keyed by state label, an explicit run status, and method diagnostics. Keep failed runs in the raw results with a reason and an empty posterior. Compute total variation distance only when a run returns a valid posterior.

## Tests and experiments

- Before merging inference changes, run `pytest`. Use a hand-checkable toy network to verify the exact posterior and evidence probability. Check sampler behavior on that network before running the shared ALARM integration case.
- Keep the two common and rare case pairs in a shared config after the pilot. Record pilot decisions, including Gibbs burn-in and initialization limits, before the main runs. Use separate random streams for pilot and main runs.
- Count generated samples for Rejection Sampling and Likelihood Weighting. For Gibbs, count retained sweeps after burn-in and record burn-in separately.
- Use a shared timer for all methods. Include method setup, initialization, burn-in, and posterior estimation. Exclude shared model loading and exact inference.
- Record the case, method, budget, seed, settings, runtime, run status, and method diagnostics for every run. Save the model checksum, dependency versions, and code revision with the experiment results. Summarize TVD across valid runs and failure rates across all runs.
