# Comparing Sampling Methods on ALARM

CSE 250A final project on approximate inference in discrete Bayesian networks.

We compare Rejection Sampling, Likelihood Weighting, and Gibbs Sampling on ALARM under different sampling budgets and evidence probabilities. Variable Elimination provides the reference posterior and exact evidence probability. We measure total variation distance from the reference posterior, runtime, and failure rate.

## Data

The [ALARM Bayesian network](https://www.bnlearn.com/bnrepository/discrete-medium.html#alarm) has 37 variables and 46 directed edges. The fixed network structure, state labels, and conditional probability tables are stored in [`data/alarm.bif`](data/alarm.bif).

From the repository root, check its SHA-256 checksum with the following command.

```sh
LC_ALL=C shasum -a 256 -c <<'EOF'
701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2  data/alarm.bif
EOF
```

The ALARM network was introduced by Beinlich, Suermondt, Chavez, and Cooper in *The ALARM Monitoring System*, AIME 1989.

## Experiment plan

- Evaluate two single-variable queries with one common and one rare evidence assignment per query. Use exact evidence probabilities and pilot runs to select the cases.
- Use planned budgets of 100, 1,000, 5,000, 10,000, 50,000, and 100,000. Count generated samples for Rejection Sampling and Likelihood Weighting, and retained sweeps after burn-in for Gibbs.
- Run 20 independent random streams per combination of case, method, and budget. Use separate streams for the pilot, which will also inform Gibbs burn-in and initialization settings.

See the [case selection rationale](docs/QUERY_EVIDENCE_SELECTION.md) and [experiment configuration](configs/experiment.toml). The configuration records the smoke, pilot, and main settings. The pilot records full Gibbs traces from several starting states to choose the main burn-in and initialization limits. Complete those settings before starting main runs.

Measure elapsed time from the start of method setup until the method returns a posterior estimate or reports failure. Include initialization and Gibbs burn-in. Exclude shared model loading, exact inference, TVD calculation, and result writing. Record elapsed time for failed runs as well.

## Experiment runner

Call `alarm_inference.experiment.run_experiment` with a phase and a mapping of sampler functions. Each function receives the shared model and the keyword arguments defined by the `Sampler` protocol. Perform method setup inside the function so the runner can time it. The sampler implementations still need to be connected to this Python entry point.

```python
from alarm_inference.experiment import run_experiment

run_experiment("smoke", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

Return a `SamplerResult` with an `ok` status and a posterior keyed by BIF state label on success. Use `zero_accepts`, `zero_total_weight`, or `gibbs_init_failed` with a `None` posterior for expected sampling failures. Include JSON-compatible method diagnostics. For a Gibbs pilot, return one query state per sweep in `query_trace` and include the initial state and number of initialization attempts in the diagnostics.

The runner saves the config, code revision, dependency versions, and exact references in `metadata.json`. It writes one record per run to `runs.jsonl`, including failures with empty posterior and TVD fields. Use a new `output_dir` for a rerun to preserve prior results. The main phase requires the post-pilot Gibbs settings and `status = "ready"` in the configuration.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `data/` | ALARM BIF source |
| `configs/` | Case definitions, sample budgets, seeds, and method settings |
| `src/alarm_inference/` | Shared model loading and validation, exact inference, and benchmarking |
| `src/alarm_inference/samplers/` | Rejection Sampling, Likelihood Weighting, and Gibbs Sampling in separate modules |
| `tests/` | Tests for a hand-checkable toy network and an ALARM integration case |
| `results/` | Generated run records and summaries |

## Python environment

Use Python 3.12 and `uv`. From the repository root, create the local `.venv` and install the versions recorded in `uv.lock`.

```sh
uv sync --locked
uv run --locked python --version
```

Use `uv run` to run commands inside the project environment. For example, `uv run --locked pytest --version` checks that the test runner is installed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, commit messages, and shared inference conventions.

## Exact inference smoke test

Run this check from the repository root after setting up `.venv`.

```sh
.venv/bin/python -m pytest -q tests/test_exact.py::test_alarm_exact_smoke
```

The test loads the validated ALARM model and queries `LVFAILURE` given `HISTORY=TRUE`. The evidence probability is `0.0545`. The posterior is approximately `0.825688` for `TRUE` and `0.174312` for `FALSE`.
