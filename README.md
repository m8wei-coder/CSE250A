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
