# Gibbs Sampling — Yidong Shi

`gibbs.py` implements the shared `Sampler` interface in `experiment.py` and uses the repository's existing Python environment and dependencies.

Import it in the team's experiment entry point:

```python
from alarm_inference.samplers.gibbs import gibbs
```

Use the shared entry point described in the repository's root README with RS and LW.

From `CSE250A/`, start Python with `PYTHONPATH=src .venv/bin/python`, then enter the Python code below. See the [repository README](../../../README.md) for the complete shell command.

```python
from alarm_inference.experiment import run_experiment
from alarm_inference.samplers.rs import rs
from alarm_inference.samplers.lw import lw
from alarm_inference.samplers.gibbs import gibbs

run_experiment("smoke", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

The first argument selects `smoke`, `pilot`, or `main`. Each call runs all methods and experiment combinations configured for that phase.
Use a new `output_dir` when repeating a run.
Before the main experiment, record the pilot decisions, fill in `burn_in_sweeps` and `init_max_attempts`, and set the phase status to `ready`.

## Algorithm

Clamp the evidence and initialize with the first complete LW assignment that has positive weight. Each sweep updates every unobserved node in lexicographic topological order.
The local conditional distribution is proportional to the node's CPT multiplied by its children's CPT factors, using the most recently updated values of the other nodes.
Factors are computed in log space. Zero probabilities are not smoothed, and BIF parent-axis and state ordering are preserved.

- N counts complete sweeps retained after burn-in. The query state is recorded once per sweep. The main experiment contains 480 runs: four cases, six values of N, and 20 repetitions per combination.
- The supplied random number generator is used throughout. Conditional CDFs are cached by Markov blanket assignment, with at most 4096 entries per node. Each call starts with a fresh cache.
- Successful runs return a posterior keyed by BIF state label. States that were not observed have probability `0.0`.
- If initialization reaches its attempt limit, the method returns `gibbs_init_failed` with an empty posterior. This does not imply that the evidence probability is zero.
- With `record_trace=True`, the method returns a `query_trace` of length N; otherwise, it returns `None`. Recording the trace does not alter the random stream.
- Diagnostics include initialization attempts, the initial log weight, initial and final states, update order, burn-in, retained and completed sweeps, and the number of variable updates.
- Invalid inputs raise `ValueError`. If a local conditional distribution has no support after valid initialization, the method raises `RuntimeError`.

The shared runner measures elapsed time, computes the exact reference and TVD, and saves `runs.jsonl` and `metadata.json`.
Method preparation, initialization, burn-in, and posterior estimation are included in the measured time.

## Validation

`tests/test_gibbs.py` checks a hand-computed posterior, local conditional distributions obtained by enumeration, evidence clamping, zero probabilities, initialization retries and failure, very small probabilities, burn-in, reproducibility, input validation, and ALARM integration.
`tests/test_gibbs_long.py` runs two 100,000-sweep ALARM chains and checks states, counts, and reproducible prefixes.

```sh
.venv/bin/python -m pytest -q tests/test_gibbs.py tests/test_gibbs_long.py
.venv/bin/python -m pytest -q
```

The integration test calls the existing `run_experiment("smoke", ...)` directly with the actual RS, LW, and Gibbs implementations. It checks the seeds, budgets, posteriors, and diagnostics saved by the shared runner. No additional runner script or changes to the shared configuration are required.

Additional validation used separate long-chain pilot settings: seed namespace 3, 100,000 sweeps per chain, no burn-in, and a limit of 5000 initialization attempts. These settings describe the completed validation and do not override the repository's default pilot configuration.
All 51 tests passed on Ubuntu with Python 3.12.14, NumPy 2.5.4, and pgmpy 1.1.2. The unmodified shared runner also completed five 100,000-sweep Gibbs chains for each of the four cases.

AI assistance was used for implementation and validation.

