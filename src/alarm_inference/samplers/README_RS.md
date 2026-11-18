# Rejection Sampling — Sai Leng

`rs.py` implements the shared `Sampler` interface in `experiment.py` and uses the repository's existing Python environment and dependencies.

Import it in the team's experiment entry point:

```python
from alarm_inference.samplers.rs import rs
```

Use the shared entry point described in the repository's root README with LW and Gibbs.

From `CSE250A/`, start Python with `PYTHONPATH=src .venv/bin/python`, then enter the Python code below. See the [repository README](../../../README.md) for the complete shell command.

```python
from alarm_inference.experiment import run_experiment
from alarm_inference.samplers.rs import rs
from alarm_inference.samplers.lw import lw
from alarm_inference.samplers.gibbs import gibbs

run_experiment("smoke", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

## Algorithm

Forward-sample all 37 nodes in lexicographic topological order. Evidence nodes are sampled normally rather than clamped. Retain only assignments that match all evidence, and estimate the posterior from the query-state frequencies among accepted assignments.
The implementation neither prunes irrelevant nodes nor stops early when evidence is inconsistent. As in LW, every generated assignment processes the full network; comparisons of elapsed time require the same execution environment.

- N counts generated samples, not accepted samples. The main RS experiment contains 480 runs: four cases, six values of N, and 20 repetitions per combination.
- Successful runs return a posterior keyed by BIF state label. Query states that were not observed have probability `0.0`.
- If no samples are accepted, the method returns `zero_accepts` with an empty posterior. This is an expected sampling failure and is included in the runner's failure rate.
- Diagnostic fields are `num_samples`, `accepted`, `rejected`, `acceptance_rate`, `chunk_size`, and `scan_order`.
- The implementation uses the supplied random number generator and fixed chunks of 8192 samples. Changing `chunk_size` changes the order in which the random stream is consumed.
- The method does not return `query_trace`. Invalid inputs raise `ValueError`.

## Validation

`tests/test_rs.py` checks the posterior, acceptance rate, zero-probability states, `zero_accepts`, reproducibility, and input validation on a toy network. It also compares RS with VE for the ALARM smoke case.

```sh
.venv/bin/python -m pytest -q tests/test_rs.py
```

Additional validation completed the smoke and pilot phases through the unmodified shared runner. With 3,000,000 generated samples per main case, acceptance rates differed from the exact P(e) by less than one standard deviation, and TVD ranged from 0.0003 to 0.0070.
For `lvedvolume_rare`, 2000 runs each at N=100 and N=1000 gave zero-acceptance fractions of 0.8315 and 0.1625, respectively. The theoretical values from \((1-P(e))^N\) are 0.8345 and 0.1637.

AI assistance was used for implementation and validation.

