# Likelihood Weighting — Junyi Xu

`lw.py` implements the shared `Sampler` interface in `experiment.py` and uses the repository's existing Python environment and dependencies.

Import it in the team's experiment entry point:

```python
from alarm_inference.samplers.lw import lw
```

Once RS and Gibbs are also available, use the shared entry point described in the repository's root README.

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
The main LW experiment contains 480 runs: four cases, six values of N, and 20 repetitions per combination. N counts generated samples.
Before the shared main experiment, finalize the Gibbs pilot decisions, fill in the corresponding settings, and set the main phase status to `ready`.

LW returns a posterior keyed by BIF state label, a run status, and weight diagnostics such as ESS. If all weights are zero, it returns `zero_total_weight` with an empty posterior.
The shared runner measures elapsed time, computes the exact reference and TVD, and saves `runs.jsonl` and `metadata.json`.
The implementation uses the supplied random number generator, fixed chunks of 8192 samples, and log weights. It does not smooth zero probabilities in the model.

Calls to the actual LW implementation have been validated through the unmodified shared runner. AI assistance was used for implementation and validation.

