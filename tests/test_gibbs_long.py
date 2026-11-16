"""Exercise consecutive long ALARM chains, beyond the short smoke tests."""

from collections import Counter
from pathlib import Path

from numpy.random import Generator, PCG64, SeedSequence

from alarm_inference.model import load_alarm_model
from alarm_inference.samplers.gibbs import gibbs


def test_consecutive_long_alarm_chains_preserve_trace_and_evidence():
    # Use the first two chains from the supplemental pilot. A reported runtime
    # failure occurred after its first 100,000-sweep chain had completed.
    model, _ = load_alarm_model(Path(__file__).parents[1] / "data/alarm.bif")
    evidence = {"STROKEVOLUME": "LOW", "CO": "LOW"}
    settings = {"burn_in_sweeps": 0, "init_max_attempts": 5000}
    budget = 100_000
    results = []
    for repeat in range(2):
        rng = Generator(PCG64(SeedSequence([250, 3, 0, 2, budget, repeat])))
        result = gibbs(
            model, query="LVEDVOLUME", evidence=evidence, budget=budget,
            rng=rng, settings=settings, record_trace=True,
        )
        assert result.status == "ok"
        assert len(result.query_trace) == budget
        assert result.diagnostics["retained_sweeps"] == budget
        assert result.diagnostics["completed_sweeps"] == budget
        assert result.diagnostics["variable_updates"] == budget * (len(model.nodes) - len(evidence))
        assert set(result.posterior) == {"LOW", "NORMAL", "HIGH"}
        counts = Counter(result.query_trace)
        assert set(counts) <= set(result.posterior)
        assert result.posterior == {label: counts[label] / budget for label in result.posterior}
        for endpoint in ["initial_state", "final_state"]:
            state = result.diagnostics[endpoint]
            for node, value in evidence.items():
                assert state[node] == value
            for cpd in model.get_cpds():
                assert cpd.get_value(**{name: state[name] for name in cpd.variables}) > 0
        results.append(result)

    # Reusing a seed after the long runs must reproduce its prefix with a fresh
    # cache; previous chains must not alter the model or subsequent random stream.
    for repeat, result in enumerate(results):
        rng = Generator(PCG64(SeedSequence([250, 3, 0, 2, budget, repeat])))
        prefix = gibbs(
            model, query="LVEDVOLUME", evidence=evidence, budget=128,
            rng=rng, settings=settings, record_trace=True,
        )
        assert prefix.status == "ok"
        assert prefix.query_trace == result.query_trace[:128]
        assert prefix.diagnostics["initial_state"] == result.diagnostics["initial_state"]
