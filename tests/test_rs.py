from pathlib import Path

import pytest
from numpy.random import Generator, PCG64, SeedSequence
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.exact import exact_inference
from alarm_inference.model import load_alarm_model
from alarm_inference.samplers.rs import rs


@pytest.fixture
def toy_model() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork([("Q", "E"), ("E", "D")])
    model.add_cpds(
        TabularCPD("Q", 2, [[0.2], [0.8]], state_names={"Q": ["yes", "no"]}),
        TabularCPD(
            "E", 2, [[0.75, 0.25], [0.25, 0.75]],
            evidence=["Q"], evidence_card=[2],
            state_names={"E": ["on", "off"], "Q": ["yes", "no"]},
        ),
        TabularCPD(
            "D", 2, [[1.0, 0.0], [0.0, 1.0]],
            evidence=["E"], evidence_card=[2],
            state_names={"D": ["on", "off"], "E": ["on", "off"]},
        ),
    )
    return model


def make_rng(seed: int) -> Generator:
    return Generator(PCG64(SeedSequence(seed)))


def run(model, query, evidence, budget, seed=0, settings=None):
    return rs(
        model, query=query, evidence=evidence, budget=budget, rng=make_rng(seed),
        settings=settings or {}, record_trace=False,
    )


def test_rs_matches_exact_posterior_and_acceptance(toy_model):
    budget = 200_000
    result = run(toy_model, "Q", {"E": "on"}, budget)

    assert result.status == "ok"
    assert list(result.posterior) == ["yes", "no"]
    assert result.posterior == pytest.approx({"yes": 3 / 7, "no": 4 / 7}, abs=0.01)
    diagnostics = result.diagnostics
    assert diagnostics["num_samples"] == budget
    assert diagnostics["accepted"] + diagnostics["rejected"] == budget
    assert diagnostics["acceptance_rate"] == pytest.approx(0.35, abs=0.005)
    assert result.query_trace is None


def test_rs_keeps_zero_probability_states(toy_model):
    result = run(toy_model, "D", {"Q": "yes", "E": "on"}, 10_000)

    assert result.status == "ok"
    assert result.posterior == {"on": 1.0, "off": 0.0}


def test_rs_reports_zero_accepts_for_impossible_evidence(toy_model):
    result = run(toy_model, "Q", {"E": "off", "D": "on"}, 5_000)

    assert result.status == "zero_accepts"
    assert result.posterior is None
    assert result.diagnostics["accepted"] == 0
    assert result.diagnostics["acceptance_rate"] == 0.0


def test_rs_is_reproducible_and_respects_chunk_size(toy_model):
    first = run(toy_model, "Q", {"E": "on"}, 1_000, seed=7)
    second = run(toy_model, "Q", {"E": "on"}, 1_000, seed=7)
    chunked = run(toy_model, "Q", {"E": "on"}, 1_000, seed=7, settings={"chunk_size": 64})

    assert first == second
    assert chunked.diagnostics["num_samples"] == 1_000
    assert chunked.diagnostics["chunk_size"] == 64


@pytest.mark.parametrize(
    ("query", "evidence", "budget", "message"),
    [
        ("unknown", {}, 10, "Unknown query variable"),
        ("Q", {"unknown": "on"}, 10, "Unknown evidence variable"),
        ("Q", {"E": "ON"}, 10, "Unknown state"),
        ("Q", {"Q": "yes"}, 10, "Query variable also appears in evidence"),
        ("Q", {}, 0, "budget"),
        ("Q", {}, True, "budget"),
    ],
)
def test_rs_rejects_invalid_inputs(toy_model, query, evidence, budget, message):
    with pytest.raises(ValueError, match=message):
        run(toy_model, query, evidence, budget)


def test_rs_alarm_smoke_case_close_to_exact():
    model, _ = load_alarm_model(Path(__file__).parents[1] / "data" / "alarm.bif")
    evidence = {"HISTORY": "TRUE"}
    exact = exact_inference(model, "LVFAILURE", evidence)
    result = run(model, "LVFAILURE", evidence, 100_000, seed=250)

    assert result.status == "ok"
    assert result.posterior.keys() == exact.posterior.keys()
    tvd = 0.5 * sum(abs(result.posterior[s] - exact.posterior[s]) for s in exact.posterior)
    assert tvd < 0.02
    assert result.diagnostics["acceptance_rate"] == pytest.approx(exact.p_evidence, abs=0.003)
