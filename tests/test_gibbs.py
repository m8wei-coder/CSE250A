from collections import Counter
from itertools import product
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from numpy.random import default_rng
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.exact import exact_inference
from alarm_inference.experiment import run_experiment
from alarm_inference.model import load_alarm_model
from alarm_inference.samplers.gibbs import _prepare, _Update, gibbs
from alarm_inference.samplers.lw import lw
from alarm_inference.samplers.rs import rs


@pytest.fixture
def toy_model():
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


def run(model, query="Q", evidence=None, budget=1000, seed=250, burn_in=0, trace=False, **settings):
    return gibbs(
        model, query=query, evidence={} if evidence is None else evidence,
        budget=budget, rng=default_rng(seed), record_trace=trace,
        settings={"burn_in_sweeps": burn_in, "init_max_attempts": 100, **settings},
    )


def test_gibbs_matches_hand_calculated_posterior(toy_model):
    result = run(toy_model, evidence={"E": "on"}, budget=20_000, burn_in=100)
    assert result.status == "ok"
    assert list(result.posterior) == ["yes", "no"]
    assert result.posterior == pytest.approx({"yes": 3 / 7, "no": 4 / 7}, abs=0.015)
    assert result.diagnostics["retained_sweeps"] == 20_000
    assert result.diagnostics["completed_sweeps"] == 20_100
    assert result.diagnostics["variable_updates"] == 40_200
    assert result.diagnostics["scan_order"] == ["Q", "D"]
    assert result.diagnostics["initial_state"]["E"] == "on"
    assert result.diagnostics["final_state"]["E"] == "on"
    assert result.query_trace is None


def test_gibbs_zero_states_and_empty_evidence(toy_model):
    fixed = run(toy_model, query="D", evidence={"E": "on"}, trace=True)
    assert fixed.posterior == {"on": 1.0, "off": 0.0}
    assert fixed.query_trace == ["on"] * 1000

    independent = DiscreteBayesianNetwork()
    independent.add_node("Q")
    independent.add_cpds(toy_model.get_cpds("Q").copy())
    result = run(independent, budget=20_000)
    assert result.posterior == pytest.approx({"yes": 0.2, "no": 0.8}, abs=0.015)


def test_burn_in_is_excluded_and_trace_does_not_change_stream(toy_model):
    complete = run(toy_model, evidence={"E": "on"}, budget=107, trace=True)
    retained = run(toy_model, evidence={"E": "on"}, budget=100, burn_in=7, trace=True)
    plain = run(toy_model, evidence={"E": "on"}, budget=100, burn_in=7)
    assert retained.query_trace == complete.query_trace[7:]
    counts = Counter(retained.query_trace)
    assert retained.posterior == {state: counts[state] / 100 for state in ["yes", "no"]}
    assert retained.posterior == plain.posterior
    assert retained.diagnostics == plain.diagnostics
    assert retained == run(toy_model, evidence={"E": "on"}, budget=100, burn_in=7, trace=True)
    json.dumps(retained.diagnostics, allow_nan=False)


def test_gibbs_retries_zero_weight_initialization():
    model = DiscreteBayesianNetwork([("Q", "E")])
    model.add_cpds(
        TabularCPD("Q", 2, [[0.5], [0.5]], state_names={"Q": ["yes", "no"]}),
        TabularCPD(
            "E", 2, [[1, 0], [0, 1]], evidence=["Q"], evidence_card=[2],
            state_names={"E": ["on", "off"], "Q": ["yes", "no"]},
        ),
    )
    failed = run(model, evidence={"E": "on"}, seed=0, init_max_attempts=1)
    assert failed.status == "gibbs_init_failed"
    # PCG64 seed 0 draws 0.6369 then 0.2698: the second LW attempt has support.
    success = run(model, evidence={"E": "on"}, seed=0, init_max_attempts=2)
    assert success.status == "ok"
    assert success.diagnostics["init_attempts"] == 2
    assert success.posterior == {"yes": 1.0, "no": 0.0}


def test_impossible_evidence_is_an_explicit_sampling_failure(toy_model):
    result = run(toy_model, evidence={"E": "off", "D": "on"}, init_max_attempts=7, trace=True)
    assert result.status == "gibbs_init_failed"
    assert result.posterior is result.query_trace is None
    assert result.diagnostics["init_attempts"] == 7
    assert result.diagnostics["retained_sweeps"] == 0
    assert result.diagnostics["completed_sweeps"] == 0
    assert result.diagnostics["initial_state"] is None
    assert result.diagnostics["failure_reason"] == "initialization_limit"
    json.dumps(result.diagnostics, allow_nan=False)


def test_positive_weight_does_not_fail_when_its_product_underflows():
    model = DiscreteBayesianNetwork([("Q", "E1"), ("Q", "E2")])
    model.add_cpds(TabularCPD("Q", 2, [[0.5], [0.5]], state_names={"Q": ["yes", "no"]}))
    for node in ["E1", "E2"]:
        model.add_cpds(TabularCPD(
            node, 2, [[1e-200, 2e-200], [1.0, 1.0]], evidence=["Q"], evidence_card=[2],
            state_names={node: ["rare", "common"], "Q": ["yes", "no"]},
        ))
    result = run(model, evidence={"E1": "rare", "E2": "rare"}, budget=20_000)
    assert result.status == "ok"
    assert result.diagnostics["init_attempts"] == 1
    assert result.diagnostics["initial_log_weight"] < -900
    assert result.posterior == pytest.approx({"yes": 0.2, "no": 0.8}, abs=0.015)


def test_cdf_preserves_leading_internal_and_trailing_zeros():
    model = DiscreteBayesianNetwork()
    model.add_node("Q")
    model.add_cpds(TabularCPD(
        "Q", 6, [[0.0], [0.1], [0.0], [0.2], [0.7], [0.0]],
        state_names={"Q": ["zero_first", "a", "zero_middle", "b", "c", "zero_last"]},
    ))
    _, tables = _prepare(model)
    update = _Update("Q", (), (tables["Q"],))
    for cdf in [tables["Q"].cdf, update.conditional_cdf({})]:
        assert cdf[0] == 0
        assert cdf[1] == cdf[2]
        assert cdf[-2] == cdf[-1] == 1
        assert cdf.searchsorted(0.0, side="right") == 1
        assert cdf.searchsorted(np.nextafter(1.0, 0.0), side="right") == 4


def test_every_local_conditional_matches_enumerated_joint():
    # Mixed cardinalities, unsorted labels, co-parents and deliberately reversed CPT parents.
    model = DiscreteBayesianNetwork([("A", "C"), ("Z", "C"), ("C", "D"), ("Z", "D")])
    labels = {"A": ["a1", "a0"], "Z": ["z2", "z0", "z1"], "C": ["c1", "c0"], "D": ["d1", "d0"]}
    model.add_cpds(
        TabularCPD("A", 2, [[0.3], [0.7]], state_names={"A": labels["A"]}),
        TabularCPD("Z", 3, [[0.1], [0.6], [0.3]], state_names={"Z": labels["Z"]}),
        TabularCPD(
            "C", 2, [[0.2, 0.9, 0.8, 0.1, 0.3, 0.7], [0.8, 0.1, 0.2, 0.9, 0.7, 0.3]],
            evidence=["Z", "A"], evidence_card=[3, 2],
            state_names={node: labels[node] for node in ["C", "Z", "A"]},
        ),
        TabularCPD(
            "D", 2, [[0.95, 0.4, 0.8, 0.1, 0.25, 0.6], [0.05, 0.6, 0.2, 0.9, 0.75, 0.4]],
            evidence=["Z", "C"], evidence_card=[3, 2],
            state_names={node: labels[node] for node in ["D", "Z", "C"]},
        ),
    )
    order, tables = _prepare(model)
    for node in order:
        factors = (tables[node], *(tables[child] for child in model.successors(node)))
        neighbors = tuple(sorted(set().union(*(cpt.variables for cpt in factors)) - {node}))
        update = _Update(node, neighbors, factors)
        others = [other for other in order if other != node]
        for assignment in product(*(range(len(labels[other])) for other in others)):
            state = dict(zip(others, assignment, strict=True))
            joint = []
            for index in range(len(labels[node])):
                values = {name: labels[name][i] for name, i in {**state, node: index}.items()}
                joint.append(np.prod([
                    cpd.get_value(**{name: values[name] for name in cpd.variables})
                    for cpd in model.get_cpds()
                ]))
            expected = np.asarray(joint) / sum(joint)
            actual = np.diff(np.r_[0.0, update.conditional_cdf(state)])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            # Cache hits must preserve the same exact conditional.
            np.testing.assert_allclose(np.diff(np.r_[0.0, update.conditional_cdf(state)]), expected)
    result = run(model, query="A", evidence={"D": "d0"}, budget=30_000, burn_in=100)
    assert result.posterior == pytest.approx(exact_inference(model, "A", {"D": "d0"}).posterior, abs=0.02)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget(toy_model, budget):
    with pytest.raises(ValueError, match="budget"):
        run(toy_model, budget=budget)


@pytest.mark.parametrize("settings, message", [
    ({"burn_in_sweeps": -1}, "burn_in_sweeps"),
    ({"burn_in_sweeps": True}, "burn_in_sweeps"),
    ({"init_max_attempts": 0}, "init_max_attempts"),
    ({"init_max_attempts": 1.5}, "init_max_attempts"),
    ({"initialization": "uniform"}, "initialization"),
    ({"sweep_order": "random"}, "sweep_order"),
])
def test_invalid_settings(toy_model, settings, message):
    with pytest.raises(ValueError, match=message):
        run(toy_model, **settings)


@pytest.mark.parametrize("query, evidence, message", [
    ("missing", {}, "Unknown query variable"),
    ("Q", {"missing": "on"}, "Unknown evidence variable"),
    ("Q", {"E": "ON"}, "Unknown state"),
    ("Q", {"Q": "yes"}, "Query variable also appears in evidence"),
])
def test_invalid_query_or_evidence(toy_model, query, evidence, message):
    with pytest.raises(ValueError, match=message):
        run(toy_model, query=query, evidence=evidence)


def test_missing_settings_are_not_silently_selected(toy_model):
    with pytest.raises(ValueError, match="burn_in_sweeps"):
        gibbs(toy_model, query="Q", evidence={}, budget=10, rng=default_rng(0), settings={})


def test_alarm_sampler_preserves_model_and_has_supported_endpoints():
    model, _ = load_alarm_model(Path(__file__).parents[1] / "data/alarm.bif")
    before = {cpd.variable: cpd.values.copy() for cpd in model.get_cpds()}
    result = run(model, "LVFAILURE", {"HISTORY": "TRUE"}, budget=20_000, burn_in=1000)
    assert result.status == "ok"
    exact = exact_inference(model, "LVFAILURE", {"HISTORY": "TRUE"})
    assert result.posterior == pytest.approx(exact.posterior, abs=0.04)
    assert result.diagnostics["scan_order"] == [
        node for node in nx.lexicographical_topological_sort(model) if node != "HISTORY"
    ]
    for cpd in model.get_cpds():
        np.testing.assert_array_equal(cpd.values, before[cpd.variable])
        for key in ["initial_state", "final_state"]:
            state = result.diagnostics[key]
            assert cpd.get_value(**{name: state[name] for name in cpd.variables}) > 0


def test_actual_samplers_integrate_with_shared_smoke_runner(tmp_path):
    output = run_experiment("smoke", {"rs": rs, "lw": lw, "gibbs": gibbs}, output_dir=tmp_path / "smoke")
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert [row["method"] for row in records] == ["rs", "lw", "gibbs"]
    row = records[-1]
    assert row["status"] == "ok"
    assert row["seed_inputs"] == [250, 0, 0, 2, 100, 0]
    assert row["budget_unit"] == "retained_sweeps"
    assert row["diagnostics"]["retained_sweeps"] == row["budget"] == 100
    assert row["runtime_seconds"] > 0
    assert row["tvd"] is not None
