from pathlib import Path

import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.exact import exact_inference
from alarm_inference.model import load_alarm_model


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


@pytest.mark.parametrize(
    ("query", "evidence", "expected_p_evidence", "expected_posterior"),
    [
        ("Q", {}, 1.0, {"yes": 0.2, "no": 0.8}),
        ("Q", {"E": "on"}, 0.35, {"yes": 3 / 7, "no": 4 / 7}),
        ("D", {"Q": "yes", "E": "on"}, 0.15, {"on": 1.0, "off": 0.0}),
    ],
    ids=["empty-evidence", "one-observation", "two-observations"],
)
def test_exact_matches_hand_calculation(
    toy_model: DiscreteBayesianNetwork,
    query: str,
    evidence: dict[str, str],
    expected_p_evidence: float,
    expected_posterior: dict[str, float],
) -> None:
    result = exact_inference(toy_model, query, evidence)

    assert result.status == "ok"
    assert result.p_evidence == pytest.approx(expected_p_evidence)
    assert result.posterior == pytest.approx(expected_posterior)


def test_exact_has_no_posterior_for_impossible_evidence(toy_model: DiscreteBayesianNetwork) -> None:
    result = exact_inference(toy_model, "Q", {"E": "off", "D": "on"})

    assert result.status == "zero_probability_evidence"
    assert result.p_evidence == 0.0
    assert result.posterior is None


@pytest.mark.parametrize(
    ("query", "evidence", "message"),
    [
        ("unknown", {}, "Unknown query variable"),
        ("Q", {"unknown": "on"}, "Unknown evidence variable"),
        ("Q", {"E": "ON"}, "Unknown state"),
        ("Q", {"Q": "yes"}, "Query variable also appears in evidence"),
    ],
)
def test_exact_rejects_invalid_inputs(
    toy_model: DiscreteBayesianNetwork, query: str, evidence: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        exact_inference(toy_model, query, evidence)


def test_alarm_exact_smoke() -> None:
    data_path = Path(__file__).parents[1] / "data" / "alarm.bif"
    model, _ = load_alarm_model(data_path)

    result = exact_inference(model, "LVFAILURE", {"HISTORY": "TRUE"})

    assert result.status == "ok"
    assert result.p_evidence == pytest.approx(0.0545)
    assert result.posterior == pytest.approx({"TRUE": 0.045 / 0.0545, "FALSE": 0.0095 / 0.0545})
