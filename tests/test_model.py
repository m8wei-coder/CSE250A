from pathlib import Path

import numpy as np
import pytest
from pgmpy.readwrite import BIFReader

from alarm_inference.model import load_alarm_model


DATA_PATH = Path(__file__).parents[1] / "data" / "alarm.bif"


def test_load_alarm_preserves_labels_parent_order_and_probabilities() -> None:
    model, report = load_alarm_model(DATA_PATH)

    assert report.node_count == 37
    assert report.edge_count == 46
    assert report.checksum == "701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2"
    assert len(report.cpt_sum_deviations) == 6
    assert len(report.zero_entries) == 5
    cpd = model.get_cpds("LVEDVOLUME")
    assert cpd.variables == ["LVEDVOLUME", "HYPOVOLEMIA", "LVFAILURE"]
    assert cpd.state_names["LVEDVOLUME"] == ["LOW", "NORMAL", "HIGH"]
    assert cpd.get_value(LVEDVOLUME="HIGH", HYPOVOLEMIA="TRUE", LVFAILURE="FALSE") == pytest.approx(0.90)
    assert cpd.get_value(LVEDVOLUME="HIGH", HYPOVOLEMIA="FALSE", LVFAILURE="TRUE") == pytest.approx(0.01)


def test_load_alarm_normalizes_only_the_six_reported_parent_assignments() -> None:
    original = BIFReader(path=str(DATA_PATH)).get_model()
    model, _ = load_alarm_model(DATA_PATH)

    for original_cpd in original.get_cpds():
        cpd = model.get_cpds(original_cpd.variable)
        remaining_values = cpd.values.copy()
        if cpd.variable in {"HREKG", "HRSAT"}:
            for errcauter, hr in [("TRUE", "LOW"), ("FALSE", "LOW"), ("TRUE", "NORMAL")]:
                index = (slice(None), cpd.get_state_no("ERRCAUTER", errcauter), cpd.get_state_no("HR", hr))
                np.testing.assert_allclose(cpd.values[index], [1 / 3] * 3, rtol=0, atol=1e-15)
                remaining_values[index] = original_cpd.values[index]
        np.testing.assert_array_equal(remaining_values, original_cpd.values)
