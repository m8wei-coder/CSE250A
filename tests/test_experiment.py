import json

import pytest

import alarm_inference.experiment as experiment


def test_smoke_records_timed_successes_and_failures(tmp_path, monkeypatch):
    ticks = iter([10.0, 12.0, 20.0, 23.0, 30.0, 35.0])
    inside_sampler = False

    def clock():
        nonlocal inside_sampler
        inside_sampler = not inside_sampler
        return next(ticks)

    monkeypatch.setattr(experiment, "perf_counter", clock)

    def rs(model, *, query, evidence, budget, rng, settings, record_trace):
        assert inside_sampler
        assert model.number_of_nodes() == 37
        assert (query, evidence, budget) == ("LVFAILURE", {"HISTORY": "TRUE"}, 100)
        return experiment.SamplerResult("zero_accepts", None, {"accepted": 0})

    def success(model, **kwargs):
        assert inside_sampler
        return experiment.SamplerResult("ok", {"FALSE": 0.25, "TRUE": 0.75}, {})

    output = experiment.run_experiment(
        "smoke", {"rs": rs, "lw": success, "gibbs": success}, output_dir=tmp_path / "smoke"
    )

    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert [row["runtime_seconds"] for row in records] == [2.0, 3.0, 5.0]
    rs_row, lw_row, gibbs_row = records
    assert rs_row["status"] == "zero_accepts"
    assert rs_row["diagnostics"] == {"accepted": 0}
    assert rs_row["posterior"] is rs_row["tvd"] is None
    assert lw_row["seed_inputs"] == [250, 0, 0, 1, 100, 0]
    assert lw_row["tvd"] == pytest.approx(0.0756880733944955)
    assert gibbs_row["budget_unit"] == "retained_sweeps"
    assert gibbs_row["settings"]["burn_in_sweeps"] == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["references"]["alarm_smoke"]["p_evidence"] == pytest.approx(0.0545)


def test_pilot_saves_full_gibbs_traces_on_separate_streams(tmp_path):
    def rs(model, **kwargs):
        return experiment.SamplerResult("zero_accepts", None, {"accepted": 0})

    def lw(model, **kwargs):
        return experiment.SamplerResult("zero_total_weight", None, {"total_weight": 0.0})

    def gibbs(model, *, query, budget, rng, settings, record_trace, **kwargs):
        assert (budget, settings["burn_in_sweeps"], record_trace) == (10000, 0, True)
        states = model.states[query]
        state = states[int(rng.integers(len(states)))]
        posterior = {label: float(label == state) for label in states}
        return experiment.SamplerResult("ok", posterior, {}, [state] * budget)

    output = experiment.run_experiment(
        "pilot", {"rs": rs, "lw": lw, "gibbs": gibbs}, output_dir=tmp_path / "pilot"
    )

    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 140
    assert len({tuple(row["seed_inputs"]) for row in records}) == 140
    chains = [row for row in records if row["method"] == "gibbs"]
    assert len(chains) == 20
    assert chains[-1]["seed_inputs"] == [250, 1, 3, 2, 10000, 4]
    assert all(len(row["query_trace"]) == 10000 for row in chains)
    assert {row["budget"] for row in records if row["method"] != "gibbs"} == {100, 1000, 5000}


def test_main_waits_for_pilot_decisions(tmp_path):
    output = tmp_path / "main"
    with pytest.raises(ValueError, match="pending pilot"):
        experiment.run_experiment("main", {}, output_dir=output)
    assert not output.exists()
