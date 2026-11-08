"""Run the configured ALARM benchmark with the supplied samplers."""

import json
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import fsum, isclose
from pathlib import Path
from platform import python_version
from time import perf_counter
from typing import Literal, Protocol

from numpy.random import Generator, PCG64, SeedSequence
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.exact import exact_inference
from alarm_inference.model import load_alarm_model


REPO_ROOT = Path(__file__).resolve().parents[2]
Phase = Literal["smoke", "pilot", "main"]
RunStatus = Literal["ok", "zero_accepts", "zero_total_weight", "gibbs_init_failed"]


@dataclass(frozen=True)
class SamplerResult:
    status: RunStatus
    posterior: dict[str, float] | None
    diagnostics: dict
    query_trace: list[str] | None = None


class Sampler(Protocol):
    def __call__(
        self,
        model: DiscreteBayesianNetwork,
        *,
        query: str,
        evidence: Mapping[str, str],
        budget: int,
        rng: Generator,
        settings: Mapping[str, str | int],
        record_trace: bool,
    ) -> SamplerResult:
        """Set up the method and return an estimate or an explicit failure."""


def run_experiment(
    phase: Phase,
    samplers: Mapping[str, Sampler],
    *,
    config_path: Path = REPO_ROOT / "configs/experiment.toml",
    output_dir: Path | None = None,
) -> Path:
    """Write metadata and one JSONL record per run to a new output directory."""
    if phase not in ("smoke", "pilot", "main"):
        raise ValueError(f"Unknown experiment phase {phase!r}")
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    phase_config = config[phase]
    if phase == "main" and phase_config.get("status") != "ready":
        raise ValueError("Main experiment is pending pilot decisions")
    gibbs_settings = {**config["gibbs"], **phase_config["gibbs"]}
    if not {"burn_in_sweeps", "init_max_attempts"} <= gibbs_settings.keys():
        raise ValueError("Gibbs requires burn_in_sweeps and init_max_attempts")

    methods = config["methods"]
    missing = set(methods) - samplers.keys()
    if missing:
        raise ValueError(f"Missing samplers: {', '.join(sorted(missing))}")
    if config["rng"]["bit_generator"] != "PCG64":
        raise ValueError("The experiment requires the configured PCG64 bit generator")

    model, report = load_alarm_model(REPO_ROOT / config["model"]["path"])
    if report.checksum != config["model"]["sha256"]:
        raise ValueError("Loaded ALARM model does not match the configured checksum")

    case_ids = phase_config["case_ids"]
    references = {}
    for case_id in case_ids:
        reference = exact_inference(model, **config["cases"][case_id])
        if reference.posterior is None:
            raise ValueError(f"Case {case_id!r} has zero-probability evidence")
        references[case_id] = reference

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT))
    metadata = {
        "phase": phase,
        "config": config,
        "model_sha256": report.checksum,
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "versions": {"python": python_version(), "numpy": version("numpy"), "pgmpy": version("pgmpy")},
        "references": {case_id: asdict(reference) for case_id, reference in references.items()},
    }
    output = output_dir if output_dir is not None else REPO_ROOT / phase_config["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    with (output / "runs.jsonl").open("x", encoding="utf-8") as runs_file:
        for case_index, case_id in enumerate(case_ids):
            case = config["cases"][case_id]
            for method_index, method in enumerate(methods):
                sampler = samplers[method]
                record_trace = phase == "pilot" and method == "gibbs"
                budgets = [gibbs_settings["trace_sweeps"]] if record_trace else phase_config["budgets"]
                settings = gibbs_settings if method == "gibbs" else {}
                for budget in budgets:
                    for repeat_index in range(phase_config["repeats"]):
                        seed_inputs = [
                            config["rng"]["root_seed"], phase_config["seed_namespace"],
                            case_index, method_index, budget, repeat_index,
                        ]
                        rng = Generator(PCG64(SeedSequence(seed_inputs)))
                        started = perf_counter()
                        result = sampler(
                            model, query=case["query"], evidence=case["evidence"],
                            budget=budget, rng=rng, settings=settings, record_trace=record_trace,
                        )
                        runtime = perf_counter() - started
                        posterior = result.posterior
                        tvd = None
                        if result.status == "ok":
                            exact = references[case_id].posterior
                            if posterior is None or posterior.keys() != exact.keys():
                                raise ValueError(f"Invalid posterior labels from {method}")
                            if not all(0 <= p <= 1 for p in posterior.values()) or not isclose(
                                fsum(posterior.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
                            ):
                                raise ValueError(f"Invalid posterior probabilities from {method}")
                            tvd = 0.5 * fsum(abs(posterior[state] - exact[state]) for state in exact)
                            if record_trace and (
                                result.query_trace is None or len(result.query_trace) != budget
                            ):
                                raise ValueError("Gibbs pilot must return one query state per sweep")
                        elif posterior is not None:
                            raise ValueError(f"Failed {method} run must return an empty posterior")

                        record = {
                            "case_id": case_id,
                            "method": method,
                            "budget": budget,
                            "budget_unit": config["budget_units"][method],
                            "repeat_index": repeat_index,
                            "seed_inputs": seed_inputs,
                            "settings": settings,
                            "runtime_seconds": runtime,
                            "status": result.status,
                            "posterior": posterior,
                            "tvd": tvd,
                            "diagnostics": result.diagnostics,
                        }
                        if record_trace:
                            record["query_trace"] = result.query_trace
                        runs_file.write(json.dumps(record, allow_nan=False) + "\n")
    return output
