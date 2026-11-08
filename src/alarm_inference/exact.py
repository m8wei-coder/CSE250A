"""Reference inference by variable elimination on a validated discrete BN."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


@dataclass(frozen=True)
class ExactResult:
    p_evidence: float
    posterior: dict[str, float] | None
    status: Literal["ok", "zero_probability_evidence"]


def exact_inference(
    model: DiscreteBayesianNetwork,
    query: str,
    evidence: Mapping[str, str] | None = None,
) -> ExactResult:
    """Return the evidence probability and label-keyed posterior for one query variable."""
    evidence = evidence if evidence is not None else {}
    states = model.states
    if query not in states:
        raise ValueError(f"Unknown query variable {query!r}")
    if query in evidence:
        raise ValueError(f"Query variable also appears in evidence: {query!r}")
    for variable, state in evidence.items():
        if variable not in states:
            raise ValueError(f"Unknown evidence variable {variable!r}")
        if state not in states[variable]:
            raise ValueError(f"Unknown state {state!r} for evidence variable {variable!r}")

    ve = VariableElimination(model)
    p_evidence = 1.0
    observed: dict[str, str] = {}
    # Apply the chain rule before querying a posterior on the full evidence.
    for variable, state in sorted(evidence.items()):
        factor = ve.query([variable], evidence=observed, show_progress=False)
        probability = float(factor.get_value(**{variable: state}))
        if probability == 0.0:
            return ExactResult(p_evidence=0.0, posterior=None, status="zero_probability_evidence")
        p_evidence *= probability
        observed[variable] = state

    factor = ve.query([query], evidence=observed, show_progress=False)
    posterior = dict(zip(factor.state_names[query], map(float, factor.values), strict=True))
    return ExactResult(p_evidence=p_evidence, posterior=posterior, status="ok")
