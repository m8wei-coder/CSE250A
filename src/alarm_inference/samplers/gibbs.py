"""Gibbs sampling using the caller's Generator and the model's CPT axes.

Evidence is clamped and every unobserved node is updated in lexicographic
topological order. Budgets count retained sweeps after separate burn-in.
Setup, positive-weight LW initialization and per-run conditional caches are
deliberately prepared inside gibbs for timing.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numpy.random import Generator
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.experiment import SamplerResult


MAX_CACHED_CONTEXTS = 4096  # Per variable, per run; no setup is shared between runs.


@dataclass
class _Cpt:
    variables: tuple[str, ...]
    labels: list[str]
    values: np.ndarray
    log_values: np.ndarray
    cdf: np.ndarray


@dataclass
class _Update:
    node: str
    neighbors: tuple[str, ...]
    factors: tuple[_Cpt, ...]
    cache: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def conditional_cdf(self, state: Mapping[str, int]) -> np.ndarray:
        """Normalize the node's CPT times every child's current CPT factor."""
        # Keep this hot method free of generator closures. This also avoids
        # creating a generator and closure cells on every cached update.
        context = []
        for node in self.neighbors:
            context.append(state[node])
        key = tuple(context)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        log_mass = np.zeros(len(self.factors[0].labels))
        for cpt in self.factors:
            # Keep the updated variable's axis and fix all other CPT axes.
            indices = []
            for node in cpt.variables:
                indices.append(slice(None) if node == self.node else state[node])
            log_mass += cpt.log_values[tuple(indices)]
        maximum = float(log_mass.max())
        if not np.isfinite(maximum):
            # A supported current assignment always gives at least one option.
            raise RuntimeError(f"Gibbs reached an unsupported assignment at {self.node!r}")
        mass = np.exp(log_mass - maximum)
        cdf = np.cumsum(mass)
        # Normalize the whole CDF so trailing zero-mass states stay unreachable.
        cdf /= cdf[-1]
        if len(self.cache) < MAX_CACHED_CONTEXTS:
            self.cache[key] = cdf
        return cdf


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prepare(model: DiscreteBayesianNetwork) -> tuple[list[str], dict[str, _Cpt]]:
    """Copy sampling tables without changing the supplied model or CPT order."""
    if not nx.is_directed_acyclic_graph(model):
        raise ValueError("model must be a directed acyclic graph")
    order = list(nx.lexicographical_topological_sort(model))
    cpds = {node: model.get_cpds(node) for node in order}
    tables = {}
    for node in order:
        cpd = cpds[node]
        if cpd is None:
            raise ValueError(f"Missing CPT for {node!r}")
        variables = tuple(cpd.variables)
        if variables[0] != node or set(variables[1:]) != set(model.predecessors(node)):
            raise ValueError(f"CPT parent mismatch for {node!r}")
        for variable, card in zip(variables, cpd.cardinality, strict=True):
            labels = cpd.state_names[variable]
            if len(labels) != card or len(set(labels)) != card:
                raise ValueError(f"Invalid CPT state labels for {variable!r}")
            if cpds[variable] is None or labels != cpds[variable].state_names[variable]:
                raise ValueError(f"Inconsistent parent state order for {node!r}")
        values = np.asarray(cpd.values, dtype=float)
        if (values.shape != tuple(cpd.cardinality) or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=0), 1.0, rtol=0, atol=1e-9)):
            raise ValueError(f"Invalid CPT probabilities for {node!r}")
        with np.errstate(divide="ignore"):
            log_values = np.log(values)  # Exact zeros remain -inf, never epsilon.
        cdf = np.cumsum(values, axis=0)
        cdf /= cdf[-1]
        tables[node] = _Cpt(variables, list(cpd.state_names[node]), values, log_values, cdf)
    return order, tables


def gibbs(
    model: DiscreteBayesianNetwork,
    *,
    query: str,
    evidence: Mapping[str, str],
    budget: int,
    rng: Generator,
    settings: Mapping[str, str | int],
    record_trace: bool = False,
) -> SamplerResult:
    """Keep exactly budget full sweeps after a separately specified burn-in.

    Initialize with the first positive-weight likelihood-weighted assignment.
    Evidence is never updated. The optional trace contains one BIF query label
    per retained sweep (the pilot uses zero burn-in to record from sweep one).
    Setup, initialization and burn-in all happen inside this timed entry point.
    An initialization limit is a sampling failure, not proof of impossible evidence.
    """
    budget = _integer(budget, "budget", 1)
    burn_in = _integer(settings.get("burn_in_sweeps"), "burn_in_sweeps", 0)
    max_attempts = _integer(settings.get("init_max_attempts"), "init_max_attempts", 1)
    initialization = settings.get("initialization", "positive_weight_lw")
    sweep_order = settings.get("sweep_order", "lexicographic_topological")
    if initialization != "positive_weight_lw":
        raise ValueError("initialization must be 'positive_weight_lw'")
    if sweep_order != "lexicographic_topological":
        raise ValueError("sweep_order must be 'lexicographic_topological'")

    order, tables = _prepare(model)
    if query not in tables:
        raise ValueError(f"Unknown query variable {query!r}")
    if query in evidence:
        raise ValueError("Query variable also appears in evidence")
    observed = {}
    for node, label in evidence.items():
        if node not in tables:
            raise ValueError(f"Unknown evidence variable {node!r}")
        if label not in tables[node].labels:
            raise ValueError(f"Unknown state {label!r} for evidence variable {node!r}")
        observed[node] = tables[node].labels.index(label)

    free_nodes = [node for node in order if node not in observed]
    diagnostics = {
        "initialization": initialization,
        "sweep_order": sweep_order,
        "scan_order": free_nodes,
        "burn_in_sweeps": burn_in,
        "init_max_attempts": max_attempts,
        "init_attempts": 0,
        "initial_state": None,
        "initial_log_weight": None,
        "retained_sweeps": 0,
        "completed_sweeps": 0,
    }

    # LW draws unobserved nodes forward while clamping observed nodes. Checking
    # factors individually avoids treating a positive but underflowing weight as 0.
    state: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        state = {}
        log_weight = 0.0
        for node in order:
            cpt = tables[node]
            parents = tuple(state[parent] for parent in cpt.variables[1:])
            if node in observed:
                state[node] = observed[node]
                probability = cpt.values[(state[node], *parents)]
                if probability == 0.0:
                    break
                log_weight += float(cpt.log_values[(state[node], *parents)])
            else:
                cdf = cpt.cdf[(slice(None), *parents)]
                state[node] = int(cdf.searchsorted(rng.random(), side="right"))
        else:
            diagnostics["initial_log_weight"] = log_weight
            break
    else:
        diagnostics.update(init_attempts=max_attempts, failure_reason="initialization_limit")
        return SamplerResult("gibbs_init_failed", None, diagnostics)

    diagnostics["init_attempts"] = attempt
    diagnostics["initial_state"] = {node: tables[node].labels[state[node]] for node in order}
    updates = []
    for node in free_nodes:
        factors = (tables[node], *(tables[child] for child in sorted(model.successors(node))))
        blanket = set().union(*(cpt.variables for cpt in factors)) - {node}
        updates.append(_Update(node, tuple(sorted(blanket)), factors))

    labels = tables[query].labels
    counts = np.zeros(len(labels), dtype=np.int64)
    trace = [] if record_trace else None
    for sweep in range(burn_in + budget):
        # Each draw uses the latest values, including earlier updates in this sweep.
        uniforms = rng.random(len(updates))
        for update, uniform in zip(updates, uniforms, strict=True):
            cdf = update.conditional_cdf(state)
            state[update.node] = int(cdf.searchsorted(uniform, side="right"))
        if sweep >= burn_in:
            query_index = state[query]
            counts[query_index] += 1
            if trace is not None:
                trace.append(labels[query_index])

    diagnostics.update(
        retained_sweeps=budget,
        completed_sweeps=burn_in + budget,
        variable_updates=(burn_in + budget) * len(updates),
        final_state={node: tables[node].labels[state[node]] for node in order},
    )
    posterior = {label: float(count / budget) for label, count in zip(labels, counts, strict=True)}
    return SamplerResult("ok", posterior, diagnostics, trace)
