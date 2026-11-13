"""Likelihood weighting using the caller's Generator and the model's CPT axes.

Node-wise vectorization is performed in fixed chunks. Changing chunk_size changes
the order in which the supplied random stream is consumed; record this setting
when reproducing runs. Setup is deliberately performed inside lw for timing.
"""

from collections.abc import Mapping
from math import exp, log

import networkx as nx
import numpy as np
from numpy.random import Generator
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.experiment import SamplerResult


DEFAULT_CHUNK_SIZE = 8192


def lw(
    model: DiscreteBayesianNetwork,
    *,
    query: str,
    evidence: Mapping[str, str],
    budget: int,
    rng: Generator,
    settings: Mapping[str, str | int],
    record_trace: bool = False,
) -> SamplerResult:
    """Generate exactly budget weighted assignments, including zero-weight draws.

    Return a label-keyed posterior, ESS, and weight diagnostics. All-zero weights
    are an expected sampling failure; invalid inputs raise ValueError. LW has no
    burn-in and does not return a Gibbs query_trace, even if record_trace is true.
    True zero probabilities are never smoothed or replaced by epsilon.
    """
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    chunk_size = settings.get("chunk_size", DEFAULT_CHUNK_SIZE)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    budget, chunk_size = int(budget), int(chunk_size)
    if not nx.is_directed_acyclic_graph(model):
        raise ValueError("model must be a directed acyclic graph")
    nodes = list(nx.lexicographical_topological_sort(model))
    if query not in nodes:
        raise ValueError(f"Unknown query variable {query!r}")
    if query in evidence:
        raise ValueError("Query variable also appears in evidence")
    for node, state in evidence.items():
        if node not in nodes:
            raise ValueError(f"Unknown evidence variable {node!r}")
        if state not in model.get_cpds(node).state_names[node]:
            raise ValueError(f"Unknown state {state!r} for evidence variable {node!r}")

    prepared = []
    for node in nodes:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"Missing CPT for {node!r}")
        parents = list(cpd.variables[1:])
        if set(parents) != set(model.predecessors(node)):
            raise ValueError(f"CPT parent mismatch for {node!r}")
        for parent in parents:
            if cpd.state_names[parent] != model.get_cpds(parent).state_names[parent]:
                raise ValueError(f"Inconsistent parent state order for {node!r}")
        # cpd.values axes are [child, parent_1, ...], not alphabetical parents.
        values = np.asarray(cpd.values, dtype=float)
        table = np.moveaxis(values, 0, -1).reshape(-1, len(cpd.state_names[node]))
        if (not np.isfinite(table).all() or (table < 0).any()
                or not np.allclose(table.sum(axis=1), 1.0, rtol=0, atol=1e-9)):
            raise ValueError(f"Invalid CPT probabilities for {node!r}")
        cdf = np.cumsum(table, axis=1)
        cdf[:, -1] = 1.0  # close machine-rounding gaps, preserving exact zeros
        with np.errstate(divide="ignore"):
            log_table = np.log(table)
        observed_index = cpd.state_names[node].index(evidence[node]) if node in evidence else None
        prepared.append((node, parents, tuple(map(int, cpd.cardinality[1:])), cdf, log_table, observed_index))

    labels = list(model.get_cpds(query).state_names[query])
    mass = np.zeros(len(labels))
    scale_log = -np.inf
    sum_weight = sum_weight_squared = 0.0
    positive_count = 0
    for start in range(0, budget, chunk_size):
        size = min(chunk_size, budget - start)
        samples = {}
        log_weight = np.zeros(size)
        for node, parents, cards, cdf, log_table, observed_index in prepared:
            parent_index = np.zeros(size, dtype=np.intp)
            for parent, card in zip(parents, cards):
                parent_index = parent_index * card + samples[parent]
            if observed_index is not None:
                samples[node] = np.full(size, observed_index, dtype=np.intp)
                log_weight += log_table[parent_index, observed_index]
            else:
                u = rng.random(size)
                samples[node] = (u[:, None] >= cdf[parent_index]).sum(axis=1)

        finite = np.isfinite(log_weight)
        positive_count += int(finite.sum())
        if not finite.any():
            continue
        batch_max = float(log_weight[finite].max())
        if batch_max > scale_log:
            factor = exp(scale_log - batch_max)
            mass *= factor
            sum_weight *= factor
            sum_weight_squared *= factor * factor
            scale_log = batch_max
        weights = np.exp(log_weight - scale_log)
        mass += np.bincount(samples[query], weights=weights, minlength=len(labels))
        sum_weight += float(weights.sum())
        sum_weight_squared += float(np.dot(weights, weights))

    diagnostics = {
        "num_samples": budget,
        "positive_weight_samples": positive_count,
        "zero_weight_samples": budget - positive_count,
        "chunk_size": chunk_size,
        "scan_order": "lexicographic_topological",
        "ess": 0.0,
        "ess_ratio": 0.0,
        "max_normalized_weight": 0.0,
        "log_total_weight": None,
    }
    if positive_count == 0:
        return SamplerResult("zero_total_weight", None, diagnostics)
    posterior_values = mass / mass.sum()
    ess = min(float(positive_count), sum_weight * sum_weight / sum_weight_squared)
    diagnostics.update(
        ess=ess,
        ess_ratio=ess / budget,
        max_normalized_weight=1.0 / sum_weight,
        log_total_weight=scale_log + log(sum_weight),
    )
    posterior = dict(zip(labels, map(float, posterior_values), strict=True))
    return SamplerResult("ok", posterior, diagnostics)
