"""Rejection sampling using the caller's Generator and the model's CPT axes.

Every node is forward sampled in lexicographic topological order, including the
evidence nodes, so the cost per generated sample matches likelihood weighting.
Node-wise vectorization is performed in fixed chunks. Changing chunk_size changes
the order in which the supplied random stream is consumed; record this setting
when reproducing runs. Setup is deliberately performed inside rs for timing.
"""

from collections.abc import Mapping

import networkx as nx
import numpy as np
from numpy.random import Generator
from pgmpy.models import DiscreteBayesianNetwork

from alarm_inference.experiment import SamplerResult


DEFAULT_CHUNK_SIZE = 8192


def rs(
    model: DiscreteBayesianNetwork,
    *,
    query: str,
    evidence: Mapping[str, str],
    budget: int,
    rng: Generator,
    settings: Mapping[str, str | int],
    record_trace: bool = False,
) -> SamplerResult:
    """Generate exactly budget prior samples and keep those that match the evidence.

    The budget counts generated samples, not accepted ones. Return a label-keyed
    posterior from the accepted samples and acceptance diagnostics. Zero accepted
    samples are an expected sampling failure; invalid inputs raise ValueError.
    RS does not return a Gibbs query_trace, even if record_trace is true.
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
    observed = {}
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
        prepared.append((node, parents, tuple(map(int, cpd.cardinality[1:])), cdf))
        if node in evidence:
            observed[node] = cpd.state_names[node].index(evidence[node])

    labels = list(model.get_cpds(query).state_names[query])
    counts = np.zeros(len(labels), dtype=np.int64)
    for start in range(0, budget, chunk_size):
        size = min(chunk_size, budget - start)
        samples = {}
        for node, parents, cards, cdf in prepared:
            parent_index = np.zeros(size, dtype=np.intp)
            for parent, card in zip(parents, cards):
                parent_index = parent_index * card + samples[parent]
            u = rng.random(size)
            samples[node] = (u[:, None] >= cdf[parent_index]).sum(axis=1)

        accept = np.ones(size, dtype=bool)
        for node, index in observed.items():
            accept &= samples[node] == index
        counts += np.bincount(samples[query][accept], minlength=len(labels))

    accepted = int(counts.sum())
    diagnostics = {
        "num_samples": budget,
        "accepted": accepted,
        "rejected": budget - accepted,
        "acceptance_rate": accepted / budget,
        "chunk_size": chunk_size,
        "scan_order": "lexicographic_topological",
    }
    if accepted == 0:
        return SamplerResult("zero_accepts", None, diagnostics)
    posterior = dict(zip(labels, map(float, counts / accepted), strict=True))
    return SamplerResult("ok", posterior, diagnostics)
