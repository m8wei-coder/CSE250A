"""Load and validate the fixed ALARM Bayesian network."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader


ALARM_SHA256 = "701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2"
NORMALIZATION_TOLERANCE = 1e-6
REPORTING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CptSumDeviation:
    node: str
    parents: tuple[tuple[str, str], ...]
    total: float


@dataclass(frozen=True)
class ZeroEntry:
    node: str
    parents: tuple[tuple[str, str], ...]
    state: str


@dataclass(frozen=True)
class ModelReport:
    checksum: str
    node_count: int
    edge_count: int
    cpt_sum_deviations: tuple[CptSumDeviation, ...]
    zero_entries: tuple[ZeroEntry, ...]


def load_alarm_model(path: str | Path) -> tuple[DiscreteBayesianNetwork, ModelReport]:
    """Return the corrected ALARM model and a report on the original probabilities."""
    source = Path(path)
    contents = source.read_bytes()
    digest = sha256(contents).hexdigest()
    if digest != ALARM_SHA256:
        raise ValueError(f"ALARM BIF checksum mismatch: {digest}")
    model = BIFReader(string=contents.decode("utf-8")).get_model(state_name_type=str)
    report = validate_model(model, checksum=digest, require_alarm_shape=True)
    for deviation in report.cpt_sum_deviations:
        cpd = model.get_cpds(deviation.node)
        parent_indices = tuple(cpd.get_state_no(parent, state) for parent, state in deviation.parents)
        cpd.values[(slice(None), *parent_indices)] /= deviation.total
    return model, report


def validate_model(
    model: DiscreteBayesianNetwork,
    *,
    checksum: str = "",
    require_alarm_shape: bool = False,
) -> ModelReport:
    """Check graph and CPD integrity while retaining the reader's original probabilities."""
    if not nx.is_directed_acyclic_graph(model):
        raise ValueError("model graph is not a DAG")
    if require_alarm_shape and (len(model.nodes) != 37 or len(model.edges) != 46):
        raise ValueError("ALARM model must have 37 nodes and 46 edges")

    cpds_by_node: dict[str, list] = {node: [] for node in model.nodes}
    for cpd in model.get_cpds():
        if cpd.variable not in cpds_by_node:
            raise ValueError(f"CPD for unknown node {cpd.variable}")
        cpds_by_node[cpd.variable].append(cpd)

    deviations: list[CptSumDeviation] = []
    zeros: list[ZeroEntry] = []
    for node in model.nodes:
        matching = cpds_by_node[node]
        if len(matching) != 1:
            raise ValueError(f"node {node} must have exactly one CPD")
        cpd = matching[0]
        graph_parents = set(model.predecessors(node))
        parents = list(cpd.variables[1:])
        if len(parents) != len(graph_parents) or set(parents) != graph_parents:
            raise ValueError(f"CPD parents for {node} do not match graph")
        if cpd.cardinality[0] != len(cpd.state_names[node]):
            raise ValueError(f"CPD child cardinality mismatch for {node}")
        for parent, card in zip(parents, cpd.cardinality[1:]):
            if card != len(cpd.state_names[parent]):
                raise ValueError(f"CPD cardinality mismatch for parent {parent} of {node}")
            parent_cpd = cpds_by_node[parent]
            if len(parent_cpd) != 1 or cpd.state_names[parent] != parent_cpd[0].state_names[parent]:
                raise ValueError(f"CPD state labels for parent {parent} of {node} do not match")
        for variable in cpd.variables:
            if len(set(cpd.state_names[variable])) != len(cpd.state_names[variable]):
                raise ValueError(f"CPD for {node} has duplicate states for {variable}")
        expected_shape = tuple(cpd.cardinality)
        if tuple(np.asarray(cpd.values).shape) != expected_shape:
            raise ValueError(f"CPD dimensions for {node} do not match cardinalities")
        values = np.asarray(cpd.values, dtype=float).reshape(cpd.cardinality)
        if not np.isfinite(values).all():
            raise ValueError(f"CPD for {node} must contain only finite values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"CPD for {node} contains probability outside [0, 1]")

        state_names = cpd.state_names
        assignments = product(*(state_names[parent] for parent in parents)) if parents else [()]
        for assignment in assignments:
            parent_map = dict(zip(parents, assignment))
            total = sum(
                float(cpd.get_value(**{node: state, **parent_map}))
                for state in state_names[node]
            )
            parent_tuple = tuple(parent_map.items())
            if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
                raise ValueError(f"CPT for {node} is not normalized for {parent_map}: {total}")
            if abs(total - 1.0) > REPORTING_TOLERANCE:
                deviations.append(CptSumDeviation(node, parent_tuple, total))
            for state in state_names[node]:
                if cpd.get_value(**{node: state, **parent_map}) == 0.0:
                    zeros.append(ZeroEntry(node, parent_tuple, state))

    return ModelReport(checksum, len(model.nodes), len(model.edges), tuple(deviations), tuple(zeros))
