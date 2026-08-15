"""Causal-evidence representation separate from generic evidence graphs.

The module prevents a common scientific-RAG error: treating correlation, prediction or
mere co-occurrence as causal evidence.  Causal assertions require an explicit estimand,
assumptions, design family and evidence identifiers; DAG paths are structural assumptions,
not citation authority or proof of causality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_RELATIONS = frozenset({"causes", "prevents", "mediates", "confounds", "modifies", "associated_with", "predicts", "unknown"})
_DESIGNS = frozenset({"randomized", "natural_experiment", "instrumental_variable", "regression_discontinuity", "difference_in_differences", "cohort", "case_control", "cross_sectional", "time_series", "mechanistic", "other"})


def _text(value: Any, label: str, maximum: int = 2000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CausalVariable:
    variable_id: str
    label: str
    role: str = "other"
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _text(self.variable_id, "variable_id", 256))
        object.__setattr__(self, "label", _text(self.label, "label", 1000))
        role = _text(self.role, "role", 64).lower()
        if role not in {"exposure", "treatment", "outcome", "confounder", "mediator", "modifier", "instrument", "selection", "other"}:
            raise ValueError("unsupported causal variable role")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "unit", _text(self.unit, "unit", 128, allow_empty=True))


@dataclass(frozen=True)
class CausalAssumption:
    assumption_id: str
    kind: str
    statement: str
    evidence_ids: tuple[str, ...] = ()
    reviewed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumption_id", _text(self.assumption_id, "assumption_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in {"exchangeability", "positivity", "consistency", "no_interference", "exclusion_restriction", "parallel_trends", "continuity", "measurement", "selection", "transportability", "other"}:
            raise ValueError("unsupported causal assumption kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "statement", _text(self.statement, "statement", 5000))
        if len(self.evidence_ids) > 1000:
            raise ValueError("evidence_ids exceed the item limit")
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(_text(item, "evidence_id", 500) for item in self.evidence_ids)))
        if not isinstance(self.reviewed, bool):
            raise ValueError("reviewed must be boolean")


@dataclass(frozen=True)
class CausalEdge:
    source_variable_id: str
    target_variable_id: str
    relation: str
    assumption_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_variable_id", _text(self.source_variable_id, "source_variable_id", 256))
        object.__setattr__(self, "target_variable_id", _text(self.target_variable_id, "target_variable_id", 256))
        if self.source_variable_id == self.target_variable_id:
            raise ValueError("causal edges may not self-reference")
        relation = _text(self.relation, "relation", 64).lower()
        if relation not in _RELATIONS:
            raise ValueError("unsupported causal relation")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "assumption_ids", tuple(dict.fromkeys(_text(item, "assumption_id", 256) for item in self.assumption_ids)))


@dataclass(frozen=True)
class CausalDAG:
    variables: tuple[CausalVariable, ...]
    edges: tuple[CausalEdge, ...]
    assumptions: tuple[CausalAssumption, ...] = ()

    def __post_init__(self) -> None:
        if len(self.variables) > 100_000 or len(self.edges) > 1_000_000 or len(self.assumptions) > 100_000:
            raise ValueError("causal DAG exceeds its size limits")
        variables = {item.variable_id for item in self.variables}
        assumptions = {item.assumption_id for item in self.assumptions}
        if len(variables) != len(self.variables) or len(assumptions) != len(self.assumptions):
            raise ValueError("causal DAG IDs must be unique")
        adjacency: dict[str, set[str]] = {item: set() for item in variables}
        for edge in self.edges:
            if edge.source_variable_id not in variables or edge.target_variable_id not in variables:
                raise ValueError("causal edge references an unknown variable")
            if not set(edge.assumption_ids).issubset(assumptions):
                raise ValueError("causal edge references an unknown assumption")
            if edge.relation in {"causes", "prevents", "mediates", "confounds", "modifies"}:
                adjacency[edge.source_variable_id].add(edge.target_variable_id)
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("causal DAG contains a directed cycle")
            if node in visited:
                return
            visiting.add(node)
            for neighbor in adjacency[node]:
                visit(neighbor)
            visiting.remove(node)
            visited.add(node)
        for variable_id in sorted(variables):
            visit(variable_id)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class CausalClaim:
    claim_id: str
    exposure_id: str
    outcome_id: str
    relation: str
    estimand: str
    population: str
    time_horizon: str
    design_family: str
    evidence_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...] = ()
    association_only: bool = True

    def __post_init__(self) -> None:
        for name in ("claim_id", "exposure_id", "outcome_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        relation = _text(self.relation, "relation", 64).lower()
        if relation not in _RELATIONS:
            raise ValueError("unsupported causal relation")
        object.__setattr__(self, "relation", relation)
        for name, maximum in (("estimand", 2000), ("population", 5000), ("time_horizon", 1000)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        design = _text(self.design_family, "design_family", 64).lower()
        if design not in _DESIGNS:
            raise ValueError("unsupported causal design family")
        object.__setattr__(self, "design_family", design)
        if not self.evidence_ids or len(self.evidence_ids) > 10_000:
            raise ValueError("causal claim must cite bounded evidence")
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(_text(item, "evidence_id", 500) for item in self.evidence_ids)))
        object.__setattr__(self, "assumption_ids", tuple(dict.fromkeys(_text(item, "assumption_id", 256) for item in self.assumption_ids)))
        if not isinstance(self.association_only, bool):
            raise ValueError("association_only must be boolean")
        causal_relation = relation in {"causes", "prevents", "mediates"}
        if causal_relation and self.association_only:
            raise ValueError("a causal relation may not simultaneously be marked association_only")
        if causal_relation and not self.assumption_ids:
            raise ValueError("causal claims require explicit causal assumptions")


def causal_claim_readiness(claim: CausalClaim, dag: CausalDAG) -> Mapping[str, Any]:
    variables = {item.variable_id for item in dag.variables}
    assumptions = {item.assumption_id: item for item in dag.assumptions}
    missing_variables = tuple(sorted({claim.exposure_id, claim.outcome_id} - variables))
    missing_assumptions = tuple(sorted(set(claim.assumption_ids) - set(assumptions)))
    unreviewed = tuple(sorted(item for item in claim.assumption_ids if item in assumptions and not assumptions[item].reviewed))
    causal_relation = claim.relation in {"causes", "prevents", "mediates"}
    return {
        "claim_id": claim.claim_id,
        "causal_relation": causal_relation,
        "association_only": claim.association_only,
        "missing_variables": missing_variables,
        "missing_assumptions": missing_assumptions,
        "unreviewed_assumptions": unreviewed,
        "ready_for_causal_language": causal_relation and not missing_variables and not missing_assumptions and not unreviewed,
        "note": "Readiness checks explicit structure/review only; it does not prove causal identification or validity.",
    }


__all__ = ["CausalAssumption", "CausalClaim", "CausalDAG", "CausalEdge", "CausalVariable", "causal_claim_readiness"]
