"""Causal identification, sensitivity and transportability diagnostics.

These routines operate on explicit reviewed causal assumptions. They assess graphical or
algebraic conditions; they do not prove a DAG is correct, discover confounders, or turn
observational associations into causal facts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.causal_evidence import CausalAssumption, CausalDAG

_CAUSAL_RELATIONS = frozenset({"causes", "prevents", "mediates", "confounds", "modifies"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _text(value: Any, label: str, maximum: int = 2000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _directed_edges(dag: CausalDAG) -> tuple[tuple[str, str], ...]:
    return tuple(
        (edge.source_variable_id, edge.target_variable_id)
        for edge in dag.edges
        if edge.relation in _CAUSAL_RELATIONS
    )


def _parents(edges: Sequence[tuple[str, str]]) -> Mapping[str, set[str]]:
    output: dict[str, set[str]] = {}
    for source, target in edges:
        output.setdefault(target, set()).add(source)
        output.setdefault(source, set())
    return output


def _children(edges: Sequence[tuple[str, str]]) -> Mapping[str, set[str]]:
    output: dict[str, set[str]] = {}
    for source, target in edges:
        output.setdefault(source, set()).add(target)
        output.setdefault(target, set())
    return output


def _ancestors(nodes: set[str], edges: Sequence[tuple[str, str]]) -> set[str]:
    parents = _parents(edges)
    output = set(nodes)
    pending = list(nodes)
    while pending:
        current = pending.pop()
        for parent in parents.get(current, ()):
            if parent not in output:
                output.add(parent)
                pending.append(parent)
    return output


def _descendants(node: str, edges: Sequence[tuple[str, str]]) -> set[str]:
    children = _children(edges)
    output: set[str] = set()
    pending = list(children.get(node, ()))
    while pending:
        current = pending.pop()
        if current in output:
            continue
        output.add(current)
        pending.extend(children.get(current, ()))
    return output


def _d_separated(
    exposure: str,
    outcome: str,
    conditioned: set[str],
    edges: Sequence[tuple[str, str]],
) -> bool:
    """Exact DAG d-separation via ancestral moralization for disjoint node sets."""

    relevant = _ancestors({exposure, outcome, *conditioned}, edges)
    parent_map = _parents(edges)
    adjacency: dict[str, set[str]] = {node: set() for node in relevant}
    for source, target in edges:
        if source in relevant and target in relevant:
            adjacency[source].add(target)
            adjacency[target].add(source)
    # Moralize co-parents of every relevant child.
    for child in relevant:
        parents = sorted(parent_map.get(child, set()) & relevant)
        for index, first in enumerate(parents):
            for second in parents[index + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)
    for node in conditioned:
        for neighbor in list(adjacency.get(node, ())):
            adjacency[neighbor].discard(node)
        adjacency.pop(node, None)
    if exposure not in adjacency or outcome not in adjacency:
        return True
    seen: set[str] = set()
    pending = [exposure]
    while pending:
        current = pending.pop()
        if current == outcome:
            return False
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, set()) - seen)
    return True


@dataclass(frozen=True)
class AdjustmentSetAssessment:
    exposure_id: str
    outcome_id: str
    adjustment_ids: tuple[str, ...]
    valid_backdoor_set: bool
    descendants_of_exposure_in_set: tuple[str, ...]
    unknown_variables: tuple[str, ...]
    backdoor_paths_blocked: bool
    fingerprint: str
    note: str = (
        "A valid set is conditional on the supplied DAG being substantively correct and "
        "does not guarantee measurement quality, positivity, or model specification."
    )


def assess_backdoor_adjustment_set(
    dag: CausalDAG,
    *,
    exposure_id: str,
    outcome_id: str,
    adjustment_ids: Sequence[str],
) -> AdjustmentSetAssessment:
    if not isinstance(dag, CausalDAG):
        raise TypeError("dag must be CausalDAG")
    exposure = _text(exposure_id, "exposure_id", 256)
    outcome = _text(outcome_id, "outcome_id", 256)
    if exposure == outcome:
        raise ValueError("exposure and outcome must differ")
    adjustment = tuple(dict.fromkeys(_text(item, "adjustment_id", 256) for item in adjustment_ids))
    variables = {item.variable_id for item in dag.variables}
    unknown = tuple(sorted(({exposure, outcome, *adjustment} - variables)))
    edges = _directed_edges(dag)
    descendants = _descendants(exposure, edges) if exposure in variables else set()
    descendant_adjustment = tuple(sorted(descendants.intersection(adjustment)))
    if unknown:
        blocked = False
    else:
        # Backdoor graph removes all arrows emanating from exposure.
        backdoor_edges = tuple((a, b) for a, b in edges if a != exposure)
        blocked = _d_separated(exposure, outcome, set(adjustment), backdoor_edges)
    valid = not unknown and not descendant_adjustment and blocked
    payload = {
        "dag": dag.fingerprint,
        "exposure": exposure,
        "outcome": outcome,
        "adjustment": adjustment,
        "unknown": unknown,
        "descendant_adjustment": descendant_adjustment,
        "blocked": blocked,
    }
    return AdjustmentSetAssessment(
        exposure_id=exposure,
        outcome_id=outcome,
        adjustment_ids=adjustment,
        valid_backdoor_set=valid,
        descendants_of_exposure_in_set=descendant_adjustment,
        unknown_variables=unknown,
        backdoor_paths_blocked=blocked,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


@dataclass(frozen=True)
class NegativeControl:
    control_id: str
    kind: str
    variable_id: str
    rationale: str
    evidence_ids: tuple[str, ...]
    reviewed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_id", _text(self.control_id, "control_id", 256))
        kind = _text(self.kind, "kind", 32).lower()
        if kind not in {"exposure", "outcome"}:
            raise ValueError("negative control kind must be exposure or outcome")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "variable_id", _text(self.variable_id, "variable_id", 256))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale", 5000))
        if not self.evidence_ids or len(self.evidence_ids) > 1000:
            raise ValueError("negative control requires bounded evidence_ids")
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(_text(item, "evidence_id", 500) for item in self.evidence_ids)))
        if not isinstance(self.reviewed, bool):
            raise ValueError("reviewed must be boolean")


@dataclass(frozen=True)
class NegativeControlReadiness:
    reviewed_controls: tuple[str, ...]
    unreviewed_controls: tuple[str, ...]
    unknown_variables: tuple[str, ...]
    ready: bool


def assess_negative_controls(
    dag: CausalDAG,
    controls: Sequence[NegativeControl],
) -> NegativeControlReadiness:
    variables = {item.variable_id for item in dag.variables}
    reviewed = tuple(sorted(item.control_id for item in controls if item.reviewed))
    unreviewed = tuple(sorted(item.control_id for item in controls if not item.reviewed))
    unknown = tuple(sorted({item.variable_id for item in controls if item.variable_id not in variables}))
    return NegativeControlReadiness(
        reviewed_controls=reviewed,
        unreviewed_controls=unreviewed,
        unknown_variables=unknown,
        ready=bool(controls) and not unreviewed and not unknown,
    )


@dataclass(frozen=True)
class RatioConfoundingSensitivity:
    observed_ratio: float
    confounder_outcome_ratio: float
    exposure_confounder_ratio: float
    bias_factor: float
    adjusted_toward_null: float
    crossed_null: bool
    note: str = (
        "This is a bounding sensitivity calculation under the specified ratio-scale "
        "unmeasured-confounding parameters, not an estimate of the true causal effect."
    )


def ratio_confounding_sensitivity(
    *,
    observed_ratio: float,
    confounder_outcome_ratio: float,
    exposure_confounder_ratio: float,
) -> RatioConfoundingSensitivity:
    observed = _positive(observed_ratio, "observed_ratio")
    rr_uy = _positive(confounder_outcome_ratio, "confounder_outcome_ratio")
    rr_eu = _positive(exposure_confounder_ratio, "exposure_confounder_ratio")
    if rr_uy < 1 or rr_eu < 1:
        raise ValueError("sensitivity strength ratios must be at least 1")
    denominator = rr_uy + rr_eu - 1.0
    if denominator <= 0:
        raise ValueError("invalid sensitivity parameters")
    bias_factor = (rr_uy * rr_eu) / denominator
    if observed >= 1:
        adjusted = observed / bias_factor
        crossed = adjusted <= 1.0
    else:
        adjusted = observed * bias_factor
        crossed = adjusted >= 1.0
    return RatioConfoundingSensitivity(
        observed_ratio=observed,
        confounder_outcome_ratio=rr_uy,
        exposure_confounder_ratio=rr_eu,
        bias_factor=bias_factor,
        adjusted_toward_null=adjusted,
        crossed_null=crossed,
    )


@dataclass(frozen=True)
class TransportabilityAssessment:
    source_population: str
    target_population: str
    effect_modifier_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    unreviewed_assumption_ids: tuple[str, ...]
    missing_assumption_ids: tuple[str, ...]
    ready_for_transport_language: bool
    fingerprint: str
    note: str = (
        "Transportability readiness checks explicit reviewed assumptions only; population "
        "similarity and effect-modifier completeness remain substantive judgments."
    )


def assess_transportability(
    *,
    source_population: str,
    target_population: str,
    effect_modifier_ids: Sequence[str],
    assumption_ids: Sequence[str],
    assumptions: Sequence[CausalAssumption],
) -> TransportabilityAssessment:
    source = _text(source_population, "source_population", 5000)
    target = _text(target_population, "target_population", 5000)
    modifiers = tuple(dict.fromkeys(_text(item, "effect_modifier_id", 256) for item in effect_modifier_ids))
    requested = tuple(dict.fromkeys(_text(item, "assumption_id", 256) for item in assumption_ids))
    by_id = {item.assumption_id: item for item in assumptions}
    if len(by_id) != len(assumptions):
        raise ValueError("causal assumption IDs must be unique")
    missing = tuple(sorted(set(requested) - set(by_id)))
    unreviewed = tuple(sorted(item for item in requested if item in by_id and not by_id[item].reviewed))
    wrong_kind = tuple(sorted(item for item in requested if item in by_id and by_id[item].kind != "transportability"))
    unresolved = tuple(sorted(set(missing) | set(unreviewed) | set(wrong_kind)))
    ready = bool(requested) and not unresolved
    payload = {
        "source_population": source,
        "target_population": target,
        "effect_modifier_ids": modifiers,
        "assumption_ids": requested,
        "unresolved": unresolved,
    }
    return TransportabilityAssessment(
        source_population=source,
        target_population=target,
        effect_modifier_ids=modifiers,
        assumption_ids=requested,
        unreviewed_assumption_ids=tuple(sorted(set(unreviewed) | set(wrong_kind))),
        missing_assumption_ids=missing,
        ready_for_transport_language=ready,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


__all__ = [
    "AdjustmentSetAssessment",
    "NegativeControl",
    "NegativeControlReadiness",
    "RatioConfoundingSensitivity",
    "TransportabilityAssessment",
    "assess_backdoor_adjustment_set",
    "assess_negative_controls",
    "assess_transportability",
    "ratio_confounding_sensitivity",
]
