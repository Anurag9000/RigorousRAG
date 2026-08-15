"""Candidate-ranked and failure-aware planning control for multi-hop retrieval.

This layer sits above ``query_decomposition``.  It can compare several proposed DAGs,
score them with deterministic structural features or an injected learned ranker, select
a bounded beam, and create repair plans after weak/failed hops.  It does not execute
retrieval itself and therefore cannot manufacture evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from tools.query_decomposition import DecompositionPlan, Subquestion, build_decomposition_plan

_MAX_PLANS = 32
_MAX_HOPS = 12


def _text(value: Any, label: str, maximum: int = 4000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class PlanFeatures:
    hop_count: int
    depth: int
    parallelism: float
    entity_coverage: float
    temporal_specificity: float
    synthesis_fraction: float
    dependency_density: float

    def __post_init__(self) -> None:
        if isinstance(self.hop_count, bool) or not isinstance(self.hop_count, int) or not 1 <= self.hop_count <= _MAX_HOPS:
            raise ValueError("hop_count is invalid")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or not 1 <= self.depth <= _MAX_HOPS:
            raise ValueError("depth is invalid")
        for name in ("parallelism", "entity_coverage", "temporal_specificity", "synthesis_fraction", "dependency_density"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


def plan_features(plan: DecompositionPlan) -> PlanFeatures:
    if not isinstance(plan, DecompositionPlan):
        raise TypeError("plan must be DecompositionPlan")
    nodes = plan.subquestions
    dependency_count = sum(len(node.depends_on) for node in nodes)
    max_edges = max(1, len(nodes) * (len(nodes) - 1) // 2)
    entity_count = sum(1 for node in nodes if node.entities)
    temporal_count = sum(1 for node in nodes if node.temporal_constraints)
    synthesis_count = sum(1 for node in nodes if node.relation in {"compare", "synthesize"})
    parallelism = max((len(batch) for batch in plan.batches), default=1) / len(nodes)
    return PlanFeatures(
        hop_count=len(nodes),
        depth=len(plan.batches),
        parallelism=min(1.0, parallelism),
        entity_coverage=entity_count / len(nodes),
        temporal_specificity=temporal_count / len(nodes),
        synthesis_fraction=synthesis_count / len(nodes),
        dependency_density=min(1.0, dependency_count / max_edges),
    )


class PlanRanker(Protocol):
    @property
    def version(self) -> str: ...
    def score(self, query: str, plan: DecompositionPlan, features: PlanFeatures) -> float: ...


@dataclass(frozen=True)
class StructuralPlanRanker:
    version: str = "structural-v1"

    def score(self, query: str, plan: DecompositionPlan, features: PlanFeatures) -> float:
        del query, plan
        # Prefer bounded depth, useful parallelism, explicit entities/temporal constraints,
        # and avoid gratuitously dense DAGs. This is a heuristic rank, not evidence quality.
        depth_penalty = min(1.0, max(0.0, (features.depth - 4) / 8.0))
        hop_penalty = min(1.0, max(0.0, (features.hop_count - 8) / 4.0))
        value = (
            0.20 * features.parallelism
            + 0.20 * features.entity_coverage
            + 0.15 * features.temporal_specificity
            + 0.15 * features.synthesis_fraction
            + 0.20 * (1.0 - features.dependency_density)
            + 0.10 * (1.0 - depth_penalty)
            - 0.10 * hop_penalty
        )
        return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RankedPlan:
    plan: DecompositionPlan
    score: float
    ranker_version: str
    features: PlanFeatures


@dataclass(frozen=True)
class PlanSelection:
    selected: RankedPlan
    beam: tuple[RankedPlan, ...]
    candidate_count: int
    fingerprint: str


def rank_candidate_plans(
    query: str,
    proposals: Sequence[Sequence[Subquestion | Mapping[str, Any]]],
    *,
    ranker: PlanRanker | None = None,
    beam_width: int = 3,
    max_subquestions: int = 8,
) -> PlanSelection:
    cleaned = _text(query, "query", 20_000)
    if not 1 <= len(proposals) <= _MAX_PLANS:
        raise ValueError("proposals must contain between 1 and 32 plans")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or not 1 <= beam_width <= 8:
        raise ValueError("beam_width is invalid")
    selected_ranker = ranker or StructuralPlanRanker()
    ranked: list[RankedPlan] = []
    seen: set[str] = set()
    for proposal in proposals:
        try:
            plan = build_decomposition_plan(cleaned, proposed_subquestions=proposal, max_subquestions=max_subquestions)
        except ValueError:
            continue
        if plan.fingerprint in seen:
            continue
        seen.add(plan.fingerprint)
        features = plan_features(plan)
        try:
            score = _probability(selected_ranker.score(cleaned, plan, features), "plan score")
        except Exception:
            score = 0.0
        ranked.append(RankedPlan(plan, score, _text(selected_ranker.version, "ranker version", 128), features))
    if not ranked:
        fallback = build_decomposition_plan(cleaned, max_subquestions=max_subquestions)
        features = plan_features(fallback)
        default_ranker = StructuralPlanRanker()
        ranked.append(RankedPlan(fallback, default_ranker.score(cleaned, fallback, features), default_ranker.version, features))
    ranked.sort(key=lambda item: (-item.score, item.plan.fingerprint))
    beam = tuple(ranked[:beam_width])
    payload = {
        "selected": beam[0].plan.fingerprint,
        "beam": [(item.plan.fingerprint, item.score, item.ranker_version) for item in beam],
        "candidate_count": len(ranked),
    }
    return PlanSelection(beam[0], beam, len(ranked), hashlib.sha256(_canonical(payload)).hexdigest())


@dataclass(frozen=True)
class HopOutcome:
    question_id: str
    status: str
    evidence_count: int
    sufficiency: float
    error_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _text(self.question_id, "question_id", 64))
        status = _text(self.status, "status", 32).lower()
        if status not in {"success", "weak", "empty", "failed", "timeout", "skipped"}:
            raise ValueError("unsupported hop status")
        object.__setattr__(self, "status", status)
        if isinstance(self.evidence_count, bool) or not isinstance(self.evidence_count, int) or not 0 <= self.evidence_count <= 100_000:
            raise ValueError("evidence_count is invalid")
        object.__setattr__(self, "sufficiency", _probability(self.sufficiency, "sufficiency"))
        object.__setattr__(self, "error_type", _text(self.error_type, "error_type", 128, ) if self.error_type else "")


@dataclass(frozen=True)
class RepairAction:
    question_id: str
    action: str
    reason: str
    budget_weight: float


def build_repair_actions(plan: DecompositionPlan, outcomes: Sequence[HopOutcome]) -> tuple[RepairAction, ...]:
    by_id = plan.by_id()
    if len(outcomes) > len(by_id):
        raise ValueError("outcomes exceed plan size")
    actions: list[RepairAction] = []
    for outcome in outcomes:
        node = by_id.get(outcome.question_id)
        if node is None:
            raise ValueError("outcome references an unknown question")
        if outcome.status == "success" and outcome.sufficiency >= 0.75:
            continue
        if outcome.status in {"failed", "timeout"}:
            action, weight = "switch_route", 1.0
        elif outcome.status == "empty":
            action, weight = "rewrite_and_expand", 0.9
        elif outcome.status == "skipped":
            action, weight = "repair_dependencies", 0.8
        elif outcome.sufficiency < 0.35:
            action, weight = "alternate_decomposition", 0.8
        else:
            action, weight = "deepen_retrieval", 0.5
        reason = f"{outcome.status}:{outcome.error_type or 'insufficient_evidence'}"
        actions.append(RepairAction(node.question_id, action, reason, weight))
    actions.sort(key=lambda item: (-item.budget_weight, item.question_id))
    return tuple(actions)


def redistribute_budget(total_remaining: float, actions: Sequence[RepairAction]) -> Mapping[str, float]:
    if isinstance(total_remaining, bool):
        raise ValueError("total_remaining is invalid")
    total = float(total_remaining)
    if not math.isfinite(total) or total < 0:
        raise ValueError("total_remaining must be finite and non-negative")
    if not actions:
        return {}
    weight_sum = sum(max(0.0, item.budget_weight) for item in actions)
    if weight_sum <= 0:
        return {item.question_id: 0.0 for item in actions}
    return {item.question_id: total * item.budget_weight / weight_sum for item in actions}


__all__ = [
    "HopOutcome",
    "PlanFeatures",
    "PlanRanker",
    "PlanSelection",
    "RankedPlan",
    "RepairAction",
    "StructuralPlanRanker",
    "build_repair_actions",
    "plan_features",
    "rank_candidate_plans",
    "redistribute_budget",
]
