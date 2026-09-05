"""Deterministic, privacy-safe active-learning acquisition for expert labeling.

The selector is task/model agnostic. Callers normalize their uncertainty, disagreement,
abstention, drift, novelty and expected-impact signals into [0, 1], bind candidates to
content/evidence digests, and provide an estimated labeling cost. Selection is owner-
scoped, deterministic, diversity-capped and budget-constrained.

No raw query, answer, document, table or chart text is required by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be in [0, 1]") from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _nonnegative(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected) or selected < 0.0 or (positive and selected <= 0.0):
        raise ValueError(f"{label} is invalid")
    return selected


@dataclass(frozen=True)
class AcquisitionSignals:
    uncertainty: float = 0.0
    disagreement: float = 0.0
    drift: float = 0.0
    novelty: float = 0.0
    expected_impact: float = 0.0
    abstained: bool = False

    def __post_init__(self) -> None:
        for name in (
            "uncertainty",
            "disagreement",
            "drift",
            "novelty",
            "expected_impact",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")


@dataclass(frozen=True)
class ActiveLearningCandidate:
    owner_id: str
    task_id: str
    item_sha256: str
    evidence_sha256s: tuple[str, ...]
    group_id: str
    signals: AcquisitionSignals
    estimated_label_cost: float = 1.0
    source_policy_sha256: str | None = None
    source_model_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "item_sha256", _sha(self.item_sha256, "item_sha256"))
        evidence = tuple(sorted(_sha(value, "evidence sha256") for value in self.evidence_sha256s))
        if not evidence or len(evidence) > 100_000 or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_sha256s must be unique, non-empty, and bounded")
        object.__setattr__(self, "evidence_sha256s", evidence)
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id", 1000))
        if not isinstance(self.signals, AcquisitionSignals):
            raise ValueError("signals must be AcquisitionSignals")
        object.__setattr__(
            self,
            "estimated_label_cost",
            _nonnegative(self.estimated_label_cost, "estimated_label_cost", positive=True),
        )
        for name in ("source_policy_sha256", "source_model_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))

    @property
    def candidate_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-active-learning-candidate/v1",
                "owner_id": self.owner_id,
                "task_id": self.task_id,
                "item_sha256": self.item_sha256,
                "evidence_sha256s": self.evidence_sha256s,
                "group_id": self.group_id,
                "signals": asdict(self.signals),
                "estimated_label_cost": self.estimated_label_cost,
                "source_policy_sha256": self.source_policy_sha256,
                "source_model_sha256": self.source_model_sha256,
            }
        )

    @property
    def item_key(self) -> tuple[str, str]:
        return self.task_id, self.item_sha256


@dataclass(frozen=True)
class ActiveLearningPolicy:
    max_items: int = 100
    max_total_cost: float = 100.0
    max_per_group: int = 10
    max_per_task: int = 100
    min_acquisition_score: float = 0.0
    uncertainty_weight: float = 1.0
    disagreement_weight: float = 1.0
    drift_weight: float = 0.5
    novelty_weight: float = 0.5
    impact_weight: float = 0.5
    abstention_bonus: float = 0.5
    cost_exponent: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_items", "max_per_group", "max_per_task"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self, "max_total_cost", _nonnegative(self.max_total_cost, "max_total_cost", positive=True)
        )
        object.__setattr__(
            self,
            "min_acquisition_score",
            _nonnegative(self.min_acquisition_score, "min_acquisition_score"),
        )
        for name in (
            "uncertainty_weight",
            "disagreement_weight",
            "drift_weight",
            "novelty_weight",
            "impact_weight",
            "abstention_bonus",
            "cost_exponent",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not any(
            getattr(self, name) > 0.0
            for name in (
                "uncertainty_weight",
                "disagreement_weight",
                "drift_weight",
                "novelty_weight",
                "impact_weight",
                "abstention_bonus",
            )
        ):
            raise ValueError("active-learning policy must assign positive weight to at least one signal")

    @property
    def policy_sha256(self) -> str:
        return _digest(
            {"schema": "rigorousrag-active-learning-policy/v1", **asdict(self)}
        )

    def acquisition_score(self, candidate: ActiveLearningCandidate) -> float:
        signals = candidate.signals
        numerator = (
            self.uncertainty_weight * signals.uncertainty
            + self.disagreement_weight * signals.disagreement
            + self.drift_weight * signals.drift
            + self.novelty_weight * signals.novelty
            + self.impact_weight * signals.expected_impact
            + self.abstention_bonus * float(signals.abstained)
        )
        denominator = candidate.estimated_label_cost ** self.cost_exponent
        return numerator / denominator


@dataclass(frozen=True)
class SelectedAcquisition:
    rank: int
    candidate_sha256: str
    task_id: str
    item_sha256: str
    group_id: str
    acquisition_score: float
    estimated_label_cost: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be positive")
        object.__setattr__(self, "candidate_sha256", _sha(self.candidate_sha256, "candidate_sha256"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "item_sha256", _sha(self.item_sha256, "item_sha256"))
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id", 1000))
        score = _nonnegative(self.acquisition_score, "acquisition_score")
        object.__setattr__(self, "acquisition_score", score)
        object.__setattr__(
            self,
            "estimated_label_cost",
            _nonnegative(self.estimated_label_cost, "estimated_label_cost", positive=True),
        )


@dataclass(frozen=True)
class ActiveLearningBatch:
    owner_id: str
    policy_sha256: str
    candidate_pool_sha256: str
    blocked_items_sha256: str
    selected: tuple[SelectedAcquisition, ...]
    total_estimated_cost: float
    batch_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        for name in ("policy_sha256", "candidate_pool_sha256", "blocked_items_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        selected = tuple(self.selected)
        if any(not isinstance(item, SelectedAcquisition) for item in selected):
            raise ValueError("selected must contain SelectedAcquisition values")
        if [item.rank for item in selected] != list(range(1, len(selected) + 1)):
            raise ValueError("selected ranks must be contiguous")
        if len({(item.task_id, item.item_sha256) for item in selected}) != len(selected):
            raise ValueError("selected batch contains duplicate task/item identities")
        object.__setattr__(self, "selected", selected)
        cost = _nonnegative(self.total_estimated_cost, "total_estimated_cost")
        if abs(cost - sum(item.estimated_label_cost for item in selected)) > 1e-9:
            raise ValueError("total_estimated_cost does not match selected items")
        object.__setattr__(self, "total_estimated_cost", cost)
        expected = _digest(self._payload())
        provided = _sha(self.batch_sha256, "batch_sha256")
        if expected != provided:
            raise ValueError("batch_sha256 does not match active-learning batch content")
        object.__setattr__(self, "batch_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-active-learning-batch/v1",
            "owner_id": self.owner_id,
            "policy_sha256": self.policy_sha256,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "blocked_items_sha256": self.blocked_items_sha256,
            "selected": [asdict(item) for item in self.selected],
            "total_estimated_cost": self.total_estimated_cost,
        }


def _pool_digest(candidates: Sequence[ActiveLearningCandidate]) -> str:
    return _digest(
        {
            "schema": "rigorousrag-active-learning-pool/v1",
            "candidates": sorted(candidate.candidate_sha256 for candidate in candidates),
        }
    )


def _blocked_digest(blocked: Sequence[tuple[str, str]]) -> str:
    return _digest(
        {
            "schema": "rigorousrag-active-learning-blocked-items/v1",
            "items": tuple(sorted(blocked)),
        }
    )


def select_active_learning_batch(
    candidates: Iterable[ActiveLearningCandidate],
    *,
    policy: ActiveLearningPolicy = ActiveLearningPolicy(),
    blocked_item_keys: Iterable[tuple[str, str]] = (),
) -> ActiveLearningBatch:
    values = tuple(candidates)
    if not values or len(values) > 10_000_000:
        raise ValueError("active-learning candidate pool must be non-empty and bounded")
    if any(not isinstance(candidate, ActiveLearningCandidate) for candidate in values):
        raise ValueError("candidate pool contains invalid values")
    owners = {candidate.owner_id for candidate in values}
    if len(owners) != 1:
        raise ValueError("one active-learning batch may contain exactly one owner")
    if len({candidate.item_key for candidate in values}) != len(values):
        raise ValueError("candidate pool contains duplicate task/item identities")

    blocked: set[tuple[str, str]] = set()
    for row in blocked_item_keys:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ValueError("blocked item keys must be task/item pairs")
        blocked.add((_text(row[0], "blocked task id"), _sha(row[1], "blocked item sha256")))

    scored = []
    for candidate in values:
        if candidate.item_key in blocked:
            continue
        score = policy.acquisition_score(candidate)
        if score + 1e-15 < policy.min_acquisition_score:
            continue
        scored.append((score, candidate))
    scored.sort(
        key=lambda row: (
            -row[0],
            -row[1].signals.expected_impact,
            -row[1].signals.uncertainty,
            row[1].estimated_label_cost,
            row[1].task_id,
            row[1].group_id,
            row[1].item_sha256,
            row[1].candidate_sha256,
        )
    )

    group_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    selected_rows: list[SelectedAcquisition] = []
    total_cost = 0.0
    for score, candidate in scored:
        if len(selected_rows) >= policy.max_items:
            break
        if group_counts.get(candidate.group_id, 0) >= policy.max_per_group:
            continue
        if task_counts.get(candidate.task_id, 0) >= policy.max_per_task:
            continue
        if total_cost + candidate.estimated_label_cost > policy.max_total_cost + 1e-12:
            continue
        selected_rows.append(
            SelectedAcquisition(
                rank=len(selected_rows) + 1,
                candidate_sha256=candidate.candidate_sha256,
                task_id=candidate.task_id,
                item_sha256=candidate.item_sha256,
                group_id=candidate.group_id,
                acquisition_score=score,
                estimated_label_cost=candidate.estimated_label_cost,
            )
        )
        total_cost += candidate.estimated_label_cost
        group_counts[candidate.group_id] = group_counts.get(candidate.group_id, 0) + 1
        task_counts[candidate.task_id] = task_counts.get(candidate.task_id, 0) + 1

    payload = {
        "schema": "rigorousrag-active-learning-batch/v1",
        "owner_id": next(iter(owners)),
        "policy_sha256": policy.policy_sha256,
        "candidate_pool_sha256": _pool_digest(values),
        "blocked_items_sha256": _blocked_digest(tuple(blocked)),
        "selected": [asdict(item) for item in selected_rows],
        "total_estimated_cost": total_cost,
    }
    return ActiveLearningBatch(
        owner_id=payload["owner_id"],
        policy_sha256=payload["policy_sha256"],
        candidate_pool_sha256=payload["candidate_pool_sha256"],
        blocked_items_sha256=payload["blocked_items_sha256"],
        selected=tuple(selected_rows),
        total_estimated_cost=total_cost,
        batch_sha256=_digest(payload),
    )


__all__ = [
    "AcquisitionSignals",
    "ActiveLearningBatch",
    "ActiveLearningCandidate",
    "ActiveLearningPolicy",
    "SelectedAcquisition",
    "select_active_learning_batch",
]
