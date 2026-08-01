"""Validated public records for privacy-safe multi-hop diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tools.security import normalize_owner_id

MAX_HOPS = 12
ALLOWED_STATUSES = frozenset({
    "success",
    "error",
    "timeout",
    "global_timeout",
    "skipped_global_timeout",
    "skipped_missing_dependency_evidence",
})


def identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is outside its allowed range.")
    return parsed


def fingerprint(value: Any) -> str:
    rendered = identifier(value, "plan_fingerprint", 64).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError("plan_fingerprint must be a SHA-256 digest.")
    return rendered


@dataclass(frozen=True)
class MultiHopTraceSummary:
    run_id: str
    owner_id: str
    plan_fingerprint: str
    started_at: float
    completed_at: float
    subquestion_count: int
    batch_count: int
    terminal_count: int
    evidence_count: int
    join_count: int
    terminal_evidence_count: int
    abstain: bool
    exhausted: bool
    used_model: bool
    planner_quality: float
    budget_limit: int
    allocated_budget: int
    error_hops: int
    timeout_hops: int
    skipped_hops: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self, "owner_id", normalize_owner_id(self.owner_id)
        )
        object.__setattr__(
            self,
            "plan_fingerprint",
            fingerprint(self.plan_fingerprint),
        )
        object.__setattr__(
            self, "started_at", number(self.started_at, "started_at", 0.0, 10**15)
        )
        object.__setattr__(
            self, "completed_at", number(self.completed_at, "completed_at", 0.0, 10**15)
        )
        if self.completed_at < self.started_at:
            raise ValueError("completed_at may not precede started_at.")
        for field_name, minimum, maximum in (
            ("subquestion_count", 1, MAX_HOPS),
            ("batch_count", 1, MAX_HOPS),
            ("terminal_count", 1, MAX_HOPS),
            ("evidence_count", 0, 200),
            ("join_count", 0, 200),
            ("terminal_evidence_count", 0, 200),
            ("budget_limit", 1, 100_000),
            ("allocated_budget", 1, 100_000),
            ("error_hops", 0, MAX_HOPS),
            ("timeout_hops", 0, MAX_HOPS),
            ("skipped_hops", 0, MAX_HOPS),
        ):
            object.__setattr__(
                self,
                field_name,
                integer(getattr(self, field_name), field_name, minimum, maximum),
            )
        for field_name in ("abstain", "exhausted", "used_model"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")
        object.__setattr__(
            self,
            "planner_quality",
            number(self.planner_quality, "planner_quality", 0.0, 1.0),
        )


@dataclass(frozen=True)
class MultiHopTraceHop:
    sequence: int
    hop_id: str
    dependency_count: int
    status: str
    returned_evidence: int
    accepted_evidence: int
    error_type: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", integer(self.sequence, "sequence", 0, MAX_HOPS - 1)
        )
        object.__setattr__(self, "hop_id", identifier(self.hop_id, "hop_id", 64))
        object.__setattr__(
            self,
            "dependency_count",
            integer(self.dependency_count, "dependency_count", 0, MAX_HOPS),
        )
        status = identifier(self.status, "status", 100)
        if status not in ALLOWED_STATUSES:
            raise ValueError("status is invalid.")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "returned_evidence",
            integer(self.returned_evidence, "returned_evidence", 0, 50),
        )
        object.__setattr__(
            self,
            "accepted_evidence",
            integer(self.accepted_evidence, "accepted_evidence", 0, 50),
        )
        if self.error_type is not None:
            object.__setattr__(
                self,
                "error_type",
                identifier(self.error_type, "error_type", 200),
            )


@dataclass(frozen=True)
class MultiHopTraceRecord:
    summary: MultiHopTraceSummary
    hops: tuple[MultiHopTraceHop, ...]


@dataclass(frozen=True)
class MultiHopTraceAggregate:
    run_count: int
    abstention_count: int
    exhausted_count: int
    model_plan_count: int
    error_run_count: int
    timeout_run_count: int
    mean_planner_quality: float
    mean_allocated_budget: float
    hop_statuses: tuple[tuple[str, int], ...]


__all__ = [
    "ALLOWED_STATUSES",
    "MAX_HOPS",
    "MultiHopTraceAggregate",
    "MultiHopTraceHop",
    "MultiHopTraceRecord",
    "MultiHopTraceSummary",
    "fingerprint",
    "identifier",
    "integer",
    "number",
]
