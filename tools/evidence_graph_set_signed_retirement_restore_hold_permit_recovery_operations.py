"""Read-only audit and retention planning for hold-permit recovery evidence."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery import (
    list_hold_permit_recoveries,
)
from tools.security import normalize_owner_id

_CLASSIFICATIONS = frozenset(
    {"quarantine_active", "quarantine_released", "released_hold_cleanup"}
)
_MAX_LIMIT = 10_000
_MAX_HOLDS = 100_000


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _held(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held_recovery_ids must be an iterable.")
    rendered: set[str] = set()
    for value in values:
        rendered.add(_digest(value, "held_recovery_id"))
        if len(rendered) > _MAX_HOLDS:
            raise ValueError("held recovery IDs exceed the limit.")
    return frozenset(rendered)


def _permit_row(connection: Any, hold_id: str) -> Any:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='signed_retirement_restore_hold_placement_permits'"
    ).fetchone()
    if table is None:
        raise RuntimeError("hold placement permit table is missing.")
    return connection.execute(
        "SELECT * FROM signed_retirement_restore_hold_placement_permits "
        "WHERE hold_id=?",
        (hold_id,),
    ).fetchone()


@dataclass(frozen=True)
class HoldPermitRecoveryOperationalItem:
    recovery_id: str
    restore_id: str
    hold_id: str
    recovered_at: float
    actor_binding_method: str
    actor_binding_digest: str
    classification: str
    quarantine_hold_id: str | None
    quarantine_hold_status: str | None
    receipt_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery_id", _digest(self.recovery_id, "recovery_id"))
        object.__setattr__(self, "restore_id", _digest(self.restore_id, "restore_id"))
        object.__setattr__(self, "hold_id", _digest(self.hold_id, "hold_id"))
        object.__setattr__(
            self,
            "recovered_at",
            _timestamp(self.recovered_at, "recovered_at"),
        )
        method = _identifier(
            self.actor_binding_method,
            "actor_binding_method",
            50,
        )
        object.__setattr__(self, "actor_binding_method", method)
        object.__setattr__(
            self,
            "actor_binding_digest",
            _digest(self.actor_binding_digest, "actor_binding_digest"),
        )
        classification = _identifier(
            self.classification,
            "classification",
            100,
        )
        if classification not in _CLASSIFICATIONS:
            raise ValueError("recovery operational classification is unsupported.")
        object.__setattr__(self, "classification", classification)
        quarantine_id = (
            None
            if self.quarantine_hold_id is None
            else _digest(self.quarantine_hold_id, "quarantine_hold_id")
        )
        status = (
            None
            if self.quarantine_hold_status is None
            else _identifier(
                self.quarantine_hold_status,
                "quarantine_hold_status",
                20,
            )
        )
        if classification == "released_hold_cleanup":
            if quarantine_id is not None or status is not None:
                raise ValueError(
                    "released-hold cleanup may not contain quarantine state."
                )
        else:
            expected = (
                "active" if classification == "quarantine_active" else "released"
            )
            if quarantine_id is None or status != expected:
                raise ValueError("quarantine classification differs from hold state.")
        object.__setattr__(self, "quarantine_hold_id", quarantine_id)
        object.__setattr__(self, "quarantine_hold_status", status)
        object.__setattr__(
            self,
            "receipt_digest",
            _digest(self.receipt_digest, "receipt_digest"),
        )


@dataclass(frozen=True)
class HoldPermitRecoveryOperationalReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[HoldPermitRecoveryOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    hold_mutation_performed: bool = False
    permit_mutation_performed: bool = False
    deletion_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        count = _integer(self.item_count, "item_count", 0, _MAX_LIMIT)
        if count != len(self.items):
            raise ValueError("operational report item count differs.")
        if len({item.recovery_id for item in self.items}) != len(self.items):
            raise ValueError("operational report contains duplicate recovery IDs.")
        expected_counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        for item in self.items:
            expected_counts[item.classification] += 1
        if self.classification_counts != expected_counts:
            raise ValueError("operational report classification counts differ.")
        if any(
            value is not False
            for value in (
                self.mutation_performed,
                self.hold_mutation_performed,
                self.permit_mutation_performed,
                self.deletion_performed,
                self.source_text_returned,
                self.raw_paths_returned,
            )
        ):
            raise ValueError("operational report safety flags must be false.")
        stable = {
            "scope": "rigorousrag-hold-permit-recovery-operational-audit-v1",
            "owner_id": owner,
            "generated_at": generated,
            "item_count": count,
            "classification_counts": expected_counts,
            "items": [asdict(item) for item in self.items],
        }
        report_digest = _digest(self.report_digest, "report_digest")
        if report_digest != _canonical_digest(stable):
            raise ValueError("operational report digest differs.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "report_digest", report_digest)


@dataclass(frozen=True)
class HoldPermitRecoveryRetentionItem:
    recovery_id: str
    restore_id: str
    recovered_at: float
    age_seconds: float
    classification: str
    held: bool
    protected_as_latest_for_restore: bool
    retention_candidate: bool
    reason: str


@dataclass(frozen=True)
class HoldPermitRecoveryRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_restore: int
    candidate_count: int
    items: tuple[HoldPermitRecoveryRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    hold_mutation_performed: bool = False
    permit_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        minimum_age = _timestamp(
            self.minimum_age_seconds,
            "minimum_age_seconds",
        )
        latest = _integer(
            self.retain_latest_per_restore,
            "retain_latest_per_restore",
            1,
            100,
        )
        candidates = _integer(
            self.candidate_count,
            "candidate_count",
            0,
            _MAX_LIMIT,
        )
        if candidates != sum(item.retention_candidate for item in self.items):
            raise ValueError("retention candidate count differs.")
        if len({item.recovery_id for item in self.items}) != len(self.items):
            raise ValueError("retention plan contains duplicate recovery IDs.")
        if any(
            value is not False
            for value in (
                self.deletion_performed,
                self.hold_mutation_performed,
                self.permit_mutation_performed,
                self.source_text_returned,
                self.raw_paths_returned,
            )
        ):
            raise ValueError("retention plan safety flags must be false.")
        stable = {
            "scope": "rigorousrag-hold-permit-recovery-retention-plan-v1",
            "owner_id": owner,
            "generated_at": generated,
            "minimum_age_seconds": minimum_age,
            "retain_latest_per_restore": latest,
            "candidate_count": candidates,
            "items": [asdict(item) for item in self.items],
        }
        plan_digest = _digest(self.plan_digest, "plan_digest")
        if plan_digest != _canonical_digest(stable):
            raise ValueError("retention plan digest differs.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "minimum_age_seconds", minimum_age)
        object.__setattr__(self, "retain_latest_per_restore", latest)
        object.__setattr__(self, "plan_digest", plan_digest)


def audit_hold_permit_recoveries(
    *,
    owner_id: str,
    restore_journal: Any,
    hold_store: Any,
    now: float | None = None,
    limit: int = 1_000,
) -> HoldPermitRecoveryOperationalReport:
    owner = normalize_owner_id(owner_id)
    generated = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    receipts = list_hold_permit_recoveries(
        restore_journal,
        owner_id=owner,
        limit=count,
    )
    if not callable(getattr(hold_store, "get", None)):
        raise ValueError("hold store lacks the get boundary.")
    rendered: list[HoldPermitRecoveryOperationalItem] = []
    with restore_journal._lock, restore_journal._connect() as connection:
        for receipt in receipts:
            permit = _permit_row(connection, receipt.hold_id)
            if permit is None:
                raise RuntimeError("recovered hold permit is missing.")
            if (
                permit["owner_id"] != owner
                or permit["restore_id"] != receipt.restore_id
                or permit["state"] != "released"
                or permit["permit_digest"] != receipt.released_permit_digest
            ):
                raise RuntimeError("recovered hold permit differs from receipt.")
            if receipt.quarantine_hold_id is None:
                classification = "released_hold_cleanup"
                quarantine_status = None
            else:
                quarantine = hold_store.get(receipt.quarantine_hold_id)
                if (
                    quarantine.owner_id != owner
                    or quarantine.restore_id != receipt.restore_id
                    or quarantine.hold_digest != receipt.quarantine_hold_digest
                    or quarantine.status not in {"active", "released"}
                ):
                    raise RuntimeError("quarantine hold differs from recovery receipt.")
                quarantine_status = quarantine.status
                classification = (
                    "quarantine_active"
                    if quarantine.status == "active"
                    else "quarantine_released"
                )
            rendered.append(
                HoldPermitRecoveryOperationalItem(
                    recovery_id=receipt.recovery_id,
                    restore_id=receipt.restore_id,
                    hold_id=receipt.hold_id,
                    recovered_at=receipt.recovered_at,
                    actor_binding_method=receipt.actor_binding_method,
                    actor_binding_digest=receipt.actor_binding_digest,
                    classification=classification,
                    quarantine_hold_id=receipt.quarantine_hold_id,
                    quarantine_hold_status=quarantine_status,
                    receipt_digest=receipt.receipt_digest,
                )
            )
    items = tuple(sorted(rendered, key=lambda item: item.recovery_id))
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    for item in items:
        counts[item.classification] += 1
    stable = {
        "scope": "rigorousrag-hold-permit-recovery-operational-audit-v1",
        "owner_id": owner,
        "generated_at": generated,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return HoldPermitRecoveryOperationalReport(
        owner_id=owner,
        generated_at=generated,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_hold_permit_recovery_retention(
    *,
    owner_id: str,
    restore_journal: Any,
    hold_store: Any,
    now: float | None = None,
    minimum_age_seconds: float = 365 * 24 * 60 * 60,
    retain_latest_per_restore: int = 1,
    held_recovery_ids: Iterable[str] | None = None,
    limit: int = _MAX_LIMIT,
) -> HoldPermitRecoveryRetentionPlan:
    generated = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _timestamp(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(
        retain_latest_per_restore,
        "retain_latest_per_restore",
        1,
        100,
    )
    holds = _held(held_recovery_ids)
    report = audit_hold_permit_recoveries(
        owner_id=owner_id,
        restore_journal=restore_journal,
        hold_store=hold_store,
        now=generated,
        limit=limit,
    )
    by_restore: dict[str, list[HoldPermitRecoveryOperationalItem]] = {}
    for item in report.items:
        by_restore.setdefault(item.restore_id, []).append(item)
    protected: set[str] = set()
    for values in by_restore.values():
        ordered = sorted(
            values,
            key=lambda item: (-item.recovered_at, item.recovery_id),
        )
        protected.update(item.recovery_id for item in ordered[:latest_count])

    rendered: list[HoldPermitRecoveryRetentionItem] = []
    for item in report.items:
        age = max(0.0, generated - item.recovered_at)
        is_held = item.recovery_id in holds
        is_latest = item.recovery_id in protected
        active_quarantine = item.classification == "quarantine_active"
        candidate = bool(
            age >= minimum_age
            and not is_held
            and not is_latest
            and not active_quarantine
        )
        if active_quarantine:
            reason = "active_quarantine_hold"
        elif is_held:
            reason = "explicit_recovery_hold"
        elif is_latest:
            reason = "latest_recovery_for_restore"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif candidate:
            reason = "old_resolved_recovery_evidence_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            HoldPermitRecoveryRetentionItem(
                recovery_id=item.recovery_id,
                restore_id=item.restore_id,
                recovered_at=item.recovered_at,
                age_seconds=age,
                classification=item.classification,
                held=is_held,
                protected_as_latest_for_restore=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.recovery_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-hold-permit-recovery-retention-plan-v1",
        "owner_id": report.owner_id,
        "generated_at": generated,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_restore": latest_count,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return HoldPermitRecoveryRetentionPlan(
        owner_id=report.owner_id,
        generated_at=generated,
        minimum_age_seconds=minimum_age,
        retain_latest_per_restore=latest_count,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "HoldPermitRecoveryOperationalItem",
    "HoldPermitRecoveryOperationalReport",
    "HoldPermitRecoveryRetentionItem",
    "HoldPermitRecoveryRetentionPlan",
    "audit_hold_permit_recoveries",
    "plan_hold_permit_recovery_retention",
]
