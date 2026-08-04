"""Strict reconstruction boundary for deletion operations and permit reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_operations import (
    RestoreDeletionOperationalItem,
    RestoreDeletionRetentionItem,
    _CLASSIFICATIONS,
    _canonical_digest,
    audit_restore_deletion_operations as _audit_deletions,
    plan_restore_deletion_retention as _plan_deletions,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_audit import (
    RestoreHoldPermitAuditItem,
    _CLASSIFICATIONS as _PERMIT_CLASSIFICATIONS,
    audit_restore_hold_placement_permits as _audit_permits,
)
from tools.security import normalize_owner_id


def _false(value: Any, label: str) -> bool:
    if value is not False:
        raise ValueError(f"{label} must be false.")
    return False


@dataclass(frozen=True)
class RestoreDeletionOperationalReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreDeletionOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    restore_row_deleted: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        items = tuple(self.items)
        count = _integer(self.item_count, "item_count", 0, 10_000)
        if count != len(items):
            raise ValueError("operational item_count differs from items.")
        identifiers = [item.deletion_id for item in items]
        if identifiers != sorted(identifiers) or len(set(identifiers)) != len(items):
            raise ValueError("operational items are not unique and ordered.")
        expected_counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        for item in items:
            if item.classification not in expected_counts:
                raise ValueError("operational classification is unsupported.")
            expected_counts[item.classification] += 1
        if self.classification_counts != expected_counts:
            raise ValueError("operational classification counts differ.")
        for label, value in (
            ("mutation_performed", self.mutation_performed),
            ("restore_row_deleted", self.restore_row_deleted),
            ("source_text_returned", self.source_text_returned),
            ("raw_paths_returned", self.raw_paths_returned),
        ):
            _false(value, label)
        stable = {
            "scope": "rigorousrag-restore-deletion-operational-audit-v1",
            "owner_id": owner,
            "generated_at": generated,
            "item_count": count,
            "classification_counts": expected_counts,
            "items": [asdict(item) for item in items],
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("operational report_digest differs.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "report_digest", digest)


@dataclass(frozen=True)
class RestoreDeletionRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_restore: int
    include_completed: bool
    candidate_count: int
    items: tuple[RestoreDeletionRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    compaction_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        minimum_age = _timestamp(
            self.minimum_age_seconds, "minimum_age_seconds"
        )
        latest = _integer(
            self.retain_latest_per_restore,
            "retain_latest_per_restore",
            1,
            100,
        )
        if not isinstance(self.include_completed, bool):
            raise ValueError("include_completed must be boolean.")
        items = tuple(self.items)
        identifiers = [item.deletion_id for item in items]
        if identifiers != sorted(identifiers) or len(set(identifiers)) != len(items):
            raise ValueError("retention items are not unique and ordered.")
        candidates = sum(item.retention_candidate for item in items)
        if _integer(
            self.candidate_count, "candidate_count", 0, 10_000
        ) != candidates:
            raise ValueError("retention candidate_count differs from items.")
        for label, value in (
            ("deletion_performed", self.deletion_performed),
            ("compaction_performed", self.compaction_performed),
            ("source_text_returned", self.source_text_returned),
            ("raw_paths_returned", self.raw_paths_returned),
        ):
            _false(value, label)
        stable = {
            "scope": "rigorousrag-restore-deletion-retention-plan-v1",
            "owner_id": owner,
            "generated_at": generated,
            "minimum_age_seconds": minimum_age,
            "retain_latest_per_restore": latest,
            "include_completed": self.include_completed,
            "candidate_count": candidates,
            "items": [asdict(item) for item in items],
        }
        digest = _digest(self.plan_digest, "plan_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("retention plan_digest differs.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "minimum_age_seconds", minimum_age)
        object.__setattr__(self, "retain_latest_per_restore", latest)
        object.__setattr__(self, "candidate_count", candidates)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "plan_digest", digest)


@dataclass(frozen=True)
class RestoreHoldPermitAuditReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreHoldPermitAuditItem, ...]
    report_digest: str
    mutation_performed: bool = False
    permit_released: bool = False
    hold_mutation_performed: bool = False
    restore_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        items = tuple(self.items)
        count = _integer(self.item_count, "item_count", 0, 10_000)
        if count != len(items):
            raise ValueError("permit item_count differs from items.")
        identifiers = [item.hold_id for item in items]
        if identifiers != sorted(identifiers) or len(set(identifiers)) != len(items):
            raise ValueError("permit items are not unique and ordered.")
        expected_counts = {
            name: 0 for name in sorted(_PERMIT_CLASSIFICATIONS)
        }
        for item in items:
            if item.classification not in expected_counts:
                raise ValueError("permit classification is unsupported.")
            expected_counts[item.classification] += 1
        if self.classification_counts != expected_counts:
            raise ValueError("permit classification counts differ.")
        for label, value in (
            ("mutation_performed", self.mutation_performed),
            ("permit_released", self.permit_released),
            ("hold_mutation_performed", self.hold_mutation_performed),
            ("restore_mutation_performed", self.restore_mutation_performed),
            ("source_text_returned", self.source_text_returned),
            ("raw_paths_returned", self.raw_paths_returned),
        ):
            _false(value, label)
        stable = {
            "scope": "rigorousrag-restore-hold-permit-audit-v1",
            "owner_id": owner,
            "generated_at": generated,
            "item_count": count,
            "classification_counts": expected_counts,
            "items": [asdict(item) for item in items],
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("permit report_digest differs.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "report_digest", digest)


def audit_restore_deletion_operations(**kwargs: Any) -> RestoreDeletionOperationalReport:
    value = _audit_deletions(**kwargs)
    return RestoreDeletionOperationalReport(**value.__dict__)


def plan_restore_deletion_retention(**kwargs: Any) -> RestoreDeletionRetentionPlan:
    value = _plan_deletions(**kwargs)
    return RestoreDeletionRetentionPlan(**value.__dict__)


def audit_restore_hold_placement_permits(**kwargs: Any) -> RestoreHoldPermitAuditReport:
    value = _audit_permits(**kwargs)
    return RestoreHoldPermitAuditReport(**value.__dict__)


__all__ = [
    "RestoreDeletionOperationalReport",
    "RestoreDeletionRetentionPlan",
    "RestoreHoldPermitAuditReport",
    "audit_restore_deletion_operations",
    "audit_restore_hold_placement_permits",
    "plan_restore_deletion_retention",
]
