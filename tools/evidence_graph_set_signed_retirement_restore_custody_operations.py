"""Read-only custody-manifest operational audit and retention planning."""

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
from tools.security import normalize_owner_id

_CLASSIFICATIONS = frozenset({"pre_bound_pending_post", "post_bound_complete"})
_STATES = frozenset({"pre_bound", "post_bound"})
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


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _holds(
    values: Iterable[str] | None,
    *,
    label: str,
) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label}s must be an iterable.")
    rendered: set[str] = set()
    for value in values:
        rendered.add(_digest(value, label))
        if len(rendered) > _MAX_HOLDS:
            raise ValueError("held identifiers exceed the limit.")
    return frozenset(rendered)


@dataclass(frozen=True)
class RestoreCustodyOperationalItem:
    custody_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    state: str
    classification: str
    backup_size_bytes: int
    pre_bound_at: float
    post_bound_at: float | None
    age_seconds: float
    pre_bound_method: str
    post_bound_method: str | None
    post_receipt_present: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "custody_id", _digest(self.custody_id, "custody_id"))
        object.__setattr__(self, "restore_id", _digest(self.restore_id, "restore_id"))
        object.__setattr__(
            self, "snapshot_digest", _digest(self.snapshot_digest, "snapshot_digest")
        )
        object.__setattr__(
            self,
            "target_path_digest",
            _digest(self.target_path_digest, "target_path_digest"),
        )
        state = _identifier(self.state, "state", 20)
        if state not in _STATES:
            raise ValueError("custody state is unsupported.")
        object.__setattr__(self, "state", state)
        classification = _identifier(self.classification, "classification", 80)
        if classification not in _CLASSIFICATIONS:
            raise ValueError("custody operational classification is unsupported.")
        expected = (
            "pre_bound_pending_post" if state == "pre_bound" else "post_bound_complete"
        )
        if classification != expected:
            raise ValueError("custody classification differs from state.")
        object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self,
            "backup_size_bytes",
            _integer(
                self.backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            ),
        )
        pre = _timestamp(self.pre_bound_at, "pre_bound_at")
        object.__setattr__(self, "pre_bound_at", pre)
        post = (
            None
            if self.post_bound_at is None
            else _timestamp(self.post_bound_at, "post_bound_at")
        )
        if state == "pre_bound":
            if post is not None or self.post_bound_method is not None:
                raise ValueError("pre-bound custody may not expose post-bound fields.")
        else:
            if post is None or self.post_bound_method is None:
                raise ValueError("post-bound custody requires post-bound fields.")
            if post < pre:
                raise ValueError("post-bound custody predates pre-bound custody.")
        object.__setattr__(self, "post_bound_at", post)
        age = _timestamp(self.age_seconds, "age_seconds")
        object.__setattr__(self, "age_seconds", age)
        object.__setattr__(
            self,
            "pre_bound_method",
            _identifier(self.pre_bound_method, "pre_bound_method", 50),
        )
        post_method = (
            None
            if self.post_bound_method is None
            else _identifier(self.post_bound_method, "post_bound_method", 50)
        )
        object.__setattr__(self, "post_bound_method", post_method)
        present = _boolean(self.post_receipt_present, "post_receipt_present")
        if present != (state == "post_bound"):
            raise ValueError("post receipt presence differs from custody state.")
        object.__setattr__(self, "post_receipt_present", present)


@dataclass(frozen=True)
class RestoreCustodyOperationalReport:
    owner_id: str
    restore_id: str | None
    snapshot_digest: str | None
    target_path_digest: str | None
    state: str | None
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreCustodyOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        object.__setattr__(self, "owner_id", owner)
        restore = None if self.restore_id is None else _digest(self.restore_id, "restore_id")
        snapshot = (
            None
            if self.snapshot_digest is None
            else _digest(self.snapshot_digest, "snapshot_digest")
        )
        target = (
            None
            if self.target_path_digest is None
            else _digest(self.target_path_digest, "target_path_digest")
        )
        selected_state = None if self.state is None else _identifier(self.state, "state", 20)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("custody state is unsupported.")
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "state", selected_state)
        generated = _timestamp(self.generated_at, "generated_at")
        object.__setattr__(self, "generated_at", generated)
        count = _integer(self.item_count, "item_count", 0, _MAX_LIMIT)
        if count != len(self.items):
            raise ValueError("custody operational item count differs from items.")
        object.__setattr__(self, "item_count", count)
        expected_counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        seen: set[str] = set()
        for item in self.items:
            if not isinstance(item, RestoreCustodyOperationalItem):
                raise ValueError("custody operational item is invalid.")
            if item.custody_id in seen:
                raise ValueError("custody operational items contain duplicate IDs.")
            seen.add(item.custody_id)
            expected_counts[item.classification] += 1
        if dict(self.classification_counts) != expected_counts:
            raise ValueError("custody classification counts differ from items.")
        object.__setattr__(self, "classification_counts", expected_counts)
        for value, label in (
            (self.mutation_performed, "mutation_performed"),
            (self.source_text_returned, "source_text_returned"),
            (self.raw_path_returned, "raw_path_returned"),
        ):
            if value is not False:
                raise ValueError(f"{label} must be false.")
        stable = {
            "scope": "rigorousrag-restore-custody-operational-audit-v1",
            "owner_id": owner,
            "restore_id": restore,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "state": selected_state,
            "generated_at": generated,
            "item_count": count,
            "classification_counts": expected_counts,
            "items": [asdict(item) for item in self.items],
        }
        report = _digest(self.report_digest, "report_digest")
        if report != _canonical_digest(stable):
            raise ValueError("report_digest differs from custody operational report.")
        object.__setattr__(self, "report_digest", report)


@dataclass(frozen=True)
class RestoreCustodyRetentionItem:
    custody_id: str
    restore_id: str
    target_path_digest: str
    state: str
    completed_at: float | None
    age_seconds: float
    held: bool
    protected_as_latest: bool
    retention_candidate: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "custody_id", _digest(self.custody_id, "custody_id"))
        object.__setattr__(self, "restore_id", _digest(self.restore_id, "restore_id"))
        object.__setattr__(
            self,
            "target_path_digest",
            _digest(self.target_path_digest, "target_path_digest"),
        )
        state = _identifier(self.state, "state", 20)
        if state not in _STATES:
            raise ValueError("custody state is unsupported.")
        object.__setattr__(self, "state", state)
        completed = (
            None
            if self.completed_at is None
            else _timestamp(self.completed_at, "completed_at")
        )
        if (state == "post_bound") != (completed is not None):
            raise ValueError("custody completion timestamp differs from state.")
        object.__setattr__(self, "completed_at", completed)
        object.__setattr__(self, "age_seconds", _timestamp(self.age_seconds, "age_seconds"))
        for field in ("held", "protected_as_latest", "retention_candidate"):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        reason = _identifier(self.reason, "reason", 100)
        object.__setattr__(self, "reason", reason)
        if self.retention_candidate and (
            state != "post_bound" or self.held or self.protected_as_latest
        ):
            raise ValueError("invalid custody retention candidate.")


@dataclass(frozen=True)
class RestoreCustodyRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_target: int
    include_post_bound: bool
    candidate_count: int
    items: tuple[RestoreCustodyRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    mutation_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        object.__setattr__(self, "owner_id", owner)
        generated = _timestamp(self.generated_at, "generated_at")
        minimum_age = _timestamp(self.minimum_age_seconds, "minimum_age_seconds")
        latest = _integer(self.retain_latest_per_target, "retain_latest_per_target", 1, 100)
        include = _boolean(self.include_post_bound, "include_post_bound")
        candidate_count = _integer(self.candidate_count, "candidate_count", 0, _MAX_LIMIT)
        if candidate_count != sum(item.retention_candidate for item in self.items):
            raise ValueError("custody retention candidate count differs from items.")
        seen: set[str] = set()
        for item in self.items:
            if not isinstance(item, RestoreCustodyRetentionItem):
                raise ValueError("custody retention item is invalid.")
            if item.custody_id in seen:
                raise ValueError("custody retention items contain duplicate IDs.")
            seen.add(item.custody_id)
        for value, label in (
            (self.deletion_performed, "deletion_performed"),
            (self.mutation_performed, "mutation_performed"),
            (self.source_text_returned, "source_text_returned"),
            (self.raw_path_returned, "raw_path_returned"),
        ):
            if value is not False:
                raise ValueError(f"{label} must be false.")
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "minimum_age_seconds", minimum_age)
        object.__setattr__(self, "retain_latest_per_target", latest)
        object.__setattr__(self, "include_post_bound", include)
        object.__setattr__(self, "candidate_count", candidate_count)
        stable = {
            "scope": "rigorousrag-restore-custody-retention-plan-v1",
            "owner_id": owner,
            "generated_at": generated,
            "minimum_age_seconds": minimum_age,
            "retain_latest_per_target": latest,
            "include_post_bound": include,
            "candidate_count": candidate_count,
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.plan_digest, "plan_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("plan_digest differs from custody retention plan.")
        object.__setattr__(self, "plan_digest", digest)


def audit_restore_custody_operations(
    *,
    owner_id: str,
    store: Any,
    restore_id: str | None = None,
    snapshot_digest: str | None = None,
    target_path_digest: str | None = None,
    state: str | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> RestoreCustodyOperationalReport:
    owner = normalize_owner_id(owner_id)
    restore = None if restore_id is None else _digest(restore_id, "restore_id")
    snapshot = None if snapshot_digest is None else _digest(snapshot_digest, "snapshot_digest")
    target = None if target_path_digest is None else _digest(target_path_digest, "target_path_digest")
    selected_state = None if state is None else _identifier(state, "state", 20)
    if selected_state is not None and selected_state not in _STATES:
        raise ValueError("custody state is unsupported.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(store, "list", None)):
        raise ValueError("custody store lacks the required read boundary.")
    values = tuple(store.list(owner_id=owner, state=selected_state, limit=count))
    if len(values) >= count:
        raise RuntimeError("custody audit reached the bounded result limit.")
    rendered: list[RestoreCustodyOperationalItem] = []
    seen: set[str] = set()
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    for value in values:
        if restore is not None and value.restore_id != restore:
            continue
        if snapshot is not None and value.snapshot_digest != snapshot:
            continue
        if target is not None and value.target_path_digest != target:
            continue
        custody_id = _digest(value.custody_id, "custody_id")
        if custody_id in seen:
            raise RuntimeError("custody store returned duplicate IDs.")
        seen.add(custody_id)
        classification = "pre_bound_pending_post" if value.state == "pre_bound" else "post_bound_complete"
        age_from = value.pre_bound_at if value.state == "pre_bound" else value.post_bound_at
        item = RestoreCustodyOperationalItem(
            custody_id=custody_id,
            restore_id=value.restore_id,
            snapshot_digest=value.snapshot_digest,
            target_path_digest=value.target_path_digest,
            state=value.state,
            classification=classification,
            backup_size_bytes=value.backup_size_bytes,
            pre_bound_at=value.pre_bound_at,
            post_bound_at=value.post_bound_at,
            age_seconds=max(0.0, timestamp - float(age_from)),
            pre_bound_method=value.pre_bound_method,
            post_bound_method=value.post_bound_method,
            post_receipt_present=value.post_receipt_digest is not None,
        )
        rendered.append(item)
        counts[classification] += 1
    items = tuple(sorted(rendered, key=lambda item: item.custody_id))
    stable = {
        "scope": "rigorousrag-restore-custody-operational-audit-v1",
        "owner_id": owner,
        "restore_id": restore,
        "snapshot_digest": snapshot,
        "target_path_digest": target,
        "state": selected_state,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return RestoreCustodyOperationalReport(
        owner_id=owner,
        restore_id=restore,
        snapshot_digest=snapshot,
        target_path_digest=target,
        state=selected_state,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_restore_custody_retention(
    *,
    owner_id: str,
    store: Any,
    now: float | None = None,
    minimum_age_seconds: float = 365 * 24 * 60 * 60,
    retain_latest_per_target: int = 1,
    include_post_bound: bool = False,
    held_custody_ids: Iterable[str] | None = None,
    held_restore_ids: Iterable[str] | None = None,
    limit: int = 10_000,
) -> RestoreCustodyRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _timestamp(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(retain_latest_per_target, "retain_latest_per_target", 1, 100)
    include = _boolean(include_post_bound, "include_post_bound")
    held_custody = _holds(held_custody_ids, label="held_custody_id")
    held_restore = _holds(held_restore_ids, label="held_restore_id")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(store, "list", None)):
        raise ValueError("custody store lacks the required read boundary.")
    values = tuple(store.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("custody retention plan reached the bounded result limit.")
    seen: set[str] = set()
    post_by_target: dict[str, list[Any]] = {}
    for value in values:
        custody_id = _digest(value.custody_id, "custody_id")
        if custody_id in seen:
            raise RuntimeError("custody store returned duplicate IDs.")
        seen.add(custody_id)
        if value.state == "post_bound":
            post_by_target.setdefault(value.target_path_digest, []).append(value)
    protected: set[str] = set()
    for target_values in post_by_target.values():
        ordered = sorted(
            target_values,
            key=lambda value: (-float(value.post_bound_at or 0.0), value.custody_id),
        )
        protected.update(value.custody_id for value in ordered[:latest_count])
    rendered: list[RestoreCustodyRetentionItem] = []
    for value in values:
        is_held = value.custody_id in held_custody or value.restore_id in held_restore
        is_latest = value.custody_id in protected
        completed = value.post_bound_at if value.state == "post_bound" else None
        age_from = value.post_bound_at if completed is not None else value.pre_bound_at
        age = max(0.0, timestamp - float(age_from))
        candidate = bool(
            value.state == "post_bound"
            and include
            and age >= minimum_age
            and not is_held
            and not is_latest
        )
        if value.state == "pre_bound":
            reason = "pre_bound_incomplete_never_candidate"
        elif is_held:
            reason = "legal_hold"
        elif is_latest:
            reason = "latest_post_bound_for_target"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif not include:
            reason = "post_bound_retained_by_default"
        elif candidate:
            reason = "old_post_bound_duplicate_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            RestoreCustodyRetentionItem(
                custody_id=value.custody_id,
                restore_id=value.restore_id,
                target_path_digest=value.target_path_digest,
                state=value.state,
                completed_at=completed,
                age_seconds=age,
                held=is_held,
                protected_as_latest=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.custody_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-restore-custody-retention-plan-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_target": latest_count,
        "include_post_bound": include,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return RestoreCustodyRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_target=latest_count,
        include_post_bound=include,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "RestoreCustodyOperationalItem",
    "RestoreCustodyOperationalReport",
    "RestoreCustodyRetentionItem",
    "RestoreCustodyRetentionPlan",
    "audit_restore_custody_operations",
    "plan_restore_custody_retention",
]
