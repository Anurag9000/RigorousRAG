"""Read-only rotation assessment for custody timestamp-authority keys."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_CLASSIFICATIONS = frozenset(
    {
        "initial_key_required",
        "healthy_single_active",
        "rotation_required_no_successor",
        "overlap_window_active",
        "retire_oldest_after_overlap",
        "too_many_active_keys",
    }
)
_ACTIONS = frozenset(
    {
        "register_initial_timestamp_authority_key",
        "register_successor_timestamp_authority_key",
        "retain_oldest_key_until_overlap_completes",
        "retire_oldest_timestamp_authority_key",
        "review_and_retire_excess_active_keys",
    }
)
_STATES = frozenset({"active", "retired"})


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


def _duration(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or not 0 <= selected <= maximum:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _optional_identifier(value: str | None, label: str) -> str | None:
    return None if value is None else _identifier(value, label, 200)


def _optional_timestamp(value: float | None, label: str) -> float | None:
    return None if value is None else _timestamp(value, label)


@dataclass(frozen=True)
class CustodyTimestampRotationPolicy:
    maximum_active_key_age_seconds: float
    minimum_overlap_seconds: float
    maximum_active_keys: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_active_key_age_seconds",
            _duration(
                self.maximum_active_key_age_seconds,
                "maximum_active_key_age_seconds",
                20 * 365 * 24 * 60 * 60,
            ),
        )
        object.__setattr__(
            self,
            "minimum_overlap_seconds",
            _duration(
                self.minimum_overlap_seconds,
                "minimum_overlap_seconds",
                365 * 24 * 60 * 60,
            ),
        )
        object.__setattr__(
            self,
            "maximum_active_keys",
            _integer(self.maximum_active_keys, "maximum_active_keys", 1, 100),
        )
        if self.schema_version != 1:
            raise ValueError("timestamp rotation policy schema is unsupported.")

    @classmethod
    def default(cls) -> "CustodyTimestampRotationPolicy":
        return cls(
            maximum_active_key_age_seconds=365 * 24 * 60 * 60,
            minimum_overlap_seconds=7 * 24 * 60 * 60,
            maximum_active_keys=2,
        )

    @property
    def policy_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-custody-timestamp-rotation-policy-v1",
                **asdict(self),
            }
        )


@dataclass(frozen=True)
class CustodyTimestampRotationItem:
    authority_id: str
    key_id: str
    public_key_sha256: str
    state: str
    registered_at: float
    retired_at: float | None
    age_seconds: float

    def __post_init__(self) -> None:
        authority = _identifier(self.authority_id, "authority_id", 200)
        key_id = _identifier(self.key_id, "key_id", 200)
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("timestamp authority state is unsupported.")
        registered = _timestamp(self.registered_at, "registered_at")
        retired = _optional_timestamp(self.retired_at, "retired_at")
        age = _duration(
            self.age_seconds,
            "age_seconds",
            100 * 365 * 24 * 60 * 60,
        )
        if state == "active" and retired is not None:
            raise ValueError("active timestamp key may not have retired_at.")
        if state == "retired" and retired is None:
            raise ValueError("retired timestamp key requires retired_at.")
        if retired is not None and retired < registered:
            raise ValueError("timestamp key retirement predates registration.")
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "public_key_sha256", fingerprint)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "registered_at", registered)
        object.__setattr__(self, "retired_at", retired)
        object.__setattr__(self, "age_seconds", age)


@dataclass(frozen=True)
class CustodyTimestampRotationReport:
    owner_id: str
    generated_at: float
    policy_digest: str
    classification: str
    active_count: int
    retired_count: int
    oldest_active_authority_id: str | None
    oldest_active_key_id: str | None
    newest_active_authority_id: str | None
    newest_active_key_id: str | None
    overlap_age_seconds: float | None
    items: tuple[CustodyTimestampRotationItem, ...]
    actions: tuple[str, ...]
    report_digest: str
    registry_mutation_performed: bool = False
    key_material_mutation_performed: bool = False
    contains_actor_ids: bool = False
    contains_raw_paths: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        policy = _digest(self.policy_digest, "policy_digest")
        classification = _identifier(self.classification, "classification", 100)
        if classification not in _CLASSIFICATIONS:
            raise ValueError("timestamp rotation classification is unsupported.")
        active = _integer(self.active_count, "active_count", 0, 10_000)
        retired = _integer(self.retired_count, "retired_count", 0, 10_000)
        oldest_authority = _optional_identifier(
            self.oldest_active_authority_id,
            "oldest_active_authority_id",
        )
        oldest_key = _optional_identifier(
            self.oldest_active_key_id,
            "oldest_active_key_id",
        )
        newest_authority = _optional_identifier(
            self.newest_active_authority_id,
            "newest_active_authority_id",
        )
        newest_key = _optional_identifier(
            self.newest_active_key_id,
            "newest_active_key_id",
        )
        if (oldest_authority is None) != (oldest_key is None):
            raise ValueError("oldest active authority/key fields must be paired.")
        if (newest_authority is None) != (newest_key is None):
            raise ValueError("newest active authority/key fields must be paired.")
        if active == 0 and any(
            value is not None
            for value in (
                oldest_authority,
                oldest_key,
                newest_authority,
                newest_key,
                self.overlap_age_seconds,
            )
        ):
            raise ValueError("empty active set may not name active keys.")
        if active > 0 and any(
            value is None
            for value in (
                oldest_authority,
                oldest_key,
                newest_authority,
                newest_key,
            )
        ):
            raise ValueError("active set requires oldest/newest identities.")
        overlap = (
            None
            if self.overlap_age_seconds is None
            else _duration(
                self.overlap_age_seconds,
                "overlap_age_seconds",
                100 * 365 * 24 * 60 * 60,
            )
        )
        if (active < 2) != (overlap is None):
            raise ValueError("overlap age differs from active key count.")
        items = tuple(self.items)
        if any(not isinstance(item, CustodyTimestampRotationItem) for item in items):
            raise ValueError("timestamp rotation items are invalid.")
        if active + retired != len(items):
            raise ValueError("timestamp rotation counts differ from items.")
        identities = tuple((item.authority_id, item.key_id) for item in items)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("timestamp rotation items must be unique and ordered.")
        actions = tuple(_identifier(value, "action", 100) for value in self.actions)
        if any(action not in _ACTIONS for action in actions) or len(set(actions)) != len(actions):
            raise ValueError("timestamp rotation actions are invalid.")
        stable = {
            "scope": "rigorousrag-custody-timestamp-rotation-report-v1",
            "owner_id": owner,
            "generated_at": generated,
            "policy_digest": policy,
            "classification": classification,
            "active_count": active,
            "retired_count": retired,
            "oldest_active_authority_id": oldest_authority,
            "oldest_active_key_id": oldest_key,
            "newest_active_authority_id": newest_authority,
            "newest_active_key_id": newest_key,
            "overlap_age_seconds": overlap,
            "items": [asdict(item) for item in items],
            "actions": list(actions),
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from timestamp rotation report.")
        if any(
            value is not False
            for value in (
                self.registry_mutation_performed,
                self.key_material_mutation_performed,
                self.contains_actor_ids,
                self.contains_raw_paths,
            )
        ):
            raise ValueError("timestamp rotation safety flags must be false.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "policy_digest", policy)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "active_count", active)
        object.__setattr__(self, "retired_count", retired)
        object.__setattr__(self, "oldest_active_authority_id", oldest_authority)
        object.__setattr__(self, "oldest_active_key_id", oldest_key)
        object.__setattr__(self, "newest_active_authority_id", newest_authority)
        object.__setattr__(self, "newest_active_key_id", newest_key)
        object.__setattr__(self, "overlap_age_seconds", overlap)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "report_digest", digest)


def assess_custody_timestamp_authority_rotation(
    *,
    owner_id: str,
    registry: Any,
    policy: CustodyTimestampRotationPolicy | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> CustodyTimestampRotationReport:
    owner = normalize_owner_id(owner_id)
    selected_policy = (
        CustodyTimestampRotationPolicy.default() if policy is None else policy
    )
    if not isinstance(selected_policy, CustodyTimestampRotationPolicy):
        raise ValueError("policy must be CustodyTimestampRotationPolicy.")
    if not callable(getattr(registry, "list", None)):
        raise ValueError("registry lacks the required read boundary.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    values = tuple(registry.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("timestamp rotation audit reached the bounded result limit.")
    seen: set[tuple[str, str]] = set()
    rendered: list[CustodyTimestampRotationItem] = []
    active_values: list[Any] = []
    for value in values:
        identity = (
            _identifier(value.authority_id, "authority_id", 200),
            _identifier(value.key_id, "key_id", 200),
        )
        if identity in seen:
            raise RuntimeError("timestamp authority registry returned duplicate keys.")
        seen.add(identity)
        registered_at = _timestamp(value.registered_at, "registered_at")
        state = _identifier(value.state, "state", 30)
        if state not in _STATES:
            raise RuntimeError("timestamp authority registry returned an unsupported state.")
        age = max(0.0, timestamp - registered_at)
        item = CustodyTimestampRotationItem(
            authority_id=identity[0],
            key_id=identity[1],
            public_key_sha256=value.public_key_sha256,
            state=state,
            registered_at=registered_at,
            retired_at=value.retired_at,
            age_seconds=age,
        )
        rendered.append(item)
        if state == "active":
            active_values.append(value)
    active_values.sort(
        key=lambda value: (
            float(value.registered_at),
            value.authority_id,
            value.key_id,
        )
    )
    retired_count = len(values) - len(active_values)
    oldest = None if not active_values else active_values[0]
    newest = None if not active_values else active_values[-1]
    overlap_age = (
        None
        if len(active_values) < 2
        else max(0.0, timestamp - float(newest.registered_at))
    )
    if not active_values:
        classification = "initial_key_required"
        actions = ("register_initial_timestamp_authority_key",)
    elif len(active_values) > selected_policy.maximum_active_keys:
        classification = "too_many_active_keys"
        actions = ("review_and_retire_excess_active_keys",)
    elif len(active_values) == 1:
        oldest_age = max(0.0, timestamp - float(oldest.registered_at))
        if oldest_age >= selected_policy.maximum_active_key_age_seconds:
            classification = "rotation_required_no_successor"
            actions = ("register_successor_timestamp_authority_key",)
        else:
            classification = "healthy_single_active"
            actions = ()
    elif overlap_age is not None and overlap_age < selected_policy.minimum_overlap_seconds:
        classification = "overlap_window_active"
        actions = ("retain_oldest_key_until_overlap_completes",)
    else:
        classification = "retire_oldest_after_overlap"
        actions = ("retire_oldest_timestamp_authority_key",)
    items = tuple(sorted(rendered, key=lambda value: (value.authority_id, value.key_id)))
    stable = {
        "scope": "rigorousrag-custody-timestamp-rotation-report-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "policy_digest": selected_policy.policy_digest,
        "classification": classification,
        "active_count": len(active_values),
        "retired_count": retired_count,
        "oldest_active_authority_id": None if oldest is None else oldest.authority_id,
        "oldest_active_key_id": None if oldest is None else oldest.key_id,
        "newest_active_authority_id": None if newest is None else newest.authority_id,
        "newest_active_key_id": None if newest is None else newest.key_id,
        "overlap_age_seconds": overlap_age,
        "items": [asdict(item) for item in items],
        "actions": list(actions),
    }
    return CustodyTimestampRotationReport(
        owner_id=owner,
        generated_at=timestamp,
        policy_digest=selected_policy.policy_digest,
        classification=classification,
        active_count=len(active_values),
        retired_count=retired_count,
        oldest_active_authority_id=None if oldest is None else oldest.authority_id,
        oldest_active_key_id=None if oldest is None else oldest.key_id,
        newest_active_authority_id=None if newest is None else newest.authority_id,
        newest_active_key_id=None if newest is None else newest.key_id,
        overlap_age_seconds=overlap_age,
        items=items,
        actions=actions,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "CustodyTimestampRotationItem",
    "CustodyTimestampRotationPolicy",
    "CustodyTimestampRotationReport",
    "assess_custody_timestamp_authority_rotation",
]
