"""Read-only custody signer audit and rotation planning."""

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

_MAX_LIMIT = 10_000
_MAX_ISSUERS = 1_000
_CLASSIFICATIONS = frozenset(
    {
        "active_current",
        "active_rotation_due",
        "active_expired",
        "active_unapproved_issuer",
        "retired",
        "retired_unapproved_issuer",
    }
)
_ACTIONS = frozenset(
    {
        "register_initial_key",
        "register_successor",
        "maintain_overlap",
        "eligible_for_operator_retirement",
        "reduce_active_key_count",
        "investigate_unapproved_issuer",
        "no_action",
    }
)


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


def _nonnegative(value: Any, label: str) -> float:
    return _timestamp(value, label)


def _issuers(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("allowed_issuers must be an iterable.")
    result: set[str] = set()
    for value in values:
        result.add(_identifier(value, "allowed_issuer", 200))
        if len(result) > _MAX_ISSUERS:
            raise ValueError("allowed issuers exceed the limit.")
    return tuple(sorted(result))


@dataclass(frozen=True)
class CustodySignerRotationPolicy:
    maximum_active_keys: int
    maximum_key_age_seconds: float
    rotation_warning_seconds: float
    minimum_overlap_seconds: float
    allowed_issuers: tuple[str, ...]
    policy_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        maximum_active = _integer(
            self.maximum_active_keys,
            "maximum_active_keys",
            1,
            100,
        )
        maximum_age = _nonnegative(
            self.maximum_key_age_seconds,
            "maximum_key_age_seconds",
        )
        warning = _nonnegative(
            self.rotation_warning_seconds,
            "rotation_warning_seconds",
        )
        overlap = _nonnegative(
            self.minimum_overlap_seconds,
            "minimum_overlap_seconds",
        )
        if warning > maximum_age:
            raise ValueError("rotation warning exceeds maximum key age.")
        issuers = _issuers(self.allowed_issuers)
        if not issuers:
            raise ValueError("rotation policy requires at least one allowed issuer.")
        if self.schema_version != 1:
            raise ValueError("rotation policy schema is unsupported.")
        stable = {
            "scope": "rigorousrag-custody-signer-rotation-policy-v1",
            "maximum_active_keys": maximum_active,
            "maximum_key_age_seconds": maximum_age,
            "rotation_warning_seconds": warning,
            "minimum_overlap_seconds": overlap,
            "allowed_issuers": list(issuers),
            "schema_version": self.schema_version,
        }
        digest = _digest(self.policy_digest, "policy_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("policy_digest differs from signer rotation policy.")
        object.__setattr__(self, "maximum_active_keys", maximum_active)
        object.__setattr__(self, "maximum_key_age_seconds", maximum_age)
        object.__setattr__(self, "rotation_warning_seconds", warning)
        object.__setattr__(self, "minimum_overlap_seconds", overlap)
        object.__setattr__(self, "allowed_issuers", issuers)
        object.__setattr__(self, "policy_digest", digest)

    @classmethod
    def create(
        cls,
        *,
        maximum_active_keys: int = 2,
        maximum_key_age_seconds: float = 365 * 24 * 60 * 60,
        rotation_warning_seconds: float = 30 * 24 * 60 * 60,
        minimum_overlap_seconds: float = 7 * 24 * 60 * 60,
        allowed_issuers: Iterable[str],
    ) -> "CustodySignerRotationPolicy":
        issuers = _issuers(allowed_issuers)
        stable = {
            "scope": "rigorousrag-custody-signer-rotation-policy-v1",
            "maximum_active_keys": maximum_active_keys,
            "maximum_key_age_seconds": maximum_key_age_seconds,
            "rotation_warning_seconds": rotation_warning_seconds,
            "minimum_overlap_seconds": minimum_overlap_seconds,
            "allowed_issuers": list(issuers),
            "schema_version": 1,
        }
        return cls(
            maximum_active_keys=maximum_active_keys,
            maximum_key_age_seconds=maximum_key_age_seconds,
            rotation_warning_seconds=rotation_warning_seconds,
            minimum_overlap_seconds=minimum_overlap_seconds,
            allowed_issuers=issuers,
            policy_digest=_canonical_digest(stable),
        )


@dataclass(frozen=True)
class CustodySignerOperationalItem:
    key_id: str
    issuer: str
    public_key_sha256: str
    state: str
    registered_at: float
    retired_at: float | None
    age_seconds: float
    issuer_approved: bool
    classification: str
    action: str
    overlap_seconds: float | None


@dataclass(frozen=True)
class CustodySignerOperationalReport:
    owner_id: str
    generated_at: float
    policy_digest: str
    item_count: int
    active_count: int
    retired_count: int
    classification_counts: dict[str, int]
    action_counts: dict[str, int]
    items: tuple[CustodySignerOperationalItem, ...]
    report_digest: str
    registry_mutation_performed: bool = False
    key_material_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        policy = _digest(self.policy_digest, "policy_digest")
        item_count = _integer(self.item_count, "item_count", 0, _MAX_LIMIT)
        if item_count != len(self.items):
            raise ValueError("signer report count differs from items.")
        active = sum(item.state == "active" for item in self.items)
        retired = sum(item.state == "retired" for item in self.items)
        if self.active_count != active or self.retired_count != retired:
            raise ValueError("signer report state counts differ from items.")
        classifications = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        actions = {name: 0 for name in sorted(_ACTIONS)}
        seen: set[str] = set()
        for item in self.items:
            if item.key_id in seen:
                raise ValueError("signer report contains duplicate key IDs.")
            seen.add(item.key_id)
            if item.classification not in _CLASSIFICATIONS or item.action not in _ACTIONS:
                raise ValueError("signer report item is unsupported.")
            classifications[item.classification] += 1
            actions[item.action] += 1
        if dict(self.classification_counts) != classifications:
            raise ValueError("signer classification counts differ from items.")
        if dict(self.action_counts) != actions:
            raise ValueError("signer action counts differ from items.")
        for field in (
            "registry_mutation_performed",
            "key_material_mutation_performed",
            "source_text_returned",
            "raw_path_returned",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        stable = {
            "scope": "rigorousrag-custody-signer-operational-report-v1",
            "owner_id": owner,
            "generated_at": generated,
            "policy_digest": policy,
            "item_count": item_count,
            "active_count": active,
            "retired_count": retired,
            "classification_counts": classifications,
            "action_counts": actions,
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from signer report.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "policy_digest", policy)
        object.__setattr__(self, "item_count", item_count)
        object.__setattr__(self, "active_count", active)
        object.__setattr__(self, "retired_count", retired)
        object.__setattr__(self, "classification_counts", classifications)
        object.__setattr__(self, "action_counts", actions)
        object.__setattr__(self, "report_digest", digest)


def audit_custody_signer_rotation(
    *,
    owner_id: str,
    registry: Any,
    policy: CustodySignerRotationPolicy,
    now: float | None = None,
    limit: int = 1_000,
) -> CustodySignerOperationalReport:
    owner = normalize_owner_id(owner_id)
    if not isinstance(policy, CustodySignerRotationPolicy):
        raise ValueError("policy must be CustodySignerRotationPolicy.")
    if not callable(getattr(registry, "list", None)):
        raise ValueError("registry lacks the required read boundary.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    values = tuple(registry.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("signer audit reached the bounded result limit.")
    seen: set[str] = set()
    for value in values:
        if value.key_id in seen:
            raise RuntimeError("signer registry returned duplicate key IDs.")
        seen.add(value.key_id)
    active_values = sorted(
        (value for value in values if value.state == "active"),
        key=lambda value: (value.registered_at, value.key_id),
    )
    newest = active_values[-1] if active_values else None
    items: list[CustodySignerOperationalItem] = []
    for value in values:
        age = max(0.0, timestamp - value.registered_at)
        approved = value.issuer in policy.allowed_issuers
        overlap: float | None = None
        if value.state == "retired":
            classification = "retired" if approved else "retired_unapproved_issuer"
            action = "no_action" if approved else "investigate_unapproved_issuer"
        elif not approved:
            classification = "active_unapproved_issuer"
            action = "investigate_unapproved_issuer"
        elif age >= policy.maximum_key_age_seconds:
            classification = "active_expired"
            if newest is value:
                action = "register_successor"
            else:
                overlap = max(0.0, timestamp - newest.registered_at)
                action = (
                    "eligible_for_operator_retirement"
                    if overlap >= policy.minimum_overlap_seconds
                    else "maintain_overlap"
                )
        elif age >= (
            policy.maximum_key_age_seconds - policy.rotation_warning_seconds
        ):
            classification = "active_rotation_due"
            if newest is value:
                action = "register_successor"
            else:
                overlap = max(0.0, timestamp - newest.registered_at)
                action = (
                    "eligible_for_operator_retirement"
                    if overlap >= policy.minimum_overlap_seconds
                    else "maintain_overlap"
                )
        else:
            classification = "active_current"
            action = "no_action"
        if (
            value.state == "active"
            and len(active_values) > policy.maximum_active_keys
            and value is not newest
            and approved
        ):
            action = "reduce_active_key_count"
        items.append(
            CustodySignerOperationalItem(
                key_id=value.key_id,
                issuer=value.issuer,
                public_key_sha256=value.public_key_sha256,
                state=value.state,
                registered_at=value.registered_at,
                retired_at=value.retired_at,
                age_seconds=age,
                issuer_approved=approved,
                classification=classification,
                action=action,
                overlap_seconds=overlap,
            )
        )
    if not active_values:
        synthetic = CustodySignerOperationalItem(
            key_id="none",
            issuer=policy.allowed_issuers[0],
            public_key_sha256="0" * 64,
            state="retired",
            registered_at=timestamp,
            retired_at=timestamp,
            age_seconds=0.0,
            issuer_approved=True,
            classification="retired",
            action="register_initial_key",
            overlap_seconds=None,
        )
        items.append(synthetic)
    rendered = tuple(sorted(items, key=lambda item: (item.key_id, item.state)))
    classifications = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    actions = {name: 0 for name in sorted(_ACTIONS)}
    for item in rendered:
        classifications[item.classification] += 1
        actions[item.action] += 1
    active_count = sum(value.state == "active" for value in values)
    retired_count = sum(value.state == "retired" for value in values)
    stable = {
        "scope": "rigorousrag-custody-signer-operational-report-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "policy_digest": policy.policy_digest,
        "item_count": len(rendered),
        "active_count": active_count,
        "retired_count": retired_count,
        "classification_counts": classifications,
        "action_counts": actions,
        "items": [asdict(item) for item in rendered],
    }
    return CustodySignerOperationalReport(
        owner_id=owner,
        generated_at=timestamp,
        policy_digest=policy.policy_digest,
        item_count=len(rendered),
        active_count=active_count,
        retired_count=retired_count,
        classification_counts=classifications,
        action_counts=actions,
        items=rendered,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "CustodySignerOperationalItem",
    "CustodySignerOperationalReport",
    "CustodySignerRotationPolicy",
    "audit_custody_signer_rotation",
]
