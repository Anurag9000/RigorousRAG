"""Canonical signer rotation assessment with explicit registry-wide actions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations import (
    CustodySignerOperationalItem,
    CustodySignerRotationPolicy,
    audit_custody_signer_rotation as _audit_with_active_key,
)
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000
_GLOBAL_ACTIONS = frozenset({"register_initial_key"})


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


@dataclass(frozen=True)
class CustodySignerRotationAssessment:
    owner_id: str
    generated_at: float
    policy_digest: str
    item_count: int
    active_count: int
    retired_count: int
    global_actions: tuple[str, ...]
    classification_counts: dict[str, int]
    action_counts: dict[str, int]
    items: tuple[CustodySignerOperationalItem, ...]
    assessment_digest: str
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
            raise ValueError("rotation assessment count differs from items.")
        active = sum(item.state == "active" for item in self.items)
        retired = sum(item.state == "retired" for item in self.items)
        if active != self.active_count or retired != self.retired_count:
            raise ValueError("rotation assessment state counts differ from items.")
        actions = tuple(
            sorted(
                {
                    _identifier(value, "global_action", 80)
                    for value in self.global_actions
                }
            )
        )
        if any(value not in _GLOBAL_ACTIONS for value in actions):
            raise ValueError("rotation assessment global action is unsupported.")
        if (active == 0) != (actions == ("register_initial_key",)):
            raise ValueError("initial-key action differs from active-key state.")
        seen: set[str] = set()
        for item in self.items:
            if not isinstance(item, CustodySignerOperationalItem):
                raise ValueError("rotation assessment item is invalid.")
            if item.key_id in seen:
                raise ValueError("rotation assessment contains duplicate key IDs.")
            seen.add(item.key_id)
        for field in (
            "registry_mutation_performed",
            "key_material_mutation_performed",
            "source_text_returned",
            "raw_path_returned",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        stable = {
            "scope": "rigorousrag-custody-signer-rotation-assessment-v1",
            "owner_id": owner,
            "generated_at": generated,
            "policy_digest": policy,
            "item_count": item_count,
            "active_count": active,
            "retired_count": retired,
            "global_actions": list(actions),
            "classification_counts": dict(self.classification_counts),
            "action_counts": dict(self.action_counts),
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.assessment_digest, "assessment_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("assessment_digest differs from rotation assessment.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "policy_digest", policy)
        object.__setattr__(self, "item_count", item_count)
        object.__setattr__(self, "active_count", active)
        object.__setattr__(self, "retired_count", retired)
        object.__setattr__(self, "global_actions", actions)
        object.__setattr__(self, "assessment_digest", digest)


def assess_custody_signer_rotation(
    *,
    owner_id: str,
    registry: Any,
    policy: CustodySignerRotationPolicy,
    now: float | None = None,
    limit: int = 1_000,
) -> CustodySignerRotationAssessment:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    values = tuple(registry.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("signer assessment reached the bounded result limit.")
    active_values = tuple(value for value in values if value.state == "active")
    if active_values:
        report = _audit_with_active_key(
            owner_id=owner,
            registry=registry,
            policy=policy,
            now=timestamp,
            limit=count,
        )
        items = report.items
        classifications = report.classification_counts
        action_counts = report.action_counts
        global_actions: tuple[str, ...] = ()
    else:
        items_list: list[CustodySignerOperationalItem] = []
        classifications = {
            "active_current": 0,
            "active_expired": 0,
            "active_rotation_due": 0,
            "active_unapproved_issuer": 0,
            "retired": 0,
            "retired_unapproved_issuer": 0,
        }
        action_counts = {
            "eligible_for_operator_retirement": 0,
            "investigate_unapproved_issuer": 0,
            "maintain_overlap": 0,
            "no_action": 0,
            "reduce_active_key_count": 0,
            "register_initial_key": 0,
            "register_successor": 0,
        }
        seen: set[str] = set()
        for value in values:
            if value.key_id in seen:
                raise RuntimeError("signer registry returned duplicate key IDs.")
            seen.add(value.key_id)
            approved = value.issuer in policy.allowed_issuers
            classification = "retired" if approved else "retired_unapproved_issuer"
            action = "no_action" if approved else "investigate_unapproved_issuer"
            classifications[classification] += 1
            action_counts[action] += 1
            items_list.append(
                CustodySignerOperationalItem(
                    key_id=value.key_id,
                    issuer=value.issuer,
                    public_key_sha256=value.public_key_sha256,
                    state="retired",
                    registered_at=value.registered_at,
                    retired_at=value.retired_at,
                    age_seconds=max(0.0, timestamp - value.registered_at),
                    issuer_approved=approved,
                    classification=classification,
                    action=action,
                    overlap_seconds=None,
                )
            )
        items = tuple(sorted(items_list, key=lambda item: item.key_id))
        global_actions = ("register_initial_key",)
        action_counts["register_initial_key"] = 1
    stable = {
        "scope": "rigorousrag-custody-signer-rotation-assessment-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "policy_digest": policy.policy_digest,
        "item_count": len(items),
        "active_count": sum(item.state == "active" for item in items),
        "retired_count": sum(item.state == "retired" for item in items),
        "global_actions": list(global_actions),
        "classification_counts": classifications,
        "action_counts": action_counts,
        "items": [asdict(item) for item in items],
    }
    return CustodySignerRotationAssessment(
        owner_id=owner,
        generated_at=timestamp,
        policy_digest=policy.policy_digest,
        item_count=len(items),
        active_count=stable["active_count"],
        retired_count=stable["retired_count"],
        global_actions=global_actions,
        classification_counts=classifications,
        action_counts=action_counts,
        items=items,
        assessment_digest=_canonical_digest(stable),
    )


__all__ = [
    "CustodySignerRotationAssessment",
    "assess_custody_signer_rotation",
]
