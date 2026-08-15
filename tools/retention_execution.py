"""Dependency-aware retention execution with plan fingerprints and exact confirmation.

Retention planning and execution are intentionally separate.  A store-specific handler
must revalidate each artifact identity immediately before deletion.  Dependencies marked
protected prevent deletion even when the nominal age policy would otherwise allow it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RetentionCandidate:
    artifact_id: str
    store: str
    identity_sha256: str
    created_at: float
    protected_by: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 500))
        object.__setattr__(self, "store", _text(self.store, "store", 64).lower())
        object.__setattr__(self, "identity_sha256", _sha(self.identity_sha256, "identity_sha256"))
        timestamp = float(self.created_at)
        if timestamp < 0:
            raise ValueError("created_at is invalid")
        object.__setattr__(self, "created_at", timestamp)
        if len(self.protected_by) > 10_000:
            raise ValueError("protected_by exceeds the item limit")
        object.__setattr__(self, "protected_by", tuple(dict.fromkeys(_text(item, "protected dependency", 500) for item in self.protected_by)))
        object.__setattr__(self, "reason", _text(self.reason, "reason", 2000, allow_empty=True))


@dataclass(frozen=True)
class RetentionAction:
    artifact_id: str
    store: str
    identity_sha256: str
    action: str
    reason: str

    def __post_init__(self) -> None:
        action = _text(self.action, "action", 32).lower()
        if action not in {"delete", "retain"}:
            raise ValueError("retention action must be delete or retain")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 500))
        object.__setattr__(self, "store", _text(self.store, "store", 64).lower())
        object.__setattr__(self, "identity_sha256", _sha(self.identity_sha256, "identity_sha256"))
        object.__setattr__(self, "reason", _text(self.reason, "reason", 2000))


@dataclass(frozen=True)
class RetentionPlan:
    policy_id: str
    cutoff_at: float
    actions: tuple[RetentionAction, ...]
    created_at: float
    plan_sha256: str

    @property
    def confirmation_token(self) -> str:
        return f"CONFIRM-RETENTION-{self.plan_sha256}"


def build_retention_plan(*, policy_id: str, candidates: Sequence[RetentionCandidate], cutoff_at: float) -> RetentionPlan:
    policy = _text(policy_id, "policy_id", 256)
    cutoff = float(cutoff_at)
    if cutoff < 0 or len(candidates) > 1_000_000:
        raise ValueError("retention planning inputs are invalid")
    actions: list[RetentionAction] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(candidates, key=lambda item: (item.store, item.artifact_id)):
        key = (candidate.store, candidate.artifact_id)
        if key in seen:
            raise ValueError("duplicate retention candidate")
        seen.add(key)
        if candidate.protected_by:
            actions.append(RetentionAction(candidate.artifact_id, candidate.store, candidate.identity_sha256, "retain", "protected_dependency:" + ",".join(candidate.protected_by[:20])))
        elif candidate.created_at > cutoff:
            actions.append(RetentionAction(candidate.artifact_id, candidate.store, candidate.identity_sha256, "retain", "younger_than_cutoff"))
        else:
            actions.append(RetentionAction(candidate.artifact_id, candidate.store, candidate.identity_sha256, "delete", candidate.reason or "retention_cutoff"))
    created = time.time()
    payload = {"policy_id": policy, "cutoff_at": cutoff, "actions": [asdict(item) for item in actions], "created_at": created}
    return RetentionPlan(policy, cutoff, tuple(actions), created, hashlib.sha256(_canonical(payload)).hexdigest())


class RetentionHandler(Protocol):
    def current_identity(self, artifact_id: str) -> str: ...
    def delete(self, artifact_id: str, *, expected_identity_sha256: str, operation_id: str) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class RetentionReceipt:
    artifact_id: str
    store: str
    status: str
    operation_id: str
    identity_sha256: str
    details: Mapping[str, str] = field(default_factory=dict)


def execute_retention_plan(plan: RetentionPlan, *, confirmation_token: str, handlers: Mapping[str, RetentionHandler], operation_id: str) -> tuple[RetentionReceipt, ...]:
    if confirmation_token != plan.confirmation_token:
        raise PermissionError("confirmation token does not match exact retention plan")
    op = _text(operation_id, "operation_id", 256)
    receipts: list[RetentionReceipt] = []
    for action in plan.actions:
        if action.action != "delete":
            receipts.append(RetentionReceipt(action.artifact_id, action.store, "retained", op, action.identity_sha256, {"reason": action.reason}))
            continue
        handler = handlers.get(action.store)
        if handler is None:
            raise RuntimeError(f"no retention handler registered for store {action.store}")
        current = _sha(handler.current_identity(action.artifact_id), "current identity")
        if current != action.identity_sha256:
            raise RuntimeError("retention target identity changed after planning")
        details = handler.delete(action.artifact_id, expected_identity_sha256=current, operation_id=op)
        receipts.append(RetentionReceipt(action.artifact_id, action.store, "deleted", op, current, {str(k)[:100]: str(v)[:1000] for k, v in details.items()}))
    return tuple(receipts)


__all__ = ["RetentionAction", "RetentionCandidate", "RetentionHandler", "RetentionPlan", "RetentionReceipt", "build_retention_plan", "execute_retention_plan"]
