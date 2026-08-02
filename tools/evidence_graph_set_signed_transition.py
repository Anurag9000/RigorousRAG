"""Read-only audit for transitioning legacy publication attempts to signed recovery."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.security import normalize_owner_id

_STATES = frozenset(
    {"planned", "running", "completed", "compensated", "failed", "cancelled"}
)
_PHASES = frozenset(
    {"planned", "candidate_stored", "pointer_activated", "verified", "compensated"}
)
_TERMINAL_STATES = frozenset({"completed", "compensated", "cancelled"})
_ACTIONS = frozenset(
    {
        "wait_for_authorization_only_lease",
        "reconcile_expired_authorization_only_attempt_before_transition",
        "cancel_authorization_only_then_reseed_signed",
        "do_not_claim_signed_provenance_reseed_with_current_pointer_if_needed",
        "no_signed_transition_required",
        "resolve_duplicate_nonterminal_attempts",
        "inspect_existing_signed_attempt_before_transition",
        "cancel_authorization_only_duplicate_after_signed_completion",
        "wait_for_authorization_only_lease_then_retire_duplicate",
        "preflight_expired_authorization_only_duplicate_retirement",
        "signed_attempt_already_completed",
    }
)
_NON_ACTIONABLE_ACTIONS = frozenset(
    {"no_signed_transition_required", "signed_attempt_already_completed"}
)
_MAX_LIMIT = 10_000


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _integer(value: Any, label: str, maximum: int = _MAX_LIMIT) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not 0 <= value <= maximum:
        raise ValueError(f"{label} must be between 0 and {maximum}.")
    return value


def _limit(value: Any) -> int:
    selected = _integer(value, "limit")
    if selected < 1:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}.")
    return selected


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


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
class SignedPublicationTransitionItem:
    operation_id: str
    graph_set_key: str
    authorization_state: str
    authorization_phase: str
    expected_current_set_id: str | None
    candidate_graph_set_id: str | None
    lease_active: bool
    lease_expires_at: float | None
    signed_attempt_present: bool
    signed_state: str | None
    signed_phase: str | None
    action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _digest(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self,
            "graph_set_key",
            _identifier(self.graph_set_key, "graph_set_key", 500),
        )
        state = _identifier(self.authorization_state, "authorization_state", 30)
        phase = _identifier(self.authorization_phase, "authorization_phase", 30)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("authorization state or phase is unsupported.")
        object.__setattr__(self, "authorization_state", state)
        object.__setattr__(self, "authorization_phase", phase)
        object.__setattr__(
            self,
            "expected_current_set_id",
            _optional_digest(self.expected_current_set_id, "expected_current_set_id"),
        )
        object.__setattr__(
            self,
            "candidate_graph_set_id",
            _optional_digest(self.candidate_graph_set_id, "candidate_graph_set_id"),
        )
        if not isinstance(self.lease_active, bool):
            raise ValueError("lease_active must be boolean.")
        lease_expires = (
            None
            if self.lease_expires_at is None
            else _timestamp(self.lease_expires_at, "lease_expires_at")
        )
        if self.lease_active and (state != "running" or lease_expires is None):
            raise ValueError("active leases require a running authorization attempt.")
        object.__setattr__(self, "lease_expires_at", lease_expires)
        if not isinstance(self.signed_attempt_present, bool):
            raise ValueError("signed_attempt_present must be boolean.")
        if self.signed_attempt_present:
            if self.signed_state is None or self.signed_phase is None:
                raise ValueError("signed attempt state must be complete.")
            signed_state = _identifier(self.signed_state, "signed_state", 30)
            signed_phase = _identifier(self.signed_phase, "signed_phase", 30)
            if signed_state not in _STATES or signed_phase not in _PHASES:
                raise ValueError("signed attempt state or phase is unsupported.")
            object.__setattr__(self, "signed_state", signed_state)
            object.__setattr__(self, "signed_phase", signed_phase)
        elif self.signed_state is not None or self.signed_phase is not None:
            raise ValueError("absent signed attempts may not contain state.")
        action = _identifier(self.action, "action", 200)
        if action not in _ACTIONS:
            raise ValueError("transition action is unsupported.")
        object.__setattr__(self, "action", action)


@dataclass(frozen=True)
class SignedPublicationTransitionReport:
    owner_id: str
    graph_set_key: str | None
    generated_at: float
    authorization_attempt_count: int
    signed_attempt_count: int
    actionable_count: int
    items: tuple[SignedPublicationTransitionItem, ...]
    report_digest: str
    mutation_performed: bool = False
    source_text_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key = (
            None
            if self.graph_set_key is None
            else _identifier(self.graph_set_key, "graph_set_key", 500)
        )
        generated = _timestamp(self.generated_at, "generated_at")
        authorization_count = _integer(
            self.authorization_attempt_count, "authorization_attempt_count"
        )
        signed_count = _integer(self.signed_attempt_count, "signed_attempt_count")
        actionable = _integer(self.actionable_count, "actionable_count")
        if not isinstance(self.items, tuple) or any(
            not isinstance(value, SignedPublicationTransitionItem)
            for value in self.items
        ):
            raise ValueError("items must be a tuple of transition items.")
        items = tuple(sorted(self.items, key=lambda value: value.operation_id))
        if len({value.operation_id for value in items}) != len(items):
            raise ValueError("transition items must have unique operation IDs.")
        if authorization_count != len(items):
            raise ValueError("authorization_attempt_count differs from transition items.")
        signed_matches = sum(value.signed_attempt_present for value in items)
        if signed_count < signed_matches:
            raise ValueError("signed_attempt_count is smaller than matched signed attempts.")
        expected_actionable = sum(
            value.action not in _NON_ACTIONABLE_ACTIONS for value in items
        )
        if actionable != expected_actionable:
            raise ValueError("actionable_count differs from transition actions.")
        if self.mutation_performed is not False or self.source_text_returned is not False:
            raise ValueError("transition reports must remain read-only and text-free.")
        stable = {
            "scope": "rigorousrag-signed-publication-transition-audit-v1",
            "owner_id": owner,
            "graph_set_key": key,
            "generated_at": generated,
            "authorization_attempt_count": authorization_count,
            "signed_attempt_count": signed_count,
            "actionable_count": actionable,
            "items": [asdict(value) for value in items],
        }
        report_digest = _digest(self.report_digest, "report_digest")
        if report_digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from transition report content.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(
            self, "authorization_attempt_count", authorization_count
        )
        object.__setattr__(self, "signed_attempt_count", signed_count)
        object.__setattr__(self, "actionable_count", actionable)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "report_digest", report_digest)


def _active_lease(value: Any, *, now: float) -> bool:
    return bool(
        value.state == "running"
        and value.lease_expires_at is not None
        and float(value.lease_expires_at) > now
    )


def _action(common: Any, signed: Any | None, *, now: float) -> tuple[bool, str]:
    if signed is not None:
        if signed.state == "completed":
            if common.state == "running":
                if _active_lease(common, now=now):
                    return (
                        True,
                        "wait_for_authorization_only_lease_then_retire_duplicate",
                    )
                return (
                    True,
                    "preflight_expired_authorization_only_duplicate_retirement",
                )
            if common.state in {"planned", "failed"}:
                return (
                    True,
                    "cancel_authorization_only_duplicate_after_signed_completion",
                )
            return False, "signed_attempt_already_completed"
        if (
            common.state not in _TERMINAL_STATES
            and signed.state not in _TERMINAL_STATES
        ):
            return True, "resolve_duplicate_nonterminal_attempts"
        return True, "inspect_existing_signed_attempt_before_transition"
    if common.state == "running":
        if _active_lease(common, now=now):
            return True, "wait_for_authorization_only_lease"
        return True, "reconcile_expired_authorization_only_attempt_before_transition"
    if common.state in {"planned", "failed"}:
        return True, "cancel_authorization_only_then_reseed_signed"
    if common.state == "completed":
        return (
            True,
            "do_not_claim_signed_provenance_reseed_with_current_pointer_if_needed",
        )
    return False, "no_signed_transition_required"


def assess_signed_publication_transition(
    *,
    owner_id: str,
    authorization_journal: Any,
    signed_journal: Any,
    graph_set_key: str | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> SignedPublicationTransitionReport:
    """Classify legacy attempts without changing either journal or graph state."""

    owner = normalize_owner_id(owner_id)
    key = (
        None
        if graph_set_key is None
        else _identifier(graph_set_key, "graph_set_key", 500)
    )
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _limit(limit)
    for journal in (authorization_journal, signed_journal):
        if not callable(getattr(journal, "list", None)):
            raise ValueError("journal lacks the required read boundary.")

    query = {"owner_id": owner, "graph_set_key": key, "limit": count}
    authorization_attempts = tuple(authorization_journal.list(**query))
    signed_attempts = tuple(signed_journal.list(**query))
    if len(authorization_attempts) >= count or len(signed_attempts) >= count:
        raise RuntimeError(
            "transition audit reached the bounded result limit; narrow the graph-set scope."
        )
    signed_by_id = {value.operation_id: value for value in signed_attempts}
    if len(signed_by_id) != len(signed_attempts):
        raise RuntimeError("signed publication journal returned duplicate operations.")

    rendered: list[SignedPublicationTransitionItem] = []
    actionable = 0
    for common in authorization_attempts:
        operation_id = _digest(common.operation_id, "operation_id")
        signed = signed_by_id.get(operation_id)
        is_actionable, action = _action(common, signed, now=timestamp)
        actionable += int(is_actionable)
        lease_expires = (
            None
            if getattr(common, "lease_expires_at", None) is None
            else _timestamp(common.lease_expires_at, "lease_expires_at")
        )
        rendered.append(
            SignedPublicationTransitionItem(
                operation_id=operation_id,
                graph_set_key=common.graph_set_key,
                authorization_state=common.state,
                authorization_phase=common.phase,
                expected_current_set_id=common.expected_current_set_id,
                candidate_graph_set_id=common.candidate_graph_set_id,
                lease_active=bool(
                    common.state == "running"
                    and lease_expires is not None
                    and lease_expires > timestamp
                ),
                lease_expires_at=lease_expires,
                signed_attempt_present=signed is not None,
                signed_state=None if signed is None else signed.state,
                signed_phase=None if signed is None else signed.phase,
                action=action,
            )
        )
    items = tuple(sorted(rendered, key=lambda value: value.operation_id))
    stable = {
        "scope": "rigorousrag-signed-publication-transition-audit-v1",
        "owner_id": owner,
        "graph_set_key": key,
        "generated_at": timestamp,
        "authorization_attempt_count": len(authorization_attempts),
        "signed_attempt_count": len(signed_attempts),
        "actionable_count": actionable,
        "items": [asdict(value) for value in items],
    }
    return SignedPublicationTransitionReport(
        owner_id=owner,
        graph_set_key=key,
        generated_at=timestamp,
        authorization_attempt_count=len(authorization_attempts),
        signed_attempt_count=len(signed_attempts),
        actionable_count=actionable,
        items=items,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedPublicationTransitionItem",
    "SignedPublicationTransitionReport",
    "assess_signed_publication_transition",
]
