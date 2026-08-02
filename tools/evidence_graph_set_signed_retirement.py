"""Read-only preflight for retiring expired authorization-only publication duplicates."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_store import assess_graph_set_authority
from tools.security import normalize_owner_id

_DISPOSITIONS = frozenset(
    {
        "authorization_attempt_not_running",
        "wait_for_authorization_only_lease",
        "signed_attempt_not_completed",
        "signed_candidate_not_authoritative",
        "retire_expired_journal_only",
        "restore_signed_pointer_then_retire",
        "external_pointer_change_refusal",
    }
)
_MAX_PROPOSALS = 100_000


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


def _proposal_ids(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("proposal_ids must be an iterable.")
    rendered = tuple(_digest(value, "proposal_id") for value in values)
    if not rendered or len(rendered) > _MAX_PROPOSALS or len(set(rendered)) != len(rendered):
        raise ValueError("proposal_ids must be bounded, non-empty and unique.")
    return tuple(sorted(rendered))


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
class SignedPublicationRetirementPreflight:
    operation_id: str
    owner_id: str
    graph_set_key: str
    authorization_state: str
    authorization_phase: str
    signed_state: str
    signed_phase: str
    lease_expires_at: float | None
    lease_expired: bool
    authorization_candidate_set_id: str | None
    signed_candidate_set_id: str | None
    current_pointer_set_id: str | None
    signed_candidate_authoritative: bool | None
    signed_authority_digest: str | None
    eligible: bool
    disposition: str
    generated_at: float
    report_digest: str
    mutation_performed: bool = False
    source_text_returned: bool = False

    def __post_init__(self) -> None:
        operation = _digest(self.operation_id, "operation_id")
        owner = normalize_owner_id(self.owner_id)
        key = _identifier(self.graph_set_key, "graph_set_key", 500)
        auth_state = _identifier(self.authorization_state, "authorization_state", 30)
        auth_phase = _identifier(self.authorization_phase, "authorization_phase", 30)
        signed_state = _identifier(self.signed_state, "signed_state", 30)
        signed_phase = _identifier(self.signed_phase, "signed_phase", 30)
        lease_expires = (
            None
            if self.lease_expires_at is None
            else _timestamp(self.lease_expires_at, "lease_expires_at")
        )
        if not isinstance(self.lease_expired, bool):
            raise ValueError("lease_expired must be boolean.")
        auth_candidate = _optional_digest(
            self.authorization_candidate_set_id, "authorization_candidate_set_id"
        )
        signed_candidate = _optional_digest(
            self.signed_candidate_set_id, "signed_candidate_set_id"
        )
        current = _optional_digest(
            self.current_pointer_set_id, "current_pointer_set_id"
        )
        if self.signed_candidate_authoritative not in {True, False, None}:
            raise ValueError("signed_candidate_authoritative must be boolean or None.")
        authority_digest = _optional_digest(
            self.signed_authority_digest, "signed_authority_digest"
        )
        if (self.signed_candidate_authoritative is None) != (authority_digest is None):
            raise ValueError("signed authority fields must be complete or absent.")
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean.")
        disposition = _identifier(self.disposition, "disposition", 100)
        if disposition not in _DISPOSITIONS:
            raise ValueError("retirement disposition is unsupported.")
        expected_eligible = disposition in {
            "retire_expired_journal_only",
            "restore_signed_pointer_then_retire",
        }
        if self.eligible != expected_eligible:
            raise ValueError("eligible differs from retirement disposition.")
        generated = _timestamp(self.generated_at, "generated_at")
        if self.mutation_performed is not False or self.source_text_returned is not False:
            raise ValueError("retirement preflight must remain read-only and text-free.")
        stable = {
            "scope": "rigorousrag-signed-publication-retirement-preflight-v1",
            "operation_id": operation,
            "owner_id": owner,
            "graph_set_key": key,
            "authorization_state": auth_state,
            "authorization_phase": auth_phase,
            "signed_state": signed_state,
            "signed_phase": signed_phase,
            "lease_expires_at": lease_expires,
            "lease_expired": self.lease_expired,
            "authorization_candidate_set_id": auth_candidate,
            "signed_candidate_set_id": signed_candidate,
            "current_pointer_set_id": current,
            "signed_candidate_authoritative": self.signed_candidate_authoritative,
            "signed_authority_digest": authority_digest,
            "eligible": self.eligible,
            "disposition": disposition,
            "generated_at": generated,
        }
        report_digest = _digest(self.report_digest, "report_digest")
        if report_digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from retirement preflight content.")
        object.__setattr__(self, "operation_id", operation)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "authorization_state", auth_state)
        object.__setattr__(self, "authorization_phase", auth_phase)
        object.__setattr__(self, "signed_state", signed_state)
        object.__setattr__(self, "signed_phase", signed_phase)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(self, "authorization_candidate_set_id", auth_candidate)
        object.__setattr__(self, "signed_candidate_set_id", signed_candidate)
        object.__setattr__(self, "current_pointer_set_id", current)
        object.__setattr__(self, "signed_authority_digest", authority_digest)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "report_digest", report_digest)


def _same_scope(common: Any, signed: Any, *, owner_id: str, operation_id: str) -> None:
    common_owner = normalize_owner_id(common.owner_id)
    signed_owner = normalize_owner_id(signed.owner_id)
    common_operation = _digest(common.operation_id, "operation_id")
    signed_operation = _digest(signed.operation_id, "operation_id")
    common_key = _identifier(common.graph_set_key, "graph_set_key", 500)
    signed_key = _identifier(signed.graph_set_key, "graph_set_key", 500)
    if (
        common_owner != owner_id
        or signed_owner != owner_id
        or common_operation != operation_id
        or signed_operation != operation_id
        or common_key != signed_key
        or _proposal_ids(common.proposal_ids) != _proposal_ids(signed.proposal_ids)
        or _optional_digest(
            common.expected_current_set_id, "expected_current_set_id"
        )
        != _optional_digest(signed.expected_current_set_id, "expected_current_set_id")
    ):
        raise RuntimeError("publication attempts differ in immutable scope.")


def _load_signed_candidate(signed: Any, set_store: Any) -> Any:
    candidate_id = _digest(signed.candidate_graph_set_id, "signed_candidate_set_id")
    candidate_digest = _digest(
        signed.candidate_graph_set_digest, "signed_candidate_set_digest"
    )
    candidate = set_store.get(owner_id=signed.owner_id, graph_set_id=candidate_id)
    if (
        candidate.graph_set_id != candidate_id
        or candidate.graph_set_digest != candidate_digest
        or candidate.owner_id != signed.owner_id
        or candidate.graph_set_key != signed.graph_set_key
    ):
        raise RuntimeError("stored signed publication candidate identity is corrupt.")
    return candidate


def preflight_expired_signed_publication_duplicate_retirement(
    *,
    owner_id: str,
    operation_id: str,
    authorization_journal: Any,
    signed_journal: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
) -> SignedPublicationRetirementPreflight:
    """Inspect one duplicate without mutating journals, pointers or graph sets."""

    owner = normalize_owner_id(owner_id)
    operation = _digest(operation_id, "operation_id")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    common = authorization_journal.get(operation)
    signed = signed_journal.get(operation)
    _same_scope(common, signed, owner_id=owner, operation_id=operation)
    lease_expires = (
        None
        if common.lease_expires_at is None
        else _timestamp(common.lease_expires_at, "lease_expires_at")
    )
    lease_expired = bool(lease_expires is not None and lease_expires <= timestamp)
    signed_candidate_id = _optional_digest(
        signed.candidate_graph_set_id, "signed_candidate_set_id"
    )
    common_candidate_id = _optional_digest(
        common.candidate_graph_set_id, "authorization_candidate_set_id"
    )
    current_value = set_store.current(owner_id=owner, graph_set_key=common.graph_set_key)
    current_id = None if current_value is None else _digest(
        current_value.graph_set_id, "current_pointer_set_id"
    )
    authoritative: bool | None = None
    authority_digest: str | None = None

    if common.state != "running":
        disposition = "authorization_attempt_not_running"
    elif not lease_expired:
        disposition = "wait_for_authorization_only_lease"
    elif signed.state != "completed" or signed.phase != "verified":
        disposition = "signed_attempt_not_completed"
    else:
        candidate = _load_signed_candidate(signed, set_store)
        authority = assess_graph_set_authority(
            candidate, generations=generations, graphs=graphs
        )
        authoritative = bool(authority.authoritative_current)
        authority_digest = _digest(authority.authority_digest, "signed_authority_digest")
        if not authoritative:
            disposition = "signed_candidate_not_authoritative"
        elif current_id == signed_candidate_id:
            disposition = "retire_expired_journal_only"
        elif common_candidate_id is not None and current_id == common_candidate_id:
            disposition = "restore_signed_pointer_then_retire"
        else:
            disposition = "external_pointer_change_refusal"

    eligible = disposition in {
        "retire_expired_journal_only",
        "restore_signed_pointer_then_retire",
    }
    stable = {
        "scope": "rigorousrag-signed-publication-retirement-preflight-v1",
        "operation_id": operation,
        "owner_id": owner,
        "graph_set_key": common.graph_set_key,
        "authorization_state": common.state,
        "authorization_phase": common.phase,
        "signed_state": signed.state,
        "signed_phase": signed.phase,
        "lease_expires_at": lease_expires,
        "lease_expired": lease_expired,
        "authorization_candidate_set_id": common_candidate_id,
        "signed_candidate_set_id": signed_candidate_id,
        "current_pointer_set_id": current_id,
        "signed_candidate_authoritative": authoritative,
        "signed_authority_digest": authority_digest,
        "eligible": eligible,
        "disposition": disposition,
        "generated_at": timestamp,
    }
    return SignedPublicationRetirementPreflight(
        operation_id=operation,
        owner_id=owner,
        graph_set_key=common.graph_set_key,
        authorization_state=common.state,
        authorization_phase=common.phase,
        signed_state=signed.state,
        signed_phase=signed.phase,
        lease_expires_at=lease_expires,
        lease_expired=lease_expired,
        authorization_candidate_set_id=common_candidate_id,
        signed_candidate_set_id=signed_candidate_id,
        current_pointer_set_id=current_id,
        signed_candidate_authoritative=authoritative,
        signed_authority_digest=authority_digest,
        eligible=eligible,
        disposition=disposition,
        generated_at=timestamp,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedPublicationRetirementPreflight",
    "preflight_expired_signed_publication_duplicate_retirement",
]
