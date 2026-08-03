"""Governed Ed25519 custody signer public-key registry."""

from __future__ import annotations

import os
import time

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _identifier,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_key_contracts import (
    ALGORITHM,
    CustodySignerKeyRecord,
    load_ed25519_public_key,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_key_store import (
    CustodySignerKeyRegistry,
)
from tools.security import normalize_owner_id


def register_custody_signer_key(
    *,
    registry: CustodySignerKeyRegistry,
    owner_id: str,
    key_id: str,
    public_key_path: str | os.PathLike[str],
    actor: ReviewActorBinding,
    valid_from: float,
    valid_until: float | None = None,
    now: float | None = None,
) -> CustodySignerKeyRecord:
    """Register one immutable key scope or return its exact active replay."""

    if not isinstance(registry, CustodySignerKeyRegistry):
        raise ValueError("registry must be CustodySignerKeyRegistry.")
    _key, raw_base64, fingerprint = load_ed25519_public_key(public_key_path)
    owner = normalize_owner_id(owner_id)
    selected_key = _identifier(key_id, "key_id", 200)
    selected_from = _timestamp(valid_from, "valid_from")
    selected_until = (
        None if valid_until is None else _timestamp(valid_until, "valid_until")
    )
    try:
        existing = registry.get(owner_id=owner, key_id=selected_key)
    except KeyError:
        existing = None
    if existing is not None:
        same_scope = (
            existing.algorithm == ALGORITHM
            and existing.public_key_raw_base64 == raw_base64
            and existing.public_key_sha256 == fingerprint
            and existing.valid_from == selected_from
            and existing.valid_until == selected_until
        )
        if not same_scope:
            raise RuntimeError("signer key identity collision.")
        if existing.state != "active":
            raise RuntimeError("retired signer key cannot be reactivated.")
        return existing
    timestamp = _timestamp(time.time() if now is None else now, "now")
    value = CustodySignerKeyRecord.active(
        owner_id=owner,
        key_id=selected_key,
        public_key_raw_base64=raw_base64,
        public_key_sha256=fingerprint,
        valid_from=selected_from,
        valid_until=selected_until,
        actor_binding=actor,
        now=timestamp,
    )
    return registry.register(value)


__all__ = [
    "CustodySignerKeyRecord",
    "CustodySignerKeyRegistry",
    "load_ed25519_public_key",
    "register_custody_signer_key",
]
