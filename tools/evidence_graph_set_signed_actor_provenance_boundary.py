"""Validated runtime boundary for signed actor-use publication provenance."""

from __future__ import annotations

import math
import time
from typing import Any

from tools.evidence_graph_set_signed_actor_provenance import (
    SignedActorPublicationLedger,
    execute_signed_actor_publication_attempt,
    publish_signed_actor_governed_graph_set,
    signed_actor_publication_ledger,
)


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("now must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("now must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError("now must be finite and non-negative.")
    return selected


def execute_next_signed_actor_publication_attempt(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    ledger: Any,
    authorization_store: Any,
    actor_use_store: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
):
    timestamp = _timestamp(time.time() if now is None else now)
    operation_id = journal.next_claimable_id(owner_id=owner_id, now=timestamp)
    if operation_id is None:
        return None
    return execute_signed_actor_publication_attempt(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        ledger=ledger,
        authorization_store=authorization_store,
        actor_use_store=actor_use_store,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=timestamp,
    )


__all__ = [
    "SignedActorPublicationLedger",
    "execute_next_signed_actor_publication_attempt",
    "execute_signed_actor_publication_attempt",
    "publish_signed_actor_governed_graph_set",
    "signed_actor_publication_ledger",
]
