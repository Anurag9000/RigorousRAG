"""Canonical failure-normalizing boundary for signed retirement execution."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_set_signed_retirement_contracts import _timestamp
from tools.evidence_graph_set_signed_retirement_reconcile import (
    SignedPublicationRetirementExecution,
    SignedPublicationRetirementRecoveryError,
    execute_signed_publication_retirement as _execute,
    seed_signed_publication_retirement,
)


def _failure_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if len(name) <= 200 else "RetirementFailure"


def execute_signed_publication_retirement(
    retirement_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    retirement_journal: Any,
    authorization_journal: Any,
    signed_journal: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
    _phase_hook: Any = None,
) -> SignedPublicationRetirementExecution:
    selected_now = None if now is None else _timestamp(now, "now")
    try:
        return _execute(
            retirement_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            retirement_journal=retirement_journal,
            authorization_journal=authorization_journal,
            signed_journal=signed_journal,
            set_store=set_store,
            generations=generations,
            graphs=graphs,
            now=selected_now,
            _phase_hook=_phase_hook,
        )
    except SignedPublicationRetirementRecoveryError:
        raise
    except Exception as exc:
        failure = _failure_name(exc)
        failure_now = _timestamp(
            time.time() if selected_now is None else selected_now,
            "now",
        )
        current = retirement_journal.get(retirement_id)
        if current.state == "running":
            try:
                current = retirement_journal.fail(
                    current.retirement_id,
                    worker_id=worker_id,
                    failure_type=failure,
                    now=failure_now,
                )
            except (KeyError, RuntimeError):
                current = retirement_journal.get(retirement_id)
        raise SignedPublicationRetirementRecoveryError(
            f"signed publication retirement failed ({failure}).",
            retirement_id=current.retirement_id,
            state=current.state,
            phase=current.phase,
        ) from exc


def execute_next_signed_publication_retirement(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    retirement_journal: Any,
    authorization_journal: Any,
    signed_journal: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
) -> SignedPublicationRetirementExecution | None:
    timestamp = _timestamp(time.time() if now is None else now, "now")
    retirement_id = retirement_journal.next_claimable_id(
        owner_id=owner_id,
        now=timestamp,
    )
    if retirement_id is None:
        return None
    return execute_signed_publication_retirement(
        retirement_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        retirement_journal=retirement_journal,
        authorization_journal=authorization_journal,
        signed_journal=signed_journal,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=timestamp,
    )


__all__ = [
    "SignedPublicationRetirementExecution",
    "SignedPublicationRetirementRecoveryError",
    "execute_next_signed_publication_retirement",
    "execute_signed_publication_retirement",
    "seed_signed_publication_retirement",
]
