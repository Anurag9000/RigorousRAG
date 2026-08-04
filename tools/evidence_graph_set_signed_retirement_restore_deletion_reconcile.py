"""Crash-recoverable execution of one authorized restore-intent deletion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_authorizations import (
    preflight_signed_retirement_restore_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_consumption import (
    mark_authorization_consumed,
    release_authorization_reservation,
    reserve_authorization_for_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_execution_contracts import (
    SignedRetirementRestoreDeletionAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_marker_boundary import (
    ensure_active_deletion_marker,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    abort_deletion_marker,
    canonical_restore_record_digest,
    delete_restore_with_tombstone,
    verify_deleted_tombstone,
)


class SignedRetirementRestoreDeletionRecoveryError(RuntimeError):
    """Generic durable failure returned by the deletion execution boundary."""

    def __init__(
        self,
        message: str,
        *,
        deletion_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.deletion_id = deletion_id
        self.state = state
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True)
class SignedRetirementRestoreDeletionExecution:
    deletion_id: str
    authorization_id: str
    restore_id: str
    state: str
    phase: str
    marker_digest: str | None
    tombstone_digest: str | None
    attempt_count: int
    restore_row_deleted: bool
    authorization_consumed: bool
    custody_preserved: bool = True
    holds_deleted: bool = False
    custody_deleted: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


def _clock(value: float | None) -> float:
    return _timestamp(time.time() if value is None else value, "now")


def _require_custody(attempt: Any, custody_store: Any) -> Any | None:
    if attempt.custody_id is None:
        if attempt.restore_state == "completed":
            raise RuntimeError(
                "completed deletion lacks custody evidence."
            )
        try:
            custody_store.get_for_restore(attempt.restore_id)
        except KeyError:
            return None
        raise RuntimeError(
            "unexpected custody evidence differs from deletion scope."
        )
    value = custody_store.get_for_restore(attempt.restore_id)
    if (
        value.custody_id != attempt.custody_id
        or value.manifest_digest != attempt.custody_manifest_digest
        or value.owner_id != attempt.owner_id
        or value.restore_id != attempt.restore_id
        or value.snapshot_digest != attempt.snapshot_digest
        or value.target_path_digest != attempt.target_path_digest
    ):
        raise RuntimeError("custody evidence differs from deletion scope.")
    if attempt.restore_state == "completed" and value.state != "post_bound":
        raise RuntimeError(
            "completed deletion requires post-bound custody."
        )
    return value


def seed_signed_retirement_restore_deletion(
    *,
    authorization_id: str,
    authorization_store: Any,
    deletion_journal: Any,
    restore_journal: Any,
    hold_store: Any,
    custody_store: Any,
    now: float | None = None,
    max_attempts: int = 3,
) -> tuple[SignedRetirementRestoreDeletionAttempt, Any]:
    timestamp = _clock(now)
    authorization = authorization_store.get(
        _digest(authorization_id, "authorization_id")
    )
    report = preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=restore_journal,
        hold_store=hold_store,
        now=timestamp,
    )
    if not report.eligible_for_future_deletion_executor:
        raise RuntimeError(
            "deletion authorization is not currently eligible."
        )
    restore = restore_journal.get(authorization.restore_id)
    if restore.state not in {"completed", "cancelled"}:
        raise RuntimeError("restore is not terminal.")
    try:
        custody = custody_store.get_for_restore(restore.restore_id)
    except KeyError:
        custody = None
    if restore.state == "completed" and (
        custody is None or custody.state != "post_bound"
    ):
        raise RuntimeError(
            "completed restore lacks post-bound custody."
        )
    if custody is not None and (
        custody.owner_id != restore.owner_id
        or custody.restore_id != restore.restore_id
        or custody.snapshot_digest != restore.snapshot_digest
        or custody.target_path_digest != restore.target_path_digest
    ):
        raise RuntimeError("custody escaped restore scope.")
    attempt = SignedRetirementRestoreDeletionAttempt.create(
        authorization_id=authorization.authorization_id,
        authorization_digest=authorization.authorization_digest,
        owner_id=restore.owner_id,
        restore_id=restore.restore_id,
        snapshot_digest=restore.snapshot_digest,
        target_path_digest=restore.target_path_digest,
        restore_state=restore.state,
        restore_phase=restore.phase,
        restore_record_digest=canonical_restore_record_digest(restore),
        custody_id=None if custody is None else custody.custody_id,
        custody_manifest_digest=(
            None if custody is None else custody.manifest_digest
        ),
        max_attempts=max_attempts,
        now=timestamp,
    )
    return deletion_journal.seed(attempt), report


def _result(
    value: Any,
    *,
    restore_row_deleted: bool,
    authorization_consumed: bool,
) -> SignedRetirementRestoreDeletionExecution:
    return SignedRetirementRestoreDeletionExecution(
        deletion_id=value.deletion_id,
        authorization_id=value.authorization_id,
        restore_id=value.restore_id,
        state=value.state,
        phase=value.phase,
        marker_digest=value.marker_digest,
        tombstone_digest=value.tombstone_digest,
        attempt_count=value.attempt_count,
        restore_row_deleted=restore_row_deleted,
        authorization_consumed=authorization_consumed,
    )


def execute_signed_retirement_restore_deletion(
    deletion_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    deletion_journal: Any,
    authorization_store: Any,
    restore_journal: Any,
    hold_store: Any,
    custody_store: Any,
    now: float | None = None,
    _phase_hook: Callable[[str, Any], None] | None = None,
) -> SignedRetirementRestoreDeletionExecution:
    timestamp = _clock(now)
    current = deletion_journal.get(deletion_id)
    if current.state == "completed":
        return _result(
            current,
            restore_row_deleted=False,
            authorization_consumed=True,
        )
    claimed = deletion_journal.claim(
        current.deletion_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=timestamp,
    )
    try:
        authorization = authorization_store.get(claimed.authorization_id)
        if authorization.authorization_digest != claimed.authorization_digest:
            raise RuntimeError(
                "authorization differs from deletion scope."
            )
        _require_custody(claimed, custody_store)
        marker, marker_state, tombstone = ensure_active_deletion_marker(
            restore_journal,
            claimed,
            now=timestamp,
        )
        if _phase_hook is not None:
            _phase_hook("marker_committed", claimed)
        if marker_state != "deleted":
            report = preflight_signed_retirement_restore_deletion(
                authorization=authorization,
                restore_journal=restore_journal,
                hold_store=hold_store,
                now=timestamp,
            )
            if not report.eligible_for_future_deletion_executor:
                abort_deletion_marker(
                    restore_journal,
                    claimed,
                    now=timestamp,
                )
                raise RuntimeError(
                    "deletion authorization changed after marker creation."
                )
            consumption = reserve_authorization_for_deletion(
                authorization_store,
                authorization_id=claimed.authorization_id,
                deletion_id=claimed.deletion_id,
                now=timestamp,
            )
            if _phase_hook is not None:
                _phase_hook("authorization_reserved", claimed)
            active_holds = hold_store.active_restore_ids(
                owner_id=claimed.owner_id,
                limit=10_000,
            )
            if claimed.restore_id in active_holds:
                release_authorization_reservation(
                    authorization_store,
                    authorization_id=claimed.authorization_id,
                    deletion_id=claimed.deletion_id,
                )
                abort_deletion_marker(
                    restore_journal,
                    claimed,
                    now=timestamp,
                )
                raise RuntimeError(
                    "durable legal hold appeared before deletion."
                )
            deletion_journal.record_marker_active(
                claimed.deletion_id,
                worker_id=worker_id,
                marker_digest=marker,
                now=timestamp,
            )
            marker, tombstone, _performed = delete_restore_with_tombstone(
                restore_journal,
                claimed,
                authorization_consumption_digest=(
                    consumption.consumption_digest
                ),
                now=timestamp,
            )
            if _phase_hook is not None:
                _phase_hook("restore_deleted", claimed)
        value = deletion_journal.get(claimed.deletion_id)
        if value.phase == "planned":
            value = deletion_journal.record_marker_active(
                value.deletion_id,
                worker_id=worker_id,
                marker_digest=marker,
                now=timestamp,
            )
        if value.phase == "marker_active":
            value = deletion_journal.record_restore_deleted(
                value.deletion_id,
                worker_id=worker_id,
                marker_digest=marker,
                tombstone_digest=tombstone,
                now=timestamp,
            )
        if _phase_hook is not None:
            _phase_hook("restore_deleted_recorded", value)
        consumption = mark_authorization_consumed(
            authorization_store,
            authorization_id=value.authorization_id,
            deletion_id=value.deletion_id,
            now=timestamp,
        )
        marker, tombstone = verify_deleted_tombstone(
            restore_journal,
            value,
        )
        _require_custody(value, custody_store)
        completed = deletion_journal.complete(
            value.deletion_id,
            worker_id=worker_id,
            marker_digest=marker,
            tombstone_digest=tombstone,
            now=timestamp,
        )
        return _result(
            completed,
            restore_row_deleted=True,
            authorization_consumed=(consumption.state == "consumed"),
        )
    except Exception as exc:
        value = deletion_journal.get(claimed.deletion_id)
        if value.state == "running":
            try:
                value = deletion_journal.fail(
                    value.deletion_id,
                    worker_id=worker_id,
                    failure_type=type(exc).__name__,
                    now=timestamp,
                )
            except (KeyError, RuntimeError):
                value = deletion_journal.get(claimed.deletion_id)
        raise SignedRetirementRestoreDeletionRecoveryError(
            "restore deletion failed.",
            deletion_id=value.deletion_id,
            state=value.state,
            phase=value.phase,
        ) from exc


def execute_next_signed_retirement_restore_deletion(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    deletion_journal: Any,
    authorization_store: Any,
    restore_journal: Any,
    hold_store: Any,
    custody_store: Any,
    now: float | None = None,
) -> SignedRetirementRestoreDeletionExecution | None:
    timestamp = _clock(now)
    deletion_id = deletion_journal.next_claimable_id(
        owner_id=owner_id,
        now=timestamp,
    )
    if deletion_id is None:
        return None
    return execute_signed_retirement_restore_deletion(
        deletion_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        deletion_journal=deletion_journal,
        authorization_store=authorization_store,
        restore_journal=restore_journal,
        hold_store=hold_store,
        custody_store=custody_store,
        now=timestamp,
    )


__all__ = [
    "SignedRetirementRestoreDeletionExecution",
    "SignedRetirementRestoreDeletionRecoveryError",
    "execute_next_signed_retirement_restore_deletion",
    "execute_signed_retirement_restore_deletion",
    "seed_signed_retirement_restore_deletion",
]
