"""Crash-recoverable execution for empty-target retirement snapshot restores."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
    _digest,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    complete_with_exact_target_lock,
    inspect_restored_target,
    restore_snapshot_into_empty_target,
    target_path_digest,
    validate_terminal_snapshot,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)


class SignedRetirementRestoreRecoveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        restore_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.restore_id = restore_id
        self.state = state
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True)
class SignedRetirementRestoreExecution:
    restore_id: str
    owner_id: str
    snapshot_digest: str
    target_path_digest: str
    snapshot_record_count: int
    state: str
    phase: str
    attempt_count: int
    target_verification_digest: str | None
    target_mutation_performed: bool
    restore_intent_mutation_performed: bool
    overwrite_performed: bool = False
    merge_performed: bool = False
    source_text_returned: bool = False


def _failure_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if len(name) <= 200 else "RestoreFailure"


def _execution(
    value: SignedRetirementRestoreAttempt,
    *,
    target_mutated: bool,
    intent_mutated: bool,
) -> SignedRetirementRestoreExecution:
    return SignedRetirementRestoreExecution(
        restore_id=value.restore_id,
        owner_id=value.owner_id,
        snapshot_digest=value.snapshot_digest,
        target_path_digest=value.target_path_digest,
        snapshot_record_count=value.snapshot_record_count,
        state=value.state,
        phase=value.phase,
        attempt_count=value.attempt_count,
        target_verification_digest=value.target_verification_digest,
        target_mutation_performed=target_mutated,
        restore_intent_mutation_performed=intent_mutated,
    )


def _load_bound_snapshot(
    attempt: SignedRetirementRestoreAttempt,
    *,
    snapshot_path: str,
    target_db_path: str,
) -> Any:
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    if (
        snapshot.owner_id != attempt.owner_id
        or snapshot.snapshot_digest != attempt.snapshot_digest
        or snapshot.record_count != attempt.snapshot_record_count
        or target_path_digest(target_db_path) != attempt.target_path_digest
    ):
        raise RuntimeError(
            "restore inputs differ from immutable operation scope."
        )
    return snapshot


def seed_signed_retirement_restore(
    *,
    snapshot_path: str,
    target_db_path: str,
    journal: SignedRetirementRestoreJournal,
    confirm_snapshot_digest: str,
    max_attempts: int = 3,
    now: float | None = None,
) -> tuple[SignedRetirementRestoreAttempt, str]:
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    confirmation = _digest(
        confirm_snapshot_digest,
        "confirm_snapshot_digest",
    )
    if confirmation != snapshot.snapshot_digest:
        raise ValueError("snapshot confirmation differs.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    target_digest = target_path_digest(target_db_path)
    attempt = SignedRetirementRestoreAttempt.create(
        owner_id=snapshot.owner_id,
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_digest,
        snapshot_record_count=snapshot.record_count,
        max_attempts=max_attempts,
        now=timestamp,
    )
    try:
        journal.get(attempt.restore_id)
    except KeyError:
        try:
            disposition, _verification = inspect_restored_target(
                snapshot=snapshot,
                target_db_path=target_db_path,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "a new restore intent requires an initialized empty target database."
            ) from exc
        if disposition != "empty":
            raise RuntimeError(
                "a new restore intent requires an initialized empty target database."
            )
        existing = journal.seed(attempt)
    else:
        existing = journal.seed(attempt)
        inspect_restored_target(
            snapshot=snapshot,
            target_db_path=target_db_path,
        )
    return existing, target_digest


def execute_signed_retirement_restore(
    restore_id: str,
    *,
    snapshot_path: str,
    target_db_path: str,
    worker_id: str,
    lease_seconds: int,
    journal: SignedRetirementRestoreJournal,
    now: float | None = None,
    _phase_hook: Callable[
        [str, SignedRetirementRestoreAttempt], None
    ]
    | None = None,
) -> SignedRetirementRestoreExecution:
    _integer(lease_seconds, "lease_seconds", 1, 86_400)

    def clock() -> float:
        return _timestamp(time.time() if now is None else now, "now")

    existing = journal.get(restore_id)
    if existing.state in {"completed", "cancelled"}:
        return _execution(
            existing,
            target_mutated=False,
            intent_mutated=False,
        )
    claimed = journal.claim(
        restore_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=clock(),
    )
    target_mutated = False
    intent_mutated = True
    try:
        snapshot = _load_bound_snapshot(
            claimed,
            snapshot_path=snapshot_path,
            target_db_path=target_db_path,
        )
        while True:
            phase_now = clock()
            current = journal.renew(
                claimed.restore_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                now=phase_now,
            )
            if current.phase == "planned":
                verification, inserted = restore_snapshot_into_empty_target(
                    snapshot=snapshot,
                    target_db_path=target_db_path,
                )
                target_mutated = target_mutated or inserted
                if _phase_hook is not None:
                    _phase_hook("target_committed", current)
                journal.record_target_committed(
                    current.restore_id,
                    worker_id=worker_id,
                    target_verification_digest=verification,
                    now=phase_now,
                )
                continue
            if current.phase == "target_committed":
                if _phase_hook is not None:
                    _phase_hook("before_complete", current)

                def complete(
                    verification: str,
                ) -> SignedRetirementRestoreAttempt:
                    if verification != current.target_verification_digest:
                        raise RuntimeError(
                            "restored target verification differs from committed phase."
                        )
                    return journal.complete(
                        current.restore_id,
                        worker_id=worker_id,
                        target_verification_digest=verification,
                        now=phase_now,
                    )

                completed, _verification = complete_with_exact_target_lock(
                    snapshot=snapshot,
                    target_db_path=target_db_path,
                    complete=complete,
                )
                return _execution(
                    completed,
                    target_mutated=target_mutated,
                    intent_mutated=intent_mutated,
                )
            raise RuntimeError(
                "restore attempt entered an unsupported phase."
            )
    except Exception as exc:
        failure = _failure_name(exc)
        current = journal.get(claimed.restore_id)
        if current.state == "running":
            try:
                current = journal.fail(
                    current.restore_id,
                    worker_id=worker_id,
                    failure_type=failure,
                    now=clock(),
                )
            except (KeyError, RuntimeError):
                current = journal.get(claimed.restore_id)
        raise SignedRetirementRestoreRecoveryError(
            f"signed retirement restore failed ({failure}).",
            restore_id=current.restore_id,
            state=current.state,
            phase=current.phase,
        ) from exc


def execute_next_signed_retirement_restore(
    *,
    owner_id: str,
    snapshot_path: str,
    target_db_path: str,
    worker_id: str,
    lease_seconds: int,
    journal: SignedRetirementRestoreJournal,
    now: float | None = None,
) -> SignedRetirementRestoreExecution | None:
    timestamp = _timestamp(time.time() if now is None else now, "now")
    restore_id = journal.next_claimable_id(
        owner_id=owner_id,
        now=timestamp,
    )
    if restore_id is None:
        return None
    return execute_signed_retirement_restore(
        restore_id,
        snapshot_path=snapshot_path,
        target_db_path=target_db_path,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        now=timestamp,
    )


__all__ = [
    "SignedRetirementRestoreExecution",
    "SignedRetirementRestoreRecoveryError",
    "execute_next_signed_retirement_restore",
    "execute_signed_retirement_restore",
    "seed_signed_retirement_restore",
]
