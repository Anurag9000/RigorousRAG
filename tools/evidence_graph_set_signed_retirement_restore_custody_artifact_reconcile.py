"""Crash-recoverable pre-restore custody artifact publication."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody import _file_sha256
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal import (
    RestoreCustodyArtifactJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    create_pre_restore_backup_receipt,
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    canonical_target_path,
    target_path_digest,
    validate_terminal_snapshot,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _path,
    _redirecting,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)


class RestoreCustodyArtifactRecoveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        artifact_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.artifact_id = artifact_id
        self.state = state
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True)
class RestoreCustodyArtifactExecution:
    artifact_id: str
    state: str
    phase: str
    snapshot_digest: str
    target_path_digest: str
    disposition: str | None
    backup_sha256: str | None
    backup_size_bytes: int | None
    receipt_digest: str | None
    attempt_count: int
    artifact_pair_created: bool
    orphan_recorded: bool
    journal_mutation_performed: bool = True
    artifact_deletion_performed: bool = False
    artifact_overwrite_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False


def artifact_path_digest(path: str | os.PathLike[str], *, label: str) -> str:
    selected = _path(path, label=label)
    return hashlib.sha256(str(selected).encode("utf-8")).hexdigest()


def _regular_exists(path: Path, *, label: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if _redirecting(info) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} exists but is not a regular file.")
    return True


def _scope(
    attempt: RestoreCustodyArtifactAttempt,
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
) -> tuple[Any, Path, Path]:
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    target = canonical_target_path(target_db_path)
    backup = _path(backup_output_path, label="backup_output_path")
    receipt = _path(receipt_output_path, label="receipt_output_path")
    if backup == receipt or backup == target or receipt == target:
        raise ValueError("custody artifact outputs must be distinct from each other and target.")
    if (
        snapshot.owner_id != attempt.owner_id
        or snapshot.snapshot_digest != attempt.snapshot_digest
        or target_path_digest(target) != attempt.target_path_digest
        or artifact_path_digest(backup, label="backup_output_path")
        != attempt.backup_path_digest
        or artifact_path_digest(receipt, label="receipt_output_path")
        != attempt.receipt_path_digest
    ):
        raise RuntimeError("custody artifact inputs differ from durable intent.")
    return snapshot, backup, receipt


def seed_restore_custody_artifact_attempt(
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    journal: RestoreCustodyArtifactJournal,
    max_attempts: int = 3,
    now: float | None = None,
) -> RestoreCustodyArtifactAttempt:
    if not isinstance(journal, RestoreCustodyArtifactJournal):
        raise ValueError("journal must be RestoreCustodyArtifactJournal.")
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    target = canonical_target_path(target_db_path)
    backup = _path(backup_output_path, label="backup_output_path")
    receipt = _path(receipt_output_path, label="receipt_output_path")
    if backup == receipt or backup == target or receipt == target:
        raise ValueError("custody artifact outputs must be distinct from each other and target.")
    if _regular_exists(backup, label="backup_output") or _regular_exists(
        receipt, label="receipt_output"
    ):
        raise RuntimeError("new artifact intent requires absent output paths.")
    timestamp = time.time() if now is None else now
    return journal.seed(
        RestoreCustodyArtifactAttempt.create(
            owner_id=snapshot.owner_id,
            snapshot_digest=snapshot.snapshot_digest,
            target_path_digest=target_path_digest(target),
            backup_path_digest=artifact_path_digest(
                backup, label="backup_output_path"
            ),
            receipt_path_digest=artifact_path_digest(
                receipt, label="receipt_output_path"
            ),
            max_attempts=max_attempts,
            now=timestamp,
        )
    )


def _execution(
    value: RestoreCustodyArtifactAttempt,
    *,
    created: bool,
) -> RestoreCustodyArtifactExecution:
    return RestoreCustodyArtifactExecution(
        artifact_id=value.artifact_id,
        state=value.state,
        phase=value.phase,
        snapshot_digest=value.snapshot_digest,
        target_path_digest=value.target_path_digest,
        disposition=value.disposition,
        backup_sha256=value.backup_sha256,
        backup_size_bytes=value.backup_size_bytes,
        receipt_digest=value.receipt_digest,
        attempt_count=value.attempt_count,
        artifact_pair_created=created,
        orphan_recorded=value.state == "orphaned",
    )


def _observe(
    *,
    attempt: RestoreCustodyArtifactAttempt,
    worker_id: str,
    journal: RestoreCustodyArtifactJournal,
    backup: Path,
    receipt: Path,
    now: float | None,
    created: bool,
) -> RestoreCustodyArtifactExecution | None:
    backup_exists = _regular_exists(backup, label="backup_output")
    receipt_exists = _regular_exists(receipt, label="receipt_output")
    if not backup_exists and not receipt_exists:
        return None
    backup_sha: str | None = None
    backup_size: int | None = None
    if backup_exists:
        try:
            backup_sha, backup_size = _file_sha256(
                backup,
                label="backup_artifact",
            )
        except Exception:
            backup_sha = backup_size = None
    if backup_exists and receipt_exists:
        try:
            verified = verify_pre_restore_backup_receipt(
                receipt_path=receipt,
                backup_path=backup,
            )
            if (
                verified.owner_id != attempt.owner_id
                or verified.snapshot_digest != attempt.snapshot_digest
                or verified.target_path_digest != attempt.target_path_digest
                or backup_sha != verified.backup_sha256
                or backup_size != verified.backup_size_bytes
            ):
                raise RuntimeError("published pair differs from durable artifact intent.")
            completed = journal.complete(
                attempt.artifact_id,
                worker_id=worker_id,
                backup_sha256=verified.backup_sha256,
                backup_size_bytes=verified.backup_size_bytes,
                receipt_digest=verified.receipt_digest,
                receipt_actor_id=verified.actor_id,
                receipt_binding_method=verified.binding_method,
                receipt_binding_digest=verified.binding_digest,
                now=now,
            )
            return _execution(completed, created=created)
        except Exception:
            orphaned = journal.orphan(
                attempt.artifact_id,
                worker_id=worker_id,
                disposition="artifact_collision",
                backup_sha256=backup_sha,
                backup_size_bytes=backup_size,
                now=now,
            )
            return _execution(orphaned, created=False)
    disposition = (
        "backup_without_receipt" if backup_exists else "receipt_without_backup"
    )
    orphaned = journal.orphan(
        attempt.artifact_id,
        worker_id=worker_id,
        disposition=disposition,
        backup_sha256=backup_sha,
        backup_size_bytes=backup_size,
        now=now,
    )
    return _execution(orphaned, created=False)


def execute_restore_custody_artifact_attempt(
    artifact_id: str,
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    actor: ReviewActorBinding,
    worker_id: str,
    lease_seconds: int,
    journal: RestoreCustodyArtifactJournal,
    now: float | None = None,
    _phase_hook: Callable[[str], None] | None = None,
) -> RestoreCustodyArtifactExecution:
    current = journal.get(artifact_id)
    _snapshot, backup, receipt = _scope(
        current,
        snapshot_path=snapshot_path,
        target_db_path=target_db_path,
        backup_output_path=backup_output_path,
        receipt_output_path=receipt_output_path,
    )
    if current.state in {"completed", "orphaned"}:
        return _execution(current, created=False)
    claimed = journal.claim(
        current.artifact_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=now,
    )
    try:
        claimed = journal.record_publication_intent(
            claimed.artifact_id,
            worker_id=worker_id,
            now=now,
        )
        if _phase_hook is not None:
            _phase_hook("publication_intent")
        observed = _observe(
            attempt=claimed,
            worker_id=worker_id,
            journal=journal,
            backup=backup,
            receipt=receipt,
            now=now,
            created=False,
        )
        if observed is not None:
            return observed
        create_pre_restore_backup_receipt(
            snapshot_path=snapshot_path,
            target_db_path=target_db_path,
            backup_output_path=backup,
            receipt_output_path=receipt,
            actor=actor,
            now=now,
        )
        if _phase_hook is not None:
            _phase_hook("artifacts_published")
        observed = _observe(
            attempt=claimed,
            worker_id=worker_id,
            journal=journal,
            backup=backup,
            receipt=receipt,
            now=now,
            created=True,
        )
        if observed is None:
            raise RuntimeError("artifact publication returned without outputs.")
        return observed
    except RestoreCustodyArtifactRecoveryError:
        raise
    except Exception as exc:
        latest = journal.get(claimed.artifact_id)
        if latest.state == "running":
            try:
                observed = _observe(
                    attempt=latest,
                    worker_id=worker_id,
                    journal=journal,
                    backup=backup,
                    receipt=receipt,
                    now=now,
                    created=False,
                )
                if observed is not None:
                    return observed
            except Exception:
                latest = journal.get(claimed.artifact_id)
            if latest.state == "running":
                failure_name = type(exc).__name__
                if len(failure_name) > 200:
                    failure_name = "ArtifactPublicationFailure"
                try:
                    latest = journal.fail(
                        latest.artifact_id,
                        worker_id=worker_id,
                        failure_type=failure_name,
                        now=now,
                    )
                except Exception:
                    latest = journal.get(claimed.artifact_id)
        raise RestoreCustodyArtifactRecoveryError(
            "custody artifact publication failed.",
            artifact_id=latest.artifact_id,
            state=latest.state,
            phase=latest.phase,
        ) from exc


def execute_next_restore_custody_artifact_attempt(
    *,
    owner_id: str,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    actor: ReviewActorBinding,
    worker_id: str,
    lease_seconds: int,
    journal: RestoreCustodyArtifactJournal,
    now: float | None = None,
) -> RestoreCustodyArtifactExecution | None:
    selected_now = time.time() if now is None else now
    artifact_id = journal.next_claimable_id(owner_id=owner_id, now=selected_now)
    if artifact_id is None:
        return None
    return execute_restore_custody_artifact_attempt(
        artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target_db_path,
        backup_output_path=backup_output_path,
        receipt_output_path=receipt_output_path,
        actor=actor,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        now=selected_now,
    )


__all__ = [
    "RestoreCustodyArtifactExecution",
    "RestoreCustodyArtifactRecoveryError",
    "artifact_path_digest",
    "execute_next_restore_custody_artifact_attempt",
    "execute_restore_custody_artifact_attempt",
    "seed_restore_custody_artifact_attempt",
]
