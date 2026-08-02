"""Canonical revalidating boundary for custody artifact publication."""

from __future__ import annotations

import os
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_custody import _file_sha256
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_reconcile import (
    RestoreCustodyArtifactExecution,
    RestoreCustodyArtifactRecoveryError,
    _execution,
    _scope,
    execute_restore_custody_artifact_attempt as _execute,
    seed_restore_custody_artifact_attempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_pre_restore_backup_receipt,
)


def _verify_completed_pair(
    current: Any,
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
) -> RestoreCustodyArtifactExecution:
    _snapshot, backup, receipt = _scope(
        current,
        snapshot_path=snapshot_path,
        target_db_path=target_db_path,
        backup_output_path=backup_output_path,
        receipt_output_path=receipt_output_path,
    )
    try:
        verified = verify_pre_restore_backup_receipt(
            receipt_path=receipt,
            backup_path=backup,
        )
        backup_sha, backup_size = _file_sha256(
            backup,
            label="backup_artifact",
        )
        if (
            verified.owner_id != current.owner_id
            or verified.snapshot_digest != current.snapshot_digest
            or verified.target_path_digest != current.target_path_digest
            or verified.backup_sha256 != current.backup_sha256
            or verified.backup_size_bytes != current.backup_size_bytes
            or verified.receipt_digest != current.receipt_digest
            or verified.actor_id != current.receipt_actor_id
            or verified.binding_method != current.receipt_binding_method
            or verified.binding_digest != current.receipt_binding_digest
            or backup_sha != current.backup_sha256
            or backup_size != current.backup_size_bytes
        ):
            raise RuntimeError("live custody artifact pair differs from completed journal state.")
    except Exception as exc:
        raise RestoreCustodyArtifactRecoveryError(
            "completed custody artifact pair is unavailable or invalid.",
            artifact_id=current.artifact_id,
            state=current.state,
            phase=current.phase,
        ) from exc
    return _execution(current, created=False)


def execute_restore_custody_artifact_attempt(
    artifact_id: str,
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    actor: Any,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    now: float | None = None,
    _phase_hook: Any = None,
) -> RestoreCustodyArtifactExecution:
    current = journal.get(artifact_id)
    if current.state == "completed":
        return _verify_completed_pair(
            current,
            snapshot_path=snapshot_path,
            target_db_path=target_db_path,
            backup_output_path=backup_output_path,
            receipt_output_path=receipt_output_path,
        )
    return _execute(
        artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target_db_path,
        backup_output_path=backup_output_path,
        receipt_output_path=receipt_output_path,
        actor=actor,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        now=now,
        _phase_hook=_phase_hook,
    )


__all__ = [
    "RestoreCustodyArtifactExecution",
    "RestoreCustodyArtifactRecoveryError",
    "execute_restore_custody_artifact_attempt",
    "seed_restore_custody_artifact_attempt",
]
