"""Pre-backup and post-comparison custody receipts for retirement restores."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_readonly import (
    ReadOnlySignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_contracts import (
    PostRestoreComparisonReceipt,
    PreRestoreBackupReceipt,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    canonical_target_path,
    inspect_restored_target,
    target_path_digest,
    validate_terminal_snapshot,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _pairs,
    _path,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)

_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_MAX_BACKUP_BYTES = 1024 * 1024 * 1024 * 1024


def _actor(actor: ReviewActorBinding, *, now: float) -> ReviewActorBinding:
    if not isinstance(actor, ReviewActorBinding):
        raise ValueError("actor must be ReviewActorBinding.")
    if actor.expires_at is not None and actor.expires_at < now:
        raise PermissionError("custody actor binding is expired.")
    return actor


def _read_regular(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum: int,
) -> bytes:
    selected = _path(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file.")
        if before.st_size <= 0 or before.st_size > maximum:
            raise ValueError(f"{label} size is invalid.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while being read.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_size) != int(before.st_size)
        ):
            raise RuntimeError(f"{label} identity changed while being read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_sha256(path: str | os.PathLike[str], *, label: str) -> tuple[str, int]:
    payload = _read_regular(path, label=label, maximum=_MAX_BACKUP_BYTES)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _schema_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type, name, tbl_name"
    ).fetchall()
    stable = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": None if row[3] is None else str(row[3]),
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def _journal_observation(path: str | os.PathLike[str]) -> tuple[int, str]:
    selected = canonical_target_path(path)
    ReadOnlySignedPublicationRetirementJournal(selected)
    uri = f"file:{selected}?mode=ro"
    with sqlite3.connect(uri, uri=True, isolation_level=None) as connection:
        connection.execute("PRAGMA query_only=ON")
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM evidence_graph_set_signed_retirements"
            ).fetchone()[0]
        )
        schema = _schema_digest(connection)
    return count, schema


def _publish_sqlite_backup(
    *,
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
) -> tuple[int, str, int, str, str, int]:
    target = canonical_target_path(target_db_path)
    output = _path(backup_output_path, label="backup_output_path")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{secrets.token_hex(16)}.tmp"
    source_journal = SignedPublicationRetirementJournal(target)
    target_count: int
    target_schema: str
    try:
        with source_journal._lock, source_journal._connect() as source:
            source.execute("BEGIN IMMEDIATE")
            try:
                target_count = int(
                    source.execute(
                        "SELECT COUNT(*) "
                        "FROM evidence_graph_set_signed_retirements"
                    ).fetchone()[0]
                )
                if target_count != 0:
                    raise RuntimeError(
                        "pre-restore backup requires a globally empty target."
                    )
                target_schema = _schema_digest(source)
                with sqlite3.connect(
                    temporary,
                    timeout=30.0,
                    isolation_level=None,
                ) as destination:
                    source.backup(destination)
                source.execute("COMMIT")
            except Exception:
                source.execute("ROLLBACK")
                raise
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    backup_count, backup_schema = _journal_observation(output)
    backup_sha, backup_size = _file_sha256(output, label="backup_artifact")
    if backup_count != target_count or backup_schema != target_schema:
        raise RuntimeError("published backup differs from captured target.")
    return (
        target_count,
        target_schema,
        backup_count,
        backup_schema,
        backup_sha,
        backup_size,
    )


def _receipt_payload(value: Any, *, kind: str) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "receipt_kind": kind,
            "contains_source_text": False,
            "contains_assertion_secrets": False,
            "restore_mutation_performed": False,
            "target_mutation_performed": False,
        }
    )
    return payload


def create_pre_restore_backup_receipt(
    *,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    actor: ReviewActorBinding,
    now: float | None = None,
) -> PreRestoreBackupReceipt:
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    binding = _actor(actor, now=timestamp)
    (
        target_count,
        target_schema,
        backup_count,
        backup_schema,
        backup_sha,
        backup_size,
    ) = _publish_sqlite_backup(
        target_db_path=target_db_path,
        backup_output_path=backup_output_path,
    )
    receipt = PreRestoreBackupReceipt.create(
        owner_id=snapshot.owner_id,
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_path_digest(target_db_path),
        backup_sha256=backup_sha,
        backup_size_bytes=backup_size,
        target_schema_digest=target_schema,
        backup_schema_digest=backup_schema,
        target_record_count=target_count,
        backup_record_count=backup_count,
        actor_id=binding.actor_id,
        binding_method=binding.binding_method,
        binding_digest=binding.binding_digest,
        created_at=timestamp,
    )
    output = _path(receipt_output_path, label="receipt_output_path")
    _atomic_create(
        output,
        _canonical_bytes(
            _receipt_payload(receipt, kind="pre_restore_backup")
        )
        + b"\n",
    )
    return receipt


def _decode_receipt(
    path: str | os.PathLike[str],
    *,
    kind: str,
) -> dict[str, Any]:
    payload = _read_regular(
        path,
        label="receipt_path",
        maximum=_MAX_RECEIPT_BYTES,
    )
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("custody receipt JSON is invalid.") from exc
    safety = {
        "receipt_kind": kind,
        "contains_source_text": False,
        "contains_assertion_secrets": False,
        "restore_mutation_performed": False,
        "target_mutation_performed": False,
    }
    if not isinstance(raw, dict) or any(raw.get(key) != value for key, value in safety.items()):
        raise ValueError("custody receipt safety fields are invalid.")
    for key in safety:
        raw.pop(key, None)
    return raw


def verify_pre_restore_backup_receipt(
    *,
    receipt_path: str | os.PathLike[str],
    backup_path: str | os.PathLike[str],
) -> PreRestoreBackupReceipt:
    raw = _decode_receipt(receipt_path, kind="pre_restore_backup")
    expected = {
        "owner_id",
        "snapshot_digest",
        "target_path_digest",
        "backup_sha256",
        "backup_size_bytes",
        "target_schema_digest",
        "backup_schema_digest",
        "target_record_count",
        "backup_record_count",
        "actor_id",
        "binding_method",
        "binding_digest",
        "created_at",
        "receipt_digest",
        "schema_version",
    }
    if set(raw) != expected:
        raise ValueError("pre-restore receipt schema is invalid.")
    receipt = PreRestoreBackupReceipt(**raw)
    backup_sha, backup_size = _file_sha256(
        backup_path,
        label="backup_artifact",
    )
    count, schema = _journal_observation(backup_path)
    if (
        backup_sha != receipt.backup_sha256
        or backup_size != receipt.backup_size_bytes
        or count != receipt.backup_record_count
        or schema != receipt.backup_schema_digest
    ):
        raise RuntimeError("backup artifact differs from pre-restore receipt.")
    return receipt


def create_post_restore_comparison_receipt(
    *,
    restore_id: str,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    pre_restore_receipt_path: str | os.PathLike[str],
    backup_path: str | os.PathLike[str],
    receipt_output_path: str | os.PathLike[str],
    restore_journal: Any,
    actor: ReviewActorBinding,
    now: float | None = None,
) -> PostRestoreComparisonReceipt:
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    pre = verify_pre_restore_backup_receipt(
        receipt_path=pre_restore_receipt_path,
        backup_path=backup_path,
    )
    selected_restore = _digest(restore_id, "restore_id")
    if not callable(getattr(restore_journal, "get", None)):
        raise ValueError("restore_journal lacks the required read boundary.")
    operation = restore_journal.get(selected_restore)
    target_digest = target_path_digest(target_db_path)
    if (
        operation.state != "completed"
        or operation.phase != "verified"
        or operation.owner_id != snapshot.owner_id
        or operation.snapshot_digest != snapshot.snapshot_digest
        or operation.target_path_digest != target_digest
        or pre.owner_id != snapshot.owner_id
        or pre.snapshot_digest != snapshot.snapshot_digest
        or pre.target_path_digest != target_digest
    ):
        raise RuntimeError("custody inputs differ from completed restore scope.")
    disposition, verification = inspect_restored_target(
        snapshot=snapshot,
        target_db_path=target_db_path,
    )
    if disposition != "exact":
        raise RuntimeError("post-restore target is not exact.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    binding = _actor(actor, now=timestamp)
    receipt = PostRestoreComparisonReceipt.create(
        owner_id=snapshot.owner_id,
        restore_id=selected_restore,
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_digest,
        pre_restore_receipt_digest=pre.receipt_digest,
        backup_sha256=pre.backup_sha256,
        target_verification_digest=verification,
        target_record_count=snapshot.record_count,
        actor_id=binding.actor_id,
        binding_method=binding.binding_method,
        binding_digest=binding.binding_digest,
        compared_at=timestamp,
    )
    output = _path(receipt_output_path, label="receipt_output_path")
    _atomic_create(
        output,
        _canonical_bytes(
            _receipt_payload(receipt, kind="post_restore_comparison")
        )
        + b"\n",
    )
    return receipt


def verify_post_restore_comparison_receipt(
    path: str | os.PathLike[str],
) -> PostRestoreComparisonReceipt:
    raw = _decode_receipt(path, kind="post_restore_comparison")
    expected = {
        "owner_id",
        "restore_id",
        "snapshot_digest",
        "target_path_digest",
        "pre_restore_receipt_digest",
        "backup_sha256",
        "target_verification_digest",
        "target_record_count",
        "actor_id",
        "binding_method",
        "binding_digest",
        "compared_at",
        "receipt_digest",
        "schema_version",
    }
    if set(raw) != expected:
        raise ValueError("post-restore receipt schema is invalid.")
    return PostRestoreComparisonReceipt(**raw)


__all__ = [
    "create_post_restore_comparison_receipt",
    "create_pre_restore_backup_receipt",
    "verify_post_restore_comparison_receipt",
    "verify_pre_restore_backup_receipt",
]
