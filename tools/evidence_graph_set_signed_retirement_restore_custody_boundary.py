"""Canonical two-connection SQLite backup boundary for restore custody."""

from __future__ import annotations

import os
import secrets
import sqlite3
import stat

from tools import evidence_graph_set_signed_retirement_restore_custody as _base
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    canonical_target_path,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _redirecting,
    _path,
)


def _publish_sqlite_backup(
    *,
    target_db_path: str | os.PathLike[str],
    backup_output_path: str | os.PathLike[str],
) -> tuple[int, str, int, str, str, int]:
    """Hold a write-reservation guard while a separate read connection backs up."""

    target = canonical_target_path(target_db_path)
    output = _path(backup_output_path, label="backup_output_path")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parent_info = output.parent.lstat()
    if _redirecting(parent_info) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("backup parent must be a non-redirecting directory.")
    temporary = output.parent / f".{output.name}.{secrets.token_hex(16)}.tmp"
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    create_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, create_flags, 0o600)
    os.close(descriptor)
    target_count: int
    target_schema: str
    source_journal = SignedPublicationRetirementJournal(target)
    try:
        with source_journal._lock, source_journal._connect() as guard:
            guard.execute("BEGIN IMMEDIATE")
            try:
                target_count = int(
                    guard.execute(
                        "SELECT COUNT(*) "
                        "FROM evidence_graph_set_signed_retirements"
                    ).fetchone()[0]
                )
                if target_count != 0:
                    raise RuntimeError(
                        "pre-restore backup requires a globally empty target."
                    )
                target_schema = _base._schema_digest(guard)
                source_uri = f"file:{target}?mode=ro"
                with sqlite3.connect(
                    source_uri,
                    uri=True,
                    timeout=30.0,
                    isolation_level=None,
                ) as source:
                    source.execute("PRAGMA query_only=ON")
                    with sqlite3.connect(
                        temporary,
                        timeout=30.0,
                        isolation_level=None,
                    ) as destination:
                        source.backup(destination)
                guard.execute("COMMIT")
            except Exception:
                guard.execute("ROLLBACK")
                raise
        descriptor = os.open(
            temporary,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
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
    backup_count, backup_schema = _base._journal_observation(output)
    backup_sha, backup_size = _base._file_sha256(
        output,
        label="backup_artifact",
    )
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


_base._publish_sqlite_backup = _publish_sqlite_backup


def create_pre_restore_backup_receipt(**kwargs):
    backup = _path(
        kwargs["backup_output_path"],
        label="backup_output_path",
    )
    receipt = _path(
        kwargs["receipt_output_path"],
        label="receipt_output_path",
    )
    if backup == receipt:
        raise ValueError("backup and receipt outputs must be distinct.")
    if backup.exists():
        raise FileExistsError(backup)
    if receipt.exists():
        raise FileExistsError(receipt)
    return _base.create_pre_restore_backup_receipt(**kwargs)


def create_post_restore_comparison_receipt(**kwargs):
    output = _path(
        kwargs["receipt_output_path"],
        label="receipt_output_path",
    )
    if output.exists():
        raise FileExistsError(output)
    return _base.create_post_restore_comparison_receipt(**kwargs)


verify_post_restore_comparison_receipt = (
    _base.verify_post_restore_comparison_receipt
)
verify_pre_restore_backup_receipt = _base.verify_pre_restore_backup_receipt


__all__ = [
    "create_post_restore_comparison_receipt",
    "create_pre_restore_backup_receipt",
    "verify_post_restore_comparison_receipt",
    "verify_pre_restore_backup_receipt",
]
