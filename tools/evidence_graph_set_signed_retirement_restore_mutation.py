"""Atomic empty-target restore mutation for signed retirement snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_readonly import (
    ReadOnlySignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    SignedRetirementSnapshot,
)

_TERMINAL_STATES = frozenset({"completed", "cancelled"})


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


def canonical_target_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("target database path must be a filesystem path.")
    rendered = os.fspath(path)
    if not isinstance(rendered, str) or not rendered or len(rendered) > 4096:
        raise ValueError("target database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    selected = Path(os.path.abspath(candidate))
    if not selected.exists() or not selected.is_file():
        raise ValueError("target retirement database must already exist.")
    for part in (selected, *selected.parents):
        try:
            if part.is_symlink():
                raise ValueError(
                    "target retirement database path may not contain redirects."
                )
        except OSError as exc:
            raise ValueError(
                "target retirement database path could not be validated."
            ) from exc
    ReadOnlySignedPublicationRetirementJournal(selected)
    return selected


def target_path_digest(path: str | os.PathLike[str]) -> str:
    selected = canonical_target_path(path)
    return hashlib.sha256(str(selected).encode("utf-8")).hexdigest()


def validate_terminal_snapshot(snapshot: SignedRetirementSnapshot) -> None:
    if not isinstance(snapshot, SignedRetirementSnapshot):
        raise ValueError("snapshot must be SignedRetirementSnapshot.")
    if snapshot.record_count <= 0:
        raise ValueError("empty snapshots may not be restored.")
    if any(value.owner_id != snapshot.owner_id for value in snapshot.records):
        raise RuntimeError("snapshot record escaped owner scope.")
    nonterminal = tuple(
        value.retirement_id
        for value in snapshot.records
        if value.state not in _TERMINAL_STATES
    )
    if nonterminal:
        raise RuntimeError(
            "snapshot contains executable or retryable retirement work."
        )


def _row_values(value: Any) -> tuple[Any, ...]:
    return (
        value.retirement_id,
        value.owner_id,
        value.publication_operation_id,
        value.graph_set_key,
        value.signed_candidate_set_id,
        value.signed_candidate_set_digest,
        value.authorization_candidate_set_id,
        value.signed_authority_digest,
        value.state,
        value.phase,
        value.attempt_count,
        value.max_attempts,
        value.lease_owner,
        value.lease_expires_at,
        value.final_pointer_set_id,
        value.verification_digest,
        value.failure_type,
        value.created_at,
        value.updated_at,
        value.completed_at,
        value.schema_version,
    )


def _load_all(
    connection: Any,
    journal: SignedPublicationRetirementJournal,
) -> tuple[Any, ...]:
    rows = connection.execute(
        "SELECT * FROM evidence_graph_set_signed_retirements "
        "ORDER BY retirement_id"
    ).fetchall()
    return tuple(journal._attempt(row) for row in rows)


def _exact(
    snapshot: SignedRetirementSnapshot,
    target_values: tuple[Any, ...],
) -> bool:
    source = tuple(
        sorted(snapshot.records, key=lambda value: value.retirement_id)
    )
    target = tuple(
        sorted(target_values, key=lambda value: value.retirement_id)
    )
    return source == target


def _verification_digest(
    snapshot: SignedRetirementSnapshot,
    *,
    target_digest: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-signed-retirement-empty-target-restore-result-v1",
            "owner_id": snapshot.owner_id,
            "snapshot_digest": snapshot.snapshot_digest,
            "target_path_digest": target_digest,
            "record_count": snapshot.record_count,
            "retirement_ids": sorted(
                value.retirement_id for value in snapshot.records
            ),
        }
    )


def inspect_restored_target(
    *,
    snapshot: SignedRetirementSnapshot,
    target_db_path: str | os.PathLike[str],
) -> tuple[str, str]:
    """Return ``empty`` or ``exact`` and a deterministic verification digest."""

    validate_terminal_snapshot(snapshot)
    selected = canonical_target_path(target_db_path)
    journal = SignedPublicationRetirementJournal(selected)
    with journal._lock, journal._connect() as connection:
        values = _load_all(connection, journal)
    if not values:
        disposition = "empty"
    elif _exact(snapshot, values):
        disposition = "exact"
    else:
        raise RuntimeError(
            "target retirement database is neither empty nor exact."
        )
    return disposition, _verification_digest(
        snapshot,
        target_digest=target_path_digest(selected),
    )


def restore_snapshot_into_empty_target(
    *,
    snapshot: SignedRetirementSnapshot,
    target_db_path: str | os.PathLike[str],
) -> tuple[str, bool]:
    """Atomically insert every terminal record or accept an exact replay."""

    validate_terminal_snapshot(snapshot)
    selected = canonical_target_path(target_db_path)
    journal = SignedPublicationRetirementJournal(selected)
    inserted = False
    with journal._lock, journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = _load_all(connection, journal)
            if existing:
                if not _exact(snapshot, existing):
                    raise RuntimeError(
                        "target retirement database is not empty and exact."
                    )
            else:
                for value in sorted(
                    snapshot.records,
                    key=lambda item: item.retirement_id,
                ):
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_retirements "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        _row_values(value),
                    )
                inserted = True
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    disposition, verification = inspect_restored_target(
        snapshot=snapshot,
        target_db_path=selected,
    )
    if disposition != "exact":
        raise RuntimeError("target retirement restore verification failed.")
    return verification, inserted


def complete_with_exact_target_lock(
    *,
    snapshot: SignedRetirementSnapshot,
    target_db_path: str | os.PathLike[str],
    complete: Any,
) -> tuple[Any, str]:
    """Complete the intent while a write lock protects exact target history."""

    if not callable(complete):
        raise ValueError("complete must be callable.")
    validate_terminal_snapshot(snapshot)
    selected = canonical_target_path(target_db_path)
    journal = SignedPublicationRetirementJournal(selected)
    verification = _verification_digest(
        snapshot,
        target_digest=target_path_digest(selected),
    )
    with journal._lock, journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            values = _load_all(connection, journal)
            if not _exact(snapshot, values):
                raise RuntimeError(
                    "restored target no longer matches snapshot."
                )
            result = complete(verification)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return result, verification


__all__ = [
    "canonical_target_path",
    "complete_with_exact_target_lock",
    "inspect_restored_target",
    "restore_snapshot_into_empty_target",
    "target_path_digest",
    "validate_terminal_snapshot",
]
