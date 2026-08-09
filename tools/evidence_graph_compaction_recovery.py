"""Durable operator evidence for exact evidence-graph compaction recovery.

Recovery of an interrupted compaction mutates only the durable compaction receipt,
never authoritative generation/vector/sparse state or graph payloads. This journal
adds intent-before-recovery audit evidence, bounded failure evidence, and an
idempotent terminal receipt so crash recovery itself is resumable and auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_compaction import (
    EvidenceGraphCompactionStore,
    _digest,
    _identifier,
    _integer,
    _path,
    _redirecting,
    _timestamp,
)
from tools.evidence_graph_compaction_reconciliation import (
    EvidenceGraphCompactionReconciliationReport,
    EvidenceGraphCompactionRecoveryResult,
    recover_reconciled_compaction_receipts,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_LIMIT = 10_000
_PHASES = frozenset({"planned", "completed"})


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


def _job_ids(values: Iterable[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("job IDs must be an iterable of SHA-256 digests.")
    result = tuple(sorted({_digest(value, "job_id") for value in values}))
    if not result and not allow_empty:
        raise ValueError("at least one recoverable compaction job is required.")
    return result


def _encode_ids(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _decode_ids(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or len(value) > 700_000:
        raise RuntimeError(f"stored {label} is corrupt.")
    try:
        raw = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"stored {label} is corrupt.") from exc
    if not isinstance(raw, list):
        raise RuntimeError(f"stored {label} is corrupt.")
    try:
        return _job_ids(raw, allow_empty=True)
    except ValueError as exc:
        raise RuntimeError(f"stored {label} is corrupt.") from exc


def _result_digest(value: EvidenceGraphCompactionRecoveryResult) -> str:
    payload = asdict(value)
    payload.pop("completed_at", None)
    return _canonical_digest(payload)


@dataclass(frozen=True)
class EvidenceGraphCompactionRecoveryReceipt:
    recovery_id: str
    owner_id: str
    report_digest: str
    confirmed_job_ids: tuple[str, ...]
    actor_id: str
    reason: str
    phase: str
    attempt_count: int
    completed_job_ids: tuple[str, ...]
    already_completed_job_ids: tuple[str, ...]
    result_digest: str | None
    last_error_type: str | None
    created_at: float
    updated_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery_id", _digest(self.recovery_id, "recovery_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "report_digest", _digest(self.report_digest, "report_digest"))
        object.__setattr__(self, "confirmed_job_ids", _job_ids(self.confirmed_job_ids))
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id", 128))
        object.__setattr__(self, "reason", _identifier(self.reason, "reason", 500))
        phase = _identifier(self.phase, "phase", 20)
        if phase not in _PHASES:
            raise ValueError("recovery receipt phase is unsupported.")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "attempt_count",
            _integer(self.attempt_count, "attempt_count", 1, 1_000_000),
        )
        for name in ("completed_job_ids", "already_completed_job_ids"):
            object.__setattr__(self, name, _job_ids(getattr(self, name), allow_empty=True))
        if set(self.completed_job_ids) & set(self.already_completed_job_ids):
            raise ValueError("recovery receipt job result sets must not overlap.")
        result = None if self.result_digest is None else _digest(
            self.result_digest, "result_digest"
        )
        error = None if self.last_error_type is None else _identifier(
            self.last_error_type, "last_error_type", 200
        )
        object.__setattr__(self, "result_digest", result)
        object.__setattr__(self, "last_error_type", error)
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("recovery receipt schema is unsupported.")
        terminal_ids = tuple(sorted(self.completed_job_ids + self.already_completed_job_ids))
        if self.phase == "planned":
            if terminal_ids or self.result_digest is not None:
                raise ValueError("planned recovery intent may not claim completion.")
        elif terminal_ids != self.confirmed_job_ids or self.result_digest is None:
            raise ValueError("completed recovery receipt must account for every confirmed job.")
        if self.phase == "completed" and self.last_error_type is not None:
            raise ValueError("completed recovery receipt may not retain a failure type.")

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "recovery_id": self.recovery_id,
                "owner_id": self.owner_id,
                "report_digest": self.report_digest,
                "confirmed_job_ids": self.confirmed_job_ids,
                "actor_id": self.actor_id,
                "reason": self.reason,
                "schema_version": self.schema_version,
            }
        )

    @property
    def receipt_digest(self) -> str:
        return _canonical_digest(
            {
                "immutable_digest": self.immutable_digest,
                "phase": self.phase,
                "attempt_count": self.attempt_count,
                "completed_job_ids": self.completed_job_ids,
                "already_completed_job_ids": self.already_completed_job_ids,
                "result_digest": self.result_digest,
                "last_error_type": self.last_error_type,
                "schema_version": self.schema_version,
            }
        )


class EvidenceGraphCompactionRecoveryJournal:
    """Append-stable recovery intents and terminal receipts in the compaction DB."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not self.path.parent.is_dir():
            raise ValueError("recovery journal parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not self.path.is_file():
            raise RuntimeError("recovery journal database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not self.path.parent.is_dir()
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("recovery journal parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("recovery journal database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_graph_compaction_recovery (
                    recovery_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    report_digest TEXT NOT NULL,
                    confirmed_job_ids_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    completed_job_ids_json TEXT NOT NULL,
                    already_completed_job_ids_json TEXT NOT NULL,
                    result_digest TEXT,
                    last_error_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_graph_compaction_recovery_owner_phase
                    ON evidence_graph_compaction_recovery(
                        owner_id, phase, updated_at, recovery_id
                    );
                """
            )

    @staticmethod
    def _receipt(row: sqlite3.Row) -> EvidenceGraphCompactionRecoveryReceipt:
        try:
            return EvidenceGraphCompactionRecoveryReceipt(
                recovery_id=row["recovery_id"],
                owner_id=row["owner_id"],
                report_digest=row["report_digest"],
                confirmed_job_ids=_decode_ids(
                    row["confirmed_job_ids_json"], "confirmed_job_ids"
                ),
                actor_id=row["actor_id"],
                reason=row["reason"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                completed_job_ids=_decode_ids(
                    row["completed_job_ids_json"], "completed_job_ids"
                ),
                already_completed_job_ids=_decode_ids(
                    row["already_completed_job_ids_json"], "already_completed_job_ids"
                ),
                result_digest=row["result_digest"],
                last_error_type=row["last_error_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                schema_version=int(row["schema_version"]),
            )
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise RuntimeError("stored recovery receipt is corrupt.") from exc

    def get(self, recovery_id: str) -> EvidenceGraphCompactionRecoveryReceipt | None:
        selected = _digest(recovery_id, "recovery_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_compaction_recovery WHERE recovery_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._receipt(row)

    def list(
        self,
        *,
        owner_id: str,
        phase: str | None = None,
        limit: int = 100,
    ) -> tuple[EvidenceGraphCompactionRecoveryReceipt, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = "SELECT * FROM evidence_graph_compaction_recovery WHERE owner_id=?"
        values: list[Any] = [owner]
        if phase is not None:
            selected_phase = _identifier(phase, "phase", 20)
            if selected_phase not in _PHASES:
                raise ValueError("recovery receipt phase is unsupported.")
            query += " AND phase=?"
            values.append(selected_phase)
        query += " ORDER BY created_at, recovery_id LIMIT ?"
        values.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(values)).fetchall()
        return tuple(self._receipt(row) for row in rows)

    def begin(
        self,
        *,
        report: EvidenceGraphCompactionReconciliationReport,
        confirm_report_digest: str,
        confirm_job_ids: Iterable[str],
        actor_id: str,
        reason: str,
        now: float | None = None,
    ) -> EvidenceGraphCompactionRecoveryReceipt:
        if not isinstance(report, EvidenceGraphCompactionReconciliationReport):
            raise ValueError("report must be EvidenceGraphCompactionReconciliationReport.")
        selected_report = _digest(confirm_report_digest, "confirm_report_digest")
        if selected_report != report.report_digest:
            raise ValueError("confirmation must exactly match report_digest.")
        confirmed = _job_ids(confirm_job_ids)
        if confirmed != report.recoverable_job_ids:
            raise ValueError("job confirmations must exactly match every recoverable finding.")
        actor = _identifier(actor_id, "actor_id", 128)
        selected_reason = _identifier(reason, "reason", 500)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        if timestamp < report.generated_at:
            raise ValueError("recovery time may not precede report generation.")
        recovery_id = _canonical_digest(
            {
                "contract": "rigorousrag-compaction-recovery-v1",
                "owner_id": report.owner_id,
                "report_digest": report.report_digest,
                "confirmed_job_ids": confirmed,
                "actor_id": actor,
                "reason": selected_reason,
            }
        )
        candidate = EvidenceGraphCompactionRecoveryReceipt(
            recovery_id=recovery_id,
            owner_id=report.owner_id,
            report_digest=report.report_digest,
            confirmed_job_ids=confirmed,
            actor_id=actor,
            reason=selected_reason,
            phase="planned",
            attempt_count=1,
            completed_job_ids=(),
            already_completed_job_ids=(),
            result_digest=None,
            last_error_type=None,
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_compaction_recovery WHERE recovery_id=?",
                    (recovery_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_compaction_recovery VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate.recovery_id,
                            candidate.owner_id,
                            candidate.report_digest,
                            _encode_ids(candidate.confirmed_job_ids),
                            candidate.actor_id,
                            candidate.reason,
                            candidate.phase,
                            candidate.attempt_count,
                            _encode_ids(candidate.completed_job_ids),
                            _encode_ids(candidate.already_completed_job_ids),
                            candidate.result_digest,
                            candidate.last_error_type,
                            candidate.created_at,
                            candidate.updated_at,
                            candidate.schema_version,
                        ),
                    )
                else:
                    stored = self._receipt(row)
                    if stored.immutable_digest != candidate.immutable_digest:
                        raise RuntimeError("compaction recovery identity collision detected.")
                    if stored.phase == "completed":
                        connection.execute("COMMIT")
                        return stored
                    if timestamp < stored.created_at:
                        raise ValueError("recovery retry may not precede intent creation.")
                    connection.execute(
                        """
                        UPDATE evidence_graph_compaction_recovery
                        SET attempt_count=attempt_count+1, updated_at=?, last_error_type=NULL
                        WHERE recovery_id=? AND phase='planned'
                        """,
                        (timestamp, recovery_id),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(recovery_id)
        if result is None:
            raise RuntimeError("compaction recovery intent disappeared.")
        return result

    def record_failure(
        self,
        recovery_id: str,
        *,
        failure_type: str,
        now: float | None = None,
    ) -> EvidenceGraphCompactionRecoveryReceipt:
        selected = _digest(recovery_id, "recovery_id")
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        stored = self.get(selected)
        if stored is None or stored.phase != "planned":
            raise RuntimeError("planned compaction recovery intent is unavailable.")
        if timestamp < stored.created_at:
            raise ValueError("failure time may not precede intent creation.")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_graph_compaction_recovery
                SET last_error_type=?, updated_at=?
                WHERE recovery_id=? AND phase='planned'
                """,
                (failure, timestamp, selected),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("planned compaction recovery intent is unavailable.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("compaction recovery intent disappeared after failure.")
        return result

    def complete(
        self,
        recovery_id: str,
        *,
        result: EvidenceGraphCompactionRecoveryResult,
        now: float | None = None,
    ) -> EvidenceGraphCompactionRecoveryReceipt:
        selected = _digest(recovery_id, "recovery_id")
        if not isinstance(result, EvidenceGraphCompactionRecoveryResult):
            raise ValueError("result must be EvidenceGraphCompactionRecoveryResult.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_compaction_recovery WHERE recovery_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("compaction recovery intent is unavailable.")
                stored = self._receipt(row)
                if stored.phase == "completed":
                    connection.execute("COMMIT")
                    return stored
                if timestamp < stored.created_at:
                    raise ValueError("completion time may not precede intent creation.")
                if result.owner_id != stored.owner_id or result.report_digest != stored.report_digest:
                    raise RuntimeError("compaction recovery result identity changed.")
                accounted = tuple(
                    sorted(result.completed_job_ids + result.already_completed_job_ids)
                )
                if accounted != stored.confirmed_job_ids:
                    raise RuntimeError("compaction recovery result is incomplete.")
                connection.execute(
                    """
                    UPDATE evidence_graph_compaction_recovery
                    SET phase='completed', completed_job_ids_json=?,
                        already_completed_job_ids_json=?, result_digest=?,
                        last_error_type=NULL, updated_at=?
                    WHERE recovery_id=? AND phase='planned'
                    """,
                    (
                        _encode_ids(result.completed_job_ids),
                        _encode_ids(result.already_completed_job_ids),
                        _result_digest(result),
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        receipt = self.get(selected)
        if receipt is None or receipt.phase != "completed":
            raise RuntimeError("compaction recovery completion was not persisted.")
        return receipt


def recover_reconciled_compaction_receipts_durable(
    *,
    report: EvidenceGraphCompactionReconciliationReport,
    compactions: EvidenceGraphCompactionStore,
    journal: Any,
    generations: Any,
    graphs: Any,
    recovery_journal: EvidenceGraphCompactionRecoveryJournal,
    confirm_report_digest: str,
    confirm_job_ids: Iterable[str],
    actor_id: str,
    reason: str,
    now: float | None = None,
) -> EvidenceGraphCompactionRecoveryReceipt:
    """Recover exact receipt states with durable intent-before-mutation evidence."""

    if not isinstance(compactions, EvidenceGraphCompactionStore):
        raise ValueError("compactions must be EvidenceGraphCompactionStore.")
    if not isinstance(recovery_journal, EvidenceGraphCompactionRecoveryJournal):
        raise ValueError("recovery_journal must be EvidenceGraphCompactionRecoveryJournal.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    intent = recovery_journal.begin(
        report=report,
        confirm_report_digest=confirm_report_digest,
        confirm_job_ids=confirm_job_ids,
        actor_id=actor_id,
        reason=reason,
        now=timestamp,
    )
    if intent.phase == "completed":
        return intent
    try:
        result = recover_reconciled_compaction_receipts(
            report=report,
            compactions=compactions,
            journal=journal,
            generations=generations,
            graphs=graphs,
            confirm_report_digest=confirm_report_digest,
            confirm_job_ids=intent.confirmed_job_ids,
            now=timestamp,
        )
    except Exception as exc:
        recovery_journal.record_failure(
            intent.recovery_id,
            failure_type=type(exc).__name__,
            now=timestamp,
        )
        raise
    return recovery_journal.complete(intent.recovery_id, result=result, now=timestamp)


__all__ = [
    "EvidenceGraphCompactionRecoveryJournal",
    "EvidenceGraphCompactionRecoveryReceipt",
    "recover_reconciled_compaction_receipts_durable",
]
