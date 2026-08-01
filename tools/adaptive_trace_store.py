"""Privacy-safe SQLite diagnostics for adaptive retrieval.

Only query hashes, route controls, aggregate signals, generic error class names,
and bounded timing/cost counters are persisted. Raw queries, evidence, paths,
provider payloads, and exception messages are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.adaptive_retrieval_runner import AdaptiveRetrievalResult
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_DECISIONS = {"empty", "insufficient", "weak", "sufficient"}
_MAX_ATTEMPTS = 6


def _redirected(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _check_path(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("Adaptive trace path could not be validated.") from exc
        if _redirected(metadata):
            raise ValueError(
                "Adaptive trace path may not contain symbolic links or reparse points."
            )


def _identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if _redirected(metadata):
        raise RuntimeError("Adaptive trace path was redirected.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is outside its allowed range.")
    return parsed


def _query_hash(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    if (
        not query.strip()
        or len(query) > 20_000
        or any(
            (ord(character) < 32 and character not in "\t\r\n")
            or ord(character) == 127
            for character in query
        )
    ):
        raise ValueError("query is empty, invalid, or too long.")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdaptiveTraceSummary:
    run_id: str
    owner_id: str
    query_sha256: str
    started_at: float
    completed_at: float
    attempt_count: int
    evidence_count: int
    final_decision: str
    final_sufficiency: float
    abstain: bool
    exhausted: bool
    estimated_cost: int
    error_count: int


@dataclass(frozen=True)
class AdaptiveTraceAttempt:
    sequence: int
    mode: str
    top_k: int
    candidate_pool: int
    use_multi_query: bool
    use_hyde: bool
    reranker: str
    reason: str
    returned_evidence: int
    accumulated_evidence: int
    decision: str
    sufficiency: float
    error_type: str | None


@dataclass(frozen=True)
class AdaptiveTraceRecord:
    summary: AdaptiveTraceSummary
    attempts: tuple[AdaptiveTraceAttempt, ...]


@dataclass(frozen=True)
class AdaptiveTraceAggregate:
    run_count: int
    abstention_count: int
    exhausted_count: int
    error_run_count: int
    mean_sufficiency: float
    mean_estimated_cost: float
    route_attempts: tuple[tuple[str, int], ...]
    decisions: tuple[tuple[str, int], ...]


class AdaptiveTraceStore:
    """Owner-scoped, identity-bound trace storage for one host."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("Adaptive trace path must be a filesystem path.")
        rendered = os.fspath(path)
        if (
            not isinstance(rendered, str)
            or not rendered
            or len(rendered) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
        ):
            raise ValueError("Adaptive trace path is invalid.")
        candidate = Path(rendered)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self.path = Path(os.path.abspath(candidate))
        _check_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _check_path(self.path)
        self._lock = threading.RLock()
        self._initialize()
        self._parent_identity = _identity(self.path.parent)
        self._database_identity = _identity(self.path)

    def _initialize(self) -> None:
        with sqlite3.connect(str(self.path), timeout=30.0) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS adaptive_trace_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS adaptive_runs (
                    run_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    query_sha256 TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    final_decision TEXT NOT NULL,
                    final_sufficiency REAL NOT NULL,
                    abstain INTEGER NOT NULL,
                    exhausted INTEGER NOT NULL,
                    estimated_cost INTEGER NOT NULL,
                    error_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS adaptive_attempts (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    candidate_pool INTEGER NOT NULL,
                    use_multi_query INTEGER NOT NULL,
                    use_hyde INTEGER NOT NULL,
                    reranker TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    returned_evidence INTEGER NOT NULL,
                    accumulated_evidence INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    sufficiency REAL NOT NULL,
                    error_type TEXT,
                    PRIMARY KEY(run_id, sequence),
                    FOREIGN KEY(run_id) REFERENCES adaptive_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS adaptive_runs_owner_completed
                    ON adaptive_runs(owner_id, completed_at DESC, run_id DESC);
                """
            )
            row = connection.execute(
                "SELECT schema_version FROM adaptive_trace_schema WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO adaptive_trace_schema VALUES(1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise RuntimeError("Adaptive trace schema version is incompatible.")

    def _verify_identity(self) -> None:
        _check_path(self.path)
        try:
            parent = _identity(self.path.parent)
            database = _identity(self.path)
        except FileNotFoundError as exc:
            raise RuntimeError("Adaptive trace database disappeared.") from exc
        if parent != self._parent_identity or database != self._database_identity:
            raise RuntimeError("Adaptive trace database or parent was replaced.")

    def _connect(self) -> sqlite3.Connection:
        self._verify_identity()
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _summary(row: sqlite3.Row) -> AdaptiveTraceSummary:
        decision = _identifier(row["final_decision"], "final_decision", 50)
        if decision not in _DECISIONS:
            raise RuntimeError("Adaptive trace decision is corrupt.")
        query_hash = _identifier(row["query_sha256"], "query_sha256", 64).lower()
        if len(query_hash) != 64 or any(character not in "0123456789abcdef" for character in query_hash):
            raise RuntimeError("Adaptive trace query fingerprint is corrupt.")
        abstain, exhausted = row["abstain"], row["exhausted"]
        if abstain not in (0, 1) or exhausted not in (0, 1):
            raise RuntimeError("Adaptive trace boolean state is corrupt.")
        return AdaptiveTraceSummary(
            run_id=_identifier(row["run_id"], "run_id"),
            owner_id=normalize_owner_id(row["owner_id"]),
            query_sha256=query_hash,
            started_at=_number(row["started_at"], "started_at", 0.0, 10**15),
            completed_at=_number(row["completed_at"], "completed_at", 0.0, 10**15),
            attempt_count=_integer(row["attempt_count"], "attempt_count", 0, _MAX_ATTEMPTS),
            evidence_count=_integer(row["evidence_count"], "evidence_count", 0, 100),
            final_decision=decision,
            final_sufficiency=_number(row["final_sufficiency"], "final_sufficiency", 0.0, 1.0),
            abstain=bool(abstain),
            exhausted=bool(exhausted),
            estimated_cost=_integer(row["estimated_cost"], "estimated_cost", 0, 100_000),
            error_count=_integer(row["error_count"], "error_count", 0, _MAX_ATTEMPTS),
        )

    @staticmethod
    def _attempts(connection: sqlite3.Connection, run_id: str) -> tuple[AdaptiveTraceAttempt, ...]:
        rows = connection.execute(
            "SELECT * FROM adaptive_attempts WHERE run_id=? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        result: list[AdaptiveTraceAttempt] = []
        for row in rows:
            decision = _identifier(row["decision"], "decision", 50)
            if decision not in _DECISIONS:
                raise RuntimeError("Adaptive attempt decision is corrupt.")
            multi, hyde = row["use_multi_query"], row["use_hyde"]
            if multi not in (0, 1) or hyde not in (0, 1):
                raise RuntimeError("Adaptive attempt boolean state is corrupt.")
            error_type = row["error_type"]
            result.append(
                AdaptiveTraceAttempt(
                    sequence=_integer(row["sequence"], "sequence", 0, _MAX_ATTEMPTS - 1),
                    mode=_identifier(row["mode"], "mode", 100),
                    top_k=_integer(row["top_k"], "top_k", 1, 50),
                    candidate_pool=_integer(row["candidate_pool"], "candidate_pool", 1, 50),
                    use_multi_query=bool(multi),
                    use_hyde=bool(hyde),
                    reranker=_identifier(row["reranker"], "reranker", 100),
                    reason=_identifier(row["reason"], "reason"),
                    returned_evidence=_integer(row["returned_evidence"], "returned_evidence", 0, 100),
                    accumulated_evidence=_integer(row["accumulated_evidence"], "accumulated_evidence", 0, 100),
                    decision=decision,
                    sufficiency=_number(row["sufficiency"], "sufficiency", 0.0, 1.0),
                    error_type=None if error_type is None else _identifier(error_type, "error_type"),
                )
            )
        if tuple(item.sequence for item in result) != tuple(range(len(result))):
            raise RuntimeError("Adaptive attempt sequence is corrupt.")
        return tuple(result)

    def record_result(
        self,
        *,
        query: str,
        owner_id: str,
        result: AdaptiveRetrievalResult,
        run_id: str | None = None,
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> str:
        if not isinstance(result, AdaptiveRetrievalResult):
            raise ValueError("result must be an AdaptiveRetrievalResult.")
        owner = normalize_owner_id(owner_id)
        identifier = _identifier(run_id or uuid.uuid4().hex, "run_id")
        query_hash = _query_hash(query)
        started = _number(time.time() if started_at is None else started_at, "started_at", 0.0, 10**15)
        completed = _number(time.time() if completed_at is None else completed_at, "completed_at", 0.0, 10**15)
        if completed < started:
            raise ValueError("completed_at may not precede started_at.")
        attempts = tuple(result.traces)
        if len(attempts) > _MAX_ATTEMPTS:
            raise ValueError("adaptive trace attempt limit exceeded.")
        errors = sum(trace.error_type is not None for trace in attempts)
        values = (
            identifier, owner, query_hash, started, completed, len(attempts),
            len(result.evidence), result.final_signals.decision,
            result.final_signals.sufficiency, int(result.abstain), int(result.exhausted),
            result.estimated_cost, errors,
        )
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM adaptive_runs WHERE run_id=?", (identifier,)
            ).fetchone()
            if existing is not None:
                if self._summary(existing) == AdaptiveTraceSummary(*values):
                    return identifier
                raise ValueError("run_id already identifies a different adaptive trace.")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO adaptive_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", values
            )
            connection.executemany(
                "INSERT INTO adaptive_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        identifier, sequence, trace.attempt.mode, trace.attempt.top_k,
                        trace.attempt.candidate_pool, int(trace.attempt.use_multi_query),
                        int(trace.attempt.use_hyde), trace.attempt.reranker,
                        trace.attempt.reason, trace.returned_evidence,
                        trace.accumulated_evidence, trace.signals.decision,
                        trace.signals.sufficiency, trace.error_type,
                    )
                    for sequence, trace in enumerate(attempts)
                ],
            )
            connection.commit()
        self._verify_identity()
        return identifier

    def get_run(self, *, owner_id: str, run_id: str) -> AdaptiveTraceRecord | None:
        owner = normalize_owner_id(owner_id)
        identifier = _identifier(run_id, "run_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM adaptive_runs WHERE owner_id=? AND run_id=?",
                (owner, identifier),
            ).fetchone()
            if row is None:
                return None
            summary = self._summary(row)
            attempts = self._attempts(connection, identifier)
            if len(attempts) != summary.attempt_count:
                raise RuntimeError("Adaptive trace attempt count is corrupt.")
            result = AdaptiveTraceRecord(summary, attempts)
        self._verify_identity()
        return result

    def list_runs(self, *, owner_id: str, limit: int = 100) -> tuple[AdaptiveTraceSummary, ...]:
        owner = normalize_owner_id(owner_id)
        bounded = _integer(limit, "limit", 1, 1_000)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_runs WHERE owner_id=? "
                "ORDER BY completed_at DESC, run_id DESC LIMIT ?",
                (owner, bounded),
            ).fetchall()
            result = tuple(self._summary(row) for row in rows)
        self._verify_identity()
        return result

    def aggregate(self, *, owner_id: str, limit: int = 1_000) -> AdaptiveTraceAggregate:
        summaries = self.list_runs(owner_id=owner_id, limit=limit)
        if not summaries:
            return AdaptiveTraceAggregate(0, 0, 0, 0, 0.0, 0.0, (), ())
        run_ids = [summary.run_id for summary in summaries]
        placeholders = ",".join("?" for _ in run_ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT mode, COUNT(*) count FROM adaptive_attempts "
                f"WHERE run_id IN ({placeholders}) GROUP BY mode ORDER BY mode",
                run_ids,
            ).fetchall()
        decisions: dict[str, int] = {}
        for summary in summaries:
            decisions[summary.final_decision] = decisions.get(summary.final_decision, 0) + 1
        return AdaptiveTraceAggregate(
            run_count=len(summaries),
            abstention_count=sum(summary.abstain for summary in summaries),
            exhausted_count=sum(summary.exhausted for summary in summaries),
            error_run_count=sum(summary.error_count > 0 for summary in summaries),
            mean_sufficiency=round(sum(s.final_sufficiency for s in summaries) / len(summaries), 9),
            mean_estimated_cost=round(sum(s.estimated_cost for s in summaries) / len(summaries), 9),
            route_attempts=tuple((row["mode"], int(row["count"])) for row in rows),
            decisions=tuple(sorted(decisions.items())),
        )

    def prune_owner(self, *, owner_id: str, retain_latest: int = 10_000) -> int:
        owner = normalize_owner_id(owner_id)
        retain = _integer(retain_latest, "retain_latest", 0, 1_000_000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM adaptive_runs WHERE owner_id=? AND run_id IN ("
                "SELECT run_id FROM adaptive_runs WHERE owner_id=? "
                "ORDER BY completed_at DESC, run_id DESC LIMIT -1 OFFSET ?)",
                (owner, owner, retain),
            )
            connection.commit()
            deleted = max(int(cursor.rowcount), 0)
        self._verify_identity()
        return deleted

    def ping(self) -> bool:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT schema_version FROM adaptive_trace_schema WHERE singleton=1"
                ).fetchone()
                return row is not None and int(row[0]) == _SCHEMA_VERSION
        except Exception:
            return False


__all__ = [
    "AdaptiveTraceAggregate",
    "AdaptiveTraceAttempt",
    "AdaptiveTraceRecord",
    "AdaptiveTraceStore",
    "AdaptiveTraceSummary",
]
