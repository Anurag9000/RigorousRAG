"""Durable exact-confirmation compaction for stale evidence-graph payloads.

Compaction removes only verified non-current graph generations. The smaller
job journal rows remain available as an audit trail. Every deletion is preceded
by a durable intent record so a crash between graph deletion and receipt
completion can be resumed without treating an intentional deletion as
unexplained corruption.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.evidence_graph_jobs import EvidenceGraphJob
from tools.evidence_graph_operations import (
    EvidenceGraphRetentionCandidate,
    EvidenceGraphRetentionPlan,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_SCHEMA_VERSION = 1
_PHASES = frozenset({"planned", "completed"})
_ACTIONS = frozenset({"delete_graph_generation", "retain_job_audit_only"})


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("evidence graph compaction database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("evidence graph compaction database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                "evidence graph compaction database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError(
                "evidence graph compaction database path may not contain redirects."
            )
    return absolute


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


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


def _job_matches_current(job: EvidenceGraphJob, generation: Any | None) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "owner_id", None) == job.owner_id
        and getattr(generation, "doc_id", None) == job.doc_id
        and getattr(generation, "sequence", None) == job.source_sequence
        and getattr(generation, "state", None) == job.source_state
        and getattr(generation, "content_sha256", None) == job.content_sha256
        and getattr(generation, "profile_fingerprint", None)
        == job.profile_fingerprint
        and getattr(generation, "sparse_generation", None)
        == job.sparse_generation
    )


def _confirmed_job_ids(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(sorted({_digest(value, "confirm_job_id") for value in values}))
    if not result:
        raise ValueError("at least one exact job confirmation is required.")
    return result


@dataclass(frozen=True)
class EvidenceGraphCompactionRecord:
    job_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    job_state: str
    graph_digest: str | None
    action: str
    plan_digest: str
    phase: str
    created_at: float
    updated_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _digest(self.job_id, "job_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            _integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        if self.job_state not in {"completed", "cancelled"}:
            raise ValueError("compaction records require completed or cancelled jobs.")
        graph_digest = _optional_digest(self.graph_digest, "graph_digest")
        action = _identifier(self.action, "action", 50)
        if action not in _ACTIONS:
            raise ValueError("compaction action is unsupported.")
        if self.job_state == "completed":
            if graph_digest is None or action != "delete_graph_generation":
                raise ValueError("completed jobs require exact graph-generation deletion.")
        elif graph_digest is not None or action != "retain_job_audit_only":
            raise ValueError("cancelled jobs retain only their audit row.")
        object.__setattr__(self, "graph_digest", graph_digest)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "plan_digest", _digest(self.plan_digest, "plan_digest"))
        phase = _identifier(self.phase, "phase", 20)
        if phase not in _PHASES:
            raise ValueError("compaction phase is unsupported.")
        object.__setattr__(self, "phase", phase)
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("compaction record schema is unsupported.")

    @property
    def immutable_digest(self) -> str:
        value = asdict(self)
        for key in ("phase", "created_at", "updated_at"):
            value.pop(key, None)
        return _canonical_digest(value)


@dataclass(frozen=True)
class EvidenceGraphCompactionResult:
    owner_id: str
    plan_digest: str
    candidate_count: int
    completed_job_ids: tuple[str, ...]
    already_completed_job_ids: tuple[str, ...]
    deleted_graph_generation_job_ids: tuple[str, ...]
    retained_job_audit_only_ids: tuple[str, ...]
    completed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "plan_digest", _digest(self.plan_digest, "plan_digest"))
        object.__setattr__(
            self,
            "candidate_count",
            _integer(self.candidate_count, "candidate_count", 0, _MAX_LIMIT),
        )
        for name in (
            "completed_job_ids",
            "already_completed_job_ids",
            "deleted_graph_generation_job_ids",
            "retained_job_audit_only_ids",
        ):
            values = tuple(_digest(value, name) for value in getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and sorted.")
            object.__setattr__(self, name, values)
        if len(self.completed_job_ids) + len(self.already_completed_job_ids) != self.candidate_count:
            raise ValueError("every compaction candidate must have a terminal result.")
        if not set(self.deleted_graph_generation_job_ids).issubset(self.completed_job_ids):
            raise ValueError("deleted graph generations must be newly completed items.")
        if not set(self.retained_job_audit_only_ids).issubset(self.completed_job_ids):
            raise ValueError("audit-only items must be newly completed items.")
        object.__setattr__(self, "completed_at", _timestamp(self.completed_at, "completed_at"))

    @property
    def result_digest(self) -> str:
        value = asdict(self)
        value.pop("completed_at", None)
        return _canonical_digest(value)


class EvidenceGraphCompactionStore:
    """Append-only compaction intent/receipt records keyed by graph job ID."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("evidence graph compaction parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("evidence graph compaction database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("evidence graph compaction parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("evidence graph compaction database identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_compaction (
                    job_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL,
                    job_state TEXT NOT NULL,
                    graph_digest TEXT,
                    action TEXT NOT NULL,
                    plan_digest TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_graph_compaction_owner_phase
                    ON evidence_graph_compaction(owner_id, phase, updated_at, job_id);
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> EvidenceGraphCompactionRecord:
        return EvidenceGraphCompactionRecord(
            job_id=row["job_id"],
            owner_id=row["owner_id"],
            doc_id=row["doc_id"],
            source_sequence=int(row["source_sequence"]),
            job_state=row["job_state"],
            graph_digest=row["graph_digest"],
            action=row["action"],
            plan_digest=row["plan_digest"],
            phase=row["phase"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            schema_version=int(row["schema_version"]),
        )

    def get(self, job_id: str) -> EvidenceGraphCompactionRecord | None:
        selected = _digest(job_id, "job_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_compaction WHERE job_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._record(row)

    def list(
        self,
        *,
        owner_id: str,
        phase: str | None = None,
        limit: int = 100,
    ) -> tuple[EvidenceGraphCompactionRecord, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        clauses = ["owner_id=?"]
        values: list[Any] = [owner]
        if phase is not None:
            selected_phase = _identifier(phase, "phase", 20)
            if selected_phase not in _PHASES:
                raise ValueError("compaction phase is unsupported.")
            clauses.append("phase=?")
            values.append(selected_phase)
        values.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence_graph_compaction WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, job_id LIMIT ?",
                values,
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def begin(
        self,
        *,
        job: EvidenceGraphJob,
        plan_digest: str,
        now: float | None = None,
    ) -> EvidenceGraphCompactionRecord:
        if not isinstance(job, EvidenceGraphJob):
            raise ValueError("job must be EvidenceGraphJob.")
        if job.state not in {"completed", "cancelled"}:
            raise ValueError("only terminal retention candidates may be compacted.")
        selected_plan = _digest(plan_digest, "plan_digest")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        action = (
            "delete_graph_generation"
            if job.state == "completed"
            else "retain_job_audit_only"
        )
        candidate = EvidenceGraphCompactionRecord(
            job_id=job.job_id,
            owner_id=job.owner_id,
            doc_id=job.doc_id,
            source_sequence=job.source_sequence,
            job_state=job.state,
            graph_digest=job.graph_digest,
            action=action,
            plan_digest=selected_plan,
            phase="planned",
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_compaction WHERE job_id=?",
                    (job.job_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_compaction VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate.job_id,
                            candidate.owner_id,
                            candidate.doc_id,
                            candidate.source_sequence,
                            candidate.job_state,
                            candidate.graph_digest,
                            candidate.action,
                            candidate.plan_digest,
                            candidate.phase,
                            candidate.created_at,
                            candidate.updated_at,
                            candidate.schema_version,
                        ),
                    )
                else:
                    stored = self._record(row)
                    if stored.immutable_digest != candidate.immutable_digest:
                        if stored.phase == "completed" and (
                            stored.job_id == candidate.job_id
                            and stored.owner_id == candidate.owner_id
                            and stored.doc_id == candidate.doc_id
                            and stored.source_sequence == candidate.source_sequence
                            and stored.job_state == candidate.job_state
                            and stored.graph_digest == candidate.graph_digest
                            and stored.action == candidate.action
                        ):
                            connection.execute("COMMIT")
                            return stored
                        raise RuntimeError("compaction job identity collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(job.job_id)
        if result is None:
            raise RuntimeError("compaction intent disappeared.")
        return result

    def complete(
        self,
        job_id: str,
        *,
        owner_id: str,
        plan_digest: str,
        now: float | None = None,
    ) -> EvidenceGraphCompactionRecord:
        selected = _digest(job_id, "job_id")
        owner = normalize_owner_id(owner_id)
        selected_plan = _digest(plan_digest, "plan_digest")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_compaction WHERE job_id=? AND owner_id=?",
                    (selected, owner),
                ).fetchone()
                if row is None:
                    raise RuntimeError("compaction intent is unavailable.")
                stored = self._record(row)
                if stored.phase == "completed":
                    connection.execute("COMMIT")
                    return stored
                if stored.plan_digest != selected_plan:
                    raise RuntimeError("compaction plan identity changed.")
                connection.execute(
                    """
                    UPDATE evidence_graph_compaction
                    SET phase='completed', updated_at=?
                    WHERE job_id=? AND owner_id=? AND phase='planned'
                    """,
                    (timestamp, selected, owner),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(selected)
        if result is None or result.phase != "completed":
            raise RuntimeError("compaction completion was not persisted.")
        return result


def _candidate_by_id(plan: EvidenceGraphRetentionPlan) -> dict[str, EvidenceGraphRetentionCandidate]:
    result: dict[str, EvidenceGraphRetentionCandidate] = {}
    for candidate in plan.candidates:
        if not isinstance(candidate, EvidenceGraphRetentionCandidate):
            raise ValueError("retention plan contains an invalid candidate.")
        if candidate.job_id in result:
            raise ValueError("retention plan contains duplicate candidate jobs.")
        result[candidate.job_id] = candidate
    return result


def compact_evidence_graph_retention_plan(
    *,
    plan: EvidenceGraphRetentionPlan,
    journal: Any,
    generations: Any,
    graphs: Any,
    compactions: EvidenceGraphCompactionStore,
    confirm_plan_digest: str,
    confirm_job_ids: Iterable[str],
    now: float | None = None,
) -> EvidenceGraphCompactionResult:
    """Execute one exact retention plan without touching authoritative stores."""

    if not isinstance(plan, EvidenceGraphRetentionPlan):
        raise ValueError("plan must be EvidenceGraphRetentionPlan.")
    if not isinstance(compactions, EvidenceGraphCompactionStore):
        raise ValueError("compactions must be EvidenceGraphCompactionStore.")
    selected_plan = _digest(confirm_plan_digest, "confirm_plan_digest")
    if selected_plan != plan.plan_digest:
        raise ValueError("confirmation must exactly match plan_digest.")
    candidates = _candidate_by_id(plan)
    expected_ids = tuple(sorted(candidates))
    confirmed_ids = _confirmed_job_ids(confirm_job_ids)
    if confirmed_ids != expected_ids:
        raise ValueError("job confirmations must exactly match every plan candidate.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    if timestamp < plan.generated_at:
        raise ValueError("compaction time may not precede plan generation.")

    completed: list[str] = []
    already_completed: list[str] = []
    deleted_graphs: list[str] = []
    audit_only: list[str] = []

    authority_cache: dict[str, Any | None] = {}
    graph_current_cache: dict[str, Any | None] = {}
    for job_id in expected_ids:
        candidate = candidates[job_id]
        existing = compactions.get(job_id)
        job = journal.get(job_id)
        if job is None:
            if (
                existing is not None
                and existing.phase == "completed"
                and existing.owner_id == plan.owner_id
                and existing.source_sequence == candidate.source_sequence
                and existing.job_state == candidate.state
            ):
                already_completed.append(job_id)
                continue
            raise RuntimeError("retention candidate job is unavailable.")
        if not isinstance(job, EvidenceGraphJob):
            raise RuntimeError("graph job journal returned an invalid job.")
        if (
            job.owner_id != plan.owner_id
            or job.job_id != job_id
            or job.state != candidate.state
            or job.source_sequence != candidate.source_sequence
        ):
            raise RuntimeError("retention candidate identity changed.")
        expected_age = max(0.0, plan.generated_at - job.updated_at)
        if not math.isclose(expected_age, candidate.age_seconds, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("retention candidate age changed from the confirmed plan.")
        authoritative = authority_cache.get(job.doc_id)
        if job.doc_id not in authority_cache:
            authoritative = generations.current(owner_id=job.owner_id, doc_id=job.doc_id)
            authority_cache[job.doc_id] = authoritative
        if _job_matches_current(job, authoritative):
            raise RuntimeError("authoritative-current graph jobs may not be compacted.")

        if existing is not None and existing.phase == "completed":
            if (
                existing.owner_id != job.owner_id
                or existing.doc_id != job.doc_id
                or existing.source_sequence != job.source_sequence
                or existing.job_state != job.state
                or existing.graph_digest != job.graph_digest
            ):
                raise RuntimeError("completed compaction receipt identity is corrupt.")
            already_completed.append(job_id)
            continue

        if job.state == "completed":
            if job.graph_digest is None:
                raise RuntimeError("completed graph job is missing its graph digest.")
            current_graph = graph_current_cache.get(job.doc_id)
            if job.doc_id not in graph_current_cache:
                current_graph = graphs.current(owner_id=job.owner_id, doc_id=job.doc_id)
                graph_current_cache[job.doc_id] = current_graph
            if (
                current_graph is not None
                and getattr(current_graph, "generation", None) == job.source_sequence
            ):
                raise RuntimeError("current evidence graph generation may not be compacted.")
            historical_missing = False
            try:
                historical = graphs.get(
                    owner_id=job.owner_id,
                    doc_id=job.doc_id,
                    generation=job.source_sequence,
                )
            except KeyError:
                historical = None
                historical_missing = True
            if historical is not None and getattr(historical, "graph_digest", None) != job.graph_digest:
                raise RuntimeError("historical graph digest changed from the retention plan.")
            if historical_missing and (existing is None or existing.phase != "planned"):
                raise RuntimeError("historical graph is missing without a resumable compaction intent.")

        intent = compactions.begin(job=job, plan_digest=selected_plan, now=timestamp)
        if intent.phase == "completed":
            already_completed.append(job_id)
            continue
        if job.state == "completed":
            deleted = graphs.delete_generation(
                owner_id=job.owner_id,
                doc_id=job.doc_id,
                generation=job.source_sequence,
                confirm_graph_digest=job.graph_digest,
            )
            if deleted:
                deleted_graphs.append(job_id)
            else:
                latest = compactions.get(job_id)
                if latest is None or latest.phase not in {"planned", "completed"}:
                    raise RuntimeError("graph generation disappeared before compaction intent.")
        else:
            audit_only.append(job_id)
        compactions.complete(
            job_id,
            owner_id=job.owner_id,
            plan_digest=selected_plan,
            now=timestamp,
        )
        completed.append(job_id)

    return EvidenceGraphCompactionResult(
        owner_id=plan.owner_id,
        plan_digest=selected_plan,
        candidate_count=len(expected_ids),
        completed_job_ids=tuple(sorted(completed)),
        already_completed_job_ids=tuple(sorted(already_completed)),
        deleted_graph_generation_job_ids=tuple(sorted(deleted_graphs)),
        retained_job_audit_only_ids=tuple(sorted(audit_only)),
        completed_at=timestamp,
    )


__all__ = [
    "EvidenceGraphCompactionRecord",
    "EvidenceGraphCompactionResult",
    "EvidenceGraphCompactionStore",
    "compact_evidence_graph_retention_plan",
]
