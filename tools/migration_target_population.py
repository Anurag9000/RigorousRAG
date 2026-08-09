"""Durable hidden-target population journal and execution fencing.

The blue/green cutover adapter deliberately keeps physical population separate from
route visibility. This module adds durable intent/receipt evidence for that hidden
population phase and a cross-process SQLite lease with monotonic fencing. Stored rows
contain only identifiers, digests, counts and timestamps; document text and embeddings
never enter the journal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_STATES = {"planned", "populated", "visible", "aborted", "rolled_back"}
_RECONCILIATION_STATES = {
    "missing_without_intent",
    "missing_with_intent",
    "in_progress",
    "populated",
    "visible",
    "aborted",
    "rolled_back",
    "orphan",
    "authority_conflict",
}
_MAX_ROWS = 100_000_000
_MAX_LEASE_SECONDS = 86_400.0


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _duration(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("lease_seconds must be finite and positive.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lease_seconds must be finite and positive.") from exc
    if not math.isfinite(selected) or not 0.0 < selected <= _MAX_LEASE_SECONDS:
        raise ValueError("lease_seconds must be between 0 and 86400.")
    return selected


@dataclass(frozen=True)
class TargetPopulationIdentity:
    operation_id: str
    owner_id: str
    doc_id: str
    target_collection_id: str
    target_profile_fingerprint: str
    content_sha256: str
    target_artifact_digest: str
    expected_vector_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", digest(self.operation_id, "operation_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id", 200))
        for name in (
            "target_collection_id",
            "target_profile_fingerprint",
            "content_sha256",
            "target_artifact_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "expected_vector_rows",
            exact_integer(
                self.expected_vector_rows,
                "expected_vector_rows",
                1,
                _MAX_ROWS,
            ),
        )

    @property
    def identity_digest(self) -> str:
        return _sha256(
            {"contract": "rigorousrag-target-population-identity-v1", **asdict(self)}
        )


@dataclass(frozen=True)
class TargetPopulationRecord:
    identity: TargetPopulationIdentity
    state: str
    attempt: int
    created_at: float
    updated_at: float
    population_digest: str | None = None
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TargetPopulationIdentity):
            raise ValueError("identity must be TargetPopulationIdentity.")
        if self.state not in _STATES:
            raise ValueError("target population state is invalid.")
        object.__setattr__(
            self,
            "attempt",
            exact_integer(self.attempt, "attempt", 0, 1_000_000),
        )
        created = timestamp(self.created_at, "created_at")
        updated = timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.population_digest is not None:
            object.__setattr__(
                self,
                "population_digest",
                digest(self.population_digest, "population_digest"),
            )
        if self.receipt_digest is not None:
            object.__setattr__(
                self,
                "receipt_digest",
                digest(self.receipt_digest, "receipt_digest"),
            )
        if self.state == "planned" and (
            self.population_digest is not None or self.receipt_digest is not None
        ):
            raise ValueError("planned population may not carry terminal evidence.")
        if self.state in {"populated", "visible", "rolled_back"} and (
            self.population_digest is None or self.receipt_digest is None
        ):
            raise ValueError("populated states require population and receipt digests.")
        if self.state == "aborted" and self.receipt_digest is None:
            raise ValueError("aborted population requires a receipt digest.")


@dataclass(frozen=True)
class TargetPopulationClaim:
    operation_id: str
    worker_id: str
    fencing_token: int
    lease_expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", digest(self.operation_id, "operation_id"))
        object.__setattr__(self, "worker_id", identifier(self.worker_id, "worker_id", 128))
        object.__setattr__(
            self,
            "fencing_token",
            exact_integer(self.fencing_token, "fencing_token", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "lease_expires_at",
            timestamp(self.lease_expires_at, "lease_expires_at"),
        )


@dataclass(frozen=True)
class TargetPopulationReconciliation:
    operation_id: str
    state: str
    expected_rows: int
    observed_rows: int
    route_points_to_target: bool
    exact_population_match: bool
    intent_present: bool
    reconciliation_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", digest(self.operation_id, "operation_id"))
        if self.state not in _RECONCILIATION_STATES:
            raise ValueError("target population reconciliation state is invalid.")
        object.__setattr__(
            self,
            "expected_rows",
            exact_integer(self.expected_rows, "expected_rows", 1, _MAX_ROWS),
        )
        object.__setattr__(
            self,
            "observed_rows",
            exact_integer(self.observed_rows, "observed_rows", 0, _MAX_ROWS),
        )
        for name in ("route_points_to_target", "exact_population_match", "intent_present"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        object.__setattr__(
            self,
            "reconciliation_digest",
            digest(self.reconciliation_digest, "reconciliation_digest"),
        )


class TargetPopulationJournal:
    """SQLite intent/receipt store with expiring monotonic-fenced executor claims."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        candidate = Path(os.fspath(path))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        self.path = candidate.absolute()
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS target_population (
                    operation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    target_collection_id TEXT NOT NULL,
                    target_profile_fingerprint TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    target_artifact_digest TEXT NOT NULL,
                    expected_vector_rows INTEGER NOT NULL,
                    identity_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    population_digest TEXT,
                    receipt_digest TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS target_population_claims (
                    operation_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS target_population_owner_doc "
                "ON target_population(owner_id, doc_id, updated_at)"
            )
        except sqlite3.Error as exc:
            raise RuntimeError("target population journal initialization failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _record(row: tuple[Any, ...] | None) -> TargetPopulationRecord | None:
        if row is None:
            return None
        try:
            identity = TargetPopulationIdentity(*row[:8])
            if row[8] != identity.identity_digest:
                raise ValueError("identity digest mismatch")
            return TargetPopulationRecord(
                identity=identity,
                state=row[9],
                attempt=row[10],
                created_at=row[11],
                updated_at=row[12],
                population_digest=row[13],
                receipt_digest=row[14],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("target population record is corrupt.") from exc

    def _get(self, operation_id: str) -> TargetPopulationRecord | None:
        row = self._connection.execute(
            """
            SELECT operation_id, owner_id, doc_id, target_collection_id,
                   target_profile_fingerprint, content_sha256, target_artifact_digest,
                   expected_vector_rows, identity_digest, state, attempt, created_at,
                   updated_at, population_digest, receipt_digest
            FROM target_population WHERE operation_id=?
            """,
            (operation_id,),
        ).fetchone()
        return self._record(row)

    def get(self, operation_id: str) -> TargetPopulationRecord | None:
        selected = digest(operation_id, "operation_id")
        with self._lock:
            try:
                return self._get(selected)
            except sqlite3.Error as exc:
                raise RuntimeError("target population journal read failed.") from exc

    def ensure_intent(
        self,
        identity: TargetPopulationIdentity,
        *,
        now: float,
    ) -> TargetPopulationRecord:
        if not isinstance(identity, TargetPopulationIdentity):
            raise ValueError("identity must be TargetPopulationIdentity.")
        current_time = timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._get(identity.operation_id)
                if existing is not None:
                    self._connection.execute("COMMIT")
                    if existing.identity != identity:
                        raise RuntimeError("target population operation identity collision.")
                    return existing
                self._connection.execute(
                    """
                    INSERT INTO target_population (
                        operation_id, owner_id, doc_id, target_collection_id,
                        target_profile_fingerprint, content_sha256, target_artifact_digest,
                        expected_vector_rows, identity_digest, state, attempt,
                        created_at, updated_at, population_digest, receipt_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', 0, ?, ?, NULL, NULL)
                    """,
                    (
                        identity.operation_id,
                        identity.owner_id,
                        identity.doc_id,
                        identity.target_collection_id,
                        identity.target_profile_fingerprint,
                        identity.content_sha256,
                        identity.target_artifact_digest,
                        identity.expected_vector_rows,
                        identity.identity_digest,
                        current_time,
                        current_time,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO target_population_claims (
                        operation_id, worker_id, fencing_token, lease_expires_at
                    ) VALUES (?, NULL, 0, NULL)
                    """,
                    (identity.operation_id,),
                )
                self._connection.execute("COMMIT")
                created = self._get(identity.operation_id)
                if created is None:
                    raise RuntimeError("target population intent disappeared after creation.")
                return created
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("target population intent creation failed.") from exc

    def claim(
        self,
        operation_id: str,
        *,
        worker_id: str,
        now: float,
        lease_seconds: float,
    ) -> TargetPopulationClaim:
        selected = digest(operation_id, "operation_id")
        worker = identifier(worker_id, "worker_id", 128)
        current_time = timestamp(now, "now")
        expiry = current_time + _duration(lease_seconds)
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if self._get(selected) is None:
                    raise KeyError(selected)
                row = self._connection.execute(
                    "SELECT worker_id, fencing_token, lease_expires_at "
                    "FROM target_population_claims WHERE operation_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("target population claim row is missing.")
                current_worker, token, current_expiry = row
                if (
                    current_worker is not None
                    and current_expiry is not None
                    and float(current_expiry) > current_time
                ):
                    raise RuntimeError("target population already has a live executor.")
                next_token = int(token) + 1
                if next_token > 2**63 - 1:
                    raise RuntimeError("target population fencing token exhausted.")
                updated = self._connection.execute(
                    "UPDATE target_population SET attempt=attempt+1, updated_at=? "
                    "WHERE operation_id=? AND attempt<1000000",
                    (current_time, selected),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("target population attempt budget exhausted.")
                self._connection.execute(
                    "UPDATE target_population_claims "
                    "SET worker_id=?, fencing_token=?, lease_expires_at=? "
                    "WHERE operation_id=?",
                    (worker, next_token, expiry, selected),
                )
                self._connection.execute("COMMIT")
                return TargetPopulationClaim(selected, worker, next_token, expiry)
            except (KeyError, RuntimeError):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("target population claim failed.") from exc

    def assert_claim(self, claim: TargetPopulationClaim, *, now: float) -> None:
        if not isinstance(claim, TargetPopulationClaim):
            raise ValueError("claim must be TargetPopulationClaim.")
        current_time = timestamp(now, "now")
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT worker_id, fencing_token, lease_expires_at "
                    "FROM target_population_claims WHERE operation_id=?",
                    (claim.operation_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise RuntimeError("target population claim read failed.") from exc
        if row is None:
            raise RuntimeError("target population claim is unavailable.")
        worker, token, expiry = row
        if (
            worker != claim.worker_id
            or token != claim.fencing_token
            or expiry is None
            or float(expiry) <= current_time
        ):
            raise RuntimeError("target population executor lease is stale or fenced.")

    def renew(
        self,
        claim: TargetPopulationClaim,
        *,
        now: float,
        lease_seconds: float,
    ) -> TargetPopulationClaim:
        self.assert_claim(claim, now=now)
        current_time = timestamp(now, "now")
        expiry = current_time + _duration(lease_seconds)
        with self._lock:
            try:
                updated = self._connection.execute(
                    """
                    UPDATE target_population_claims SET lease_expires_at=?
                    WHERE operation_id=? AND worker_id=? AND fencing_token=?
                      AND lease_expires_at>?
                    """,
                    (
                        expiry,
                        claim.operation_id,
                        claim.worker_id,
                        claim.fencing_token,
                        current_time,
                    ),
                )
            except sqlite3.Error as exc:
                raise RuntimeError("target population claim renewal failed.") from exc
        if updated.rowcount != 1:
            raise RuntimeError("target population executor lease was lost during renewal.")
        return TargetPopulationClaim(
            claim.operation_id,
            claim.worker_id,
            claim.fencing_token,
            expiry,
        )

    def release(self, claim: TargetPopulationClaim, *, now: float) -> None:
        self.assert_claim(claim, now=now)
        current_time = timestamp(now, "now")
        with self._lock:
            try:
                updated = self._connection.execute(
                    """
                    UPDATE target_population_claims
                    SET worker_id=NULL, lease_expires_at=NULL
                    WHERE operation_id=? AND worker_id=? AND fencing_token=?
                    """,
                    (claim.operation_id, claim.worker_id, claim.fencing_token),
                )
                self._connection.execute(
                    "UPDATE target_population SET updated_at=? WHERE operation_id=?",
                    (current_time, claim.operation_id),
                )
            except sqlite3.Error as exc:
                raise RuntimeError("target population claim release failed.") from exc
        if updated.rowcount != 1:
            raise RuntimeError("target population executor lease was lost before release.")

    @staticmethod
    def population_digest(
        identity: TargetPopulationIdentity,
        *,
        row_digest: str,
    ) -> str:
        if not isinstance(identity, TargetPopulationIdentity):
            raise ValueError("identity must be TargetPopulationIdentity.")
        observed = digest(row_digest, "row_digest")
        return _sha256(
            {
                "contract": "rigorousrag-target-population-content-v1",
                "identity_digest": identity.identity_digest,
                "row_digest": observed,
                "rows": identity.expected_vector_rows,
            }
        )

    def _mark(
        self,
        identity: TargetPopulationIdentity,
        *,
        state: str,
        row_digest: str | None,
        evidence: dict[str, Any],
        now: float,
    ) -> TargetPopulationRecord:
        if not isinstance(identity, TargetPopulationIdentity):
            raise ValueError("identity must be TargetPopulationIdentity.")
        if state not in _STATES - {"planned"}:
            raise ValueError("target population terminal state is invalid.")
        current_time = timestamp(now, "now")
        population = (
            None
            if row_digest is None
            else self.population_digest(identity, row_digest=row_digest)
        )
        if state in {"populated", "visible", "rolled_back"} and population is None:
            raise ValueError("population evidence is required for this state.")
        receipt = _sha256(
            {
                "contract": "rigorousrag-target-population-receipt-v1",
                "identity_digest": identity.identity_digest,
                "state": state,
                "population_digest": population,
                **evidence,
            }
        )
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                current = self._get(identity.operation_id)
                if current is None or current.identity != identity:
                    raise RuntimeError("target population intent is missing or changed.")
                if current.state == "visible" and state not in {"visible", "rolled_back"}:
                    raise RuntimeError("visible target population cannot transition backward.")
                if current.state == "aborted" and state != "aborted":
                    raise RuntimeError("aborted target population cannot be revived.")
                if current.population_digest is not None and population is not None:
                    if current.population_digest != population:
                        raise RuntimeError("target population content changed across receipts.")
                self._connection.execute(
                    """
                    UPDATE target_population
                    SET state=?, updated_at=?, population_digest=?, receipt_digest=?
                    WHERE operation_id=?
                    """,
                    (state, current_time, population, receipt, identity.operation_id),
                )
                self._connection.execute("COMMIT")
                result = self._get(identity.operation_id)
                if result is None:
                    raise RuntimeError("target population receipt disappeared.")
                return result
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("target population receipt write failed.") from exc

    def mark_populated(
        self,
        identity: TargetPopulationIdentity,
        *,
        row_digest: str,
        now: float,
    ) -> TargetPopulationRecord:
        return self._mark(
            identity,
            state="populated",
            row_digest=row_digest,
            evidence={},
            now=now,
        )

    def mark_visible(
        self,
        identity: TargetPopulationIdentity,
        *,
        row_digest: str,
        route_digest: str,
        generation_sequence: int,
        now: float,
    ) -> TargetPopulationRecord:
        return self._mark(
            identity,
            state="visible",
            row_digest=row_digest,
            evidence={
                "route_digest": digest(route_digest, "route_digest"),
                "generation_sequence": exact_integer(
                    generation_sequence,
                    "generation_sequence",
                    1,
                    2**63 - 1,
                ),
            },
            now=now,
        )

    def mark_aborted(
        self,
        identity: TargetPopulationIdentity,
        *,
        now: float,
    ) -> TargetPopulationRecord:
        return self._mark(
            identity,
            state="aborted",
            row_digest=None,
            evidence={},
            now=now,
        )

    def mark_rolled_back(
        self,
        identity: TargetPopulationIdentity,
        *,
        row_digest: str,
        route_digest: str,
        generation_sequence: int,
        now: float,
    ) -> TargetPopulationRecord:
        return self._mark(
            identity,
            state="rolled_back",
            row_digest=row_digest,
            evidence={
                "route_digest": digest(route_digest, "route_digest"),
                "generation_sequence": exact_integer(
                    generation_sequence,
                    "generation_sequence",
                    1,
                    2**63 - 1,
                ),
            },
            now=now,
        )


def reconcile_target_population(
    identity: TargetPopulationIdentity,
    record: TargetPopulationRecord | None,
    *,
    observed_rows: int,
    exact_population_match: bool,
    route_collection_id: str | None,
) -> TargetPopulationReconciliation:
    """Classify physical target state without mutating any authoritative store."""

    if not isinstance(identity, TargetPopulationIdentity):
        raise ValueError("identity must be TargetPopulationIdentity.")
    observed = exact_integer(observed_rows, "observed_rows", 0, _MAX_ROWS)
    if not isinstance(exact_population_match, bool):
        raise ValueError("exact_population_match must be boolean.")
    if record is not None and (
        not isinstance(record, TargetPopulationRecord) or record.identity != identity
    ):
        raise ValueError("record does not match target population identity.")
    route_to_target = False
    if route_collection_id is not None:
        route_to_target = (
            digest(route_collection_id, "route_collection_id")
            == identity.target_collection_id
        )

    exact_rows = exact_population_match and observed == identity.expected_vector_rows
    if route_to_target:
        state = "visible" if record is not None and exact_rows else "authority_conflict"
    elif record is None:
        state = "missing_without_intent" if observed == 0 else "orphan"
    elif record.state == "aborted":
        state = "aborted" if observed == 0 else "authority_conflict"
    elif record.state == "rolled_back":
        state = "rolled_back" if exact_rows else "authority_conflict"
    elif observed == 0:
        state = "missing_with_intent"
    elif exact_rows:
        state = "populated"
    elif 0 < observed < identity.expected_vector_rows:
        state = "in_progress"
    else:
        state = "authority_conflict"

    evidence = {
        "contract": "rigorousrag-target-population-reconciliation-v1",
        "identity_digest": identity.identity_digest,
        "record_state": None if record is None else record.state,
        "state": state,
        "expected_rows": identity.expected_vector_rows,
        "observed_rows": observed,
        "route_points_to_target": route_to_target,
        "exact_population_match": exact_population_match,
        "intent_present": record is not None,
    }
    return TargetPopulationReconciliation(
        operation_id=identity.operation_id,
        state=state,
        expected_rows=identity.expected_vector_rows,
        observed_rows=observed,
        route_points_to_target=route_to_target,
        exact_population_match=exact_population_match,
        intent_present=record is not None,
        reconciliation_digest=_sha256(evidence),
    )


__all__ = [
    "TargetPopulationClaim",
    "TargetPopulationIdentity",
    "TargetPopulationJournal",
    "TargetPopulationReconciliation",
    "TargetPopulationRecord",
    "reconcile_target_population",
]
