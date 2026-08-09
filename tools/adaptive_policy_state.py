"""Durable shadow/promotion/rollback state for adaptive retrieval policies."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.adaptive_policy_governance import (
    AdaptivePolicyComparison,
    AdaptivePolicyDecision,
)
from tools.security import normalize_owner_id

_STATES = frozenset({"shadow", "promoted", "superseded", "rolled_back"})


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**63 - 1:
        raise ValueError("revision must be a positive integer.")
    return value


@dataclass(frozen=True)
class AdaptivePolicyState:
    owner_id: str
    revision: int
    policy_id: str
    policy_digest: str
    baseline_policy_id: str | None
    state: str
    comparison_digest: str | None
    shadow_metrics_digest: str | None
    decision_digest: str | None
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(self, "policy_digest", _digest(self.policy_digest, "policy_digest"))
        if self.baseline_policy_id is not None:
            object.__setattr__(
                self,
                "baseline_policy_id",
                _identifier(self.baseline_policy_id, "baseline_policy_id"),
            )
        if self.state not in _STATES:
            raise ValueError("policy state is unsupported.")
        for name in ("comparison_digest", "shadow_metrics_digest", "decision_digest"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.state == "shadow" and self.baseline_policy_id is None:
            raise ValueError("shadow policy state requires a baseline policy.")
        if self.state == "promoted" and self.baseline_policy_id is not None:
            if self.comparison_digest is None or self.decision_digest is None:
                raise ValueError("promoted candidate requires comparison and decision evidence.")
        if self.state == "rolled_back" and self.decision_digest is None:
            raise ValueError("rolled-back policy requires decision evidence.")


class AdaptivePolicyStateStore:
    """SQLite journal with monotonic revisions and exact decision-evidence binding."""

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
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_policy_state (
                    owner_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    baseline_policy_id TEXT,
                    state TEXT NOT NULL,
                    comparison_digest TEXT,
                    shadow_metrics_digest TEXT,
                    decision_digest TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, revision)
                )
                """
            )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS adaptive_policy_one_promoted "
                "ON adaptive_policy_state(owner_id) WHERE state='promoted'"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS adaptive_policy_by_policy "
                "ON adaptive_policy_state(owner_id, policy_id, revision)"
            )
        except sqlite3.Error as exc:
            raise RuntimeError("adaptive policy state initialization failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _state(row: tuple[Any, ...] | None) -> AdaptivePolicyState | None:
        if row is None:
            return None
        try:
            return AdaptivePolicyState(*row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("adaptive policy state is corrupt.") from exc

    def _get(self, owner_id: str, revision: int) -> AdaptivePolicyState | None:
        row = self._connection.execute(
            """
            SELECT owner_id, revision, policy_id, policy_digest, baseline_policy_id,
                   state, comparison_digest, shadow_metrics_digest, decision_digest,
                   created_at, updated_at
            FROM adaptive_policy_state WHERE owner_id=? AND revision=?
            """,
            (owner_id, revision),
        ).fetchone()
        return self._state(row)

    def get(self, owner_id: str, revision: int) -> AdaptivePolicyState | None:
        owner = normalize_owner_id(owner_id)
        selected = _revision(revision)
        with self._lock:
            try:
                return self._get(owner, selected)
            except sqlite3.Error as exc:
                raise RuntimeError("adaptive policy state read failed.") from exc

    def current_promoted(self, owner_id: str) -> AdaptivePolicyState | None:
        owner = normalize_owner_id(owner_id)
        with self._lock:
            try:
                row = self._connection.execute(
                    """
                    SELECT owner_id, revision, policy_id, policy_digest, baseline_policy_id,
                           state, comparison_digest, shadow_metrics_digest, decision_digest,
                           created_at, updated_at
                    FROM adaptive_policy_state
                    WHERE owner_id=? AND state='promoted'
                    """,
                    (owner,),
                ).fetchone()
                return self._state(row)
            except sqlite3.Error as exc:
                raise RuntimeError("adaptive policy state read failed.") from exc

    def _next_revision(self, owner_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(revision), 0) FROM adaptive_policy_state WHERE owner_id=?",
            (owner_id,),
        ).fetchone()
        return int(row[0]) + 1

    def bootstrap_promoted(
        self,
        *,
        owner_id: str,
        policy_id: str,
        policy_digest: str,
        now: float,
    ) -> AdaptivePolicyState:
        owner = normalize_owner_id(owner_id)
        policy = _identifier(policy_id, "policy_id")
        policy_hash = _digest(policy_digest, "policy_digest")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                current = self.current_promoted(owner)
                if current is not None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("a promoted policy already exists for this owner.")
                revision = self._next_revision(owner)
                self._connection.execute(
                    """
                    INSERT INTO adaptive_policy_state (
                        owner_id, revision, policy_id, policy_digest, baseline_policy_id,
                        state, comparison_digest, shadow_metrics_digest, decision_digest,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, NULL, 'promoted', NULL, NULL, NULL, ?, ?)
                    """,
                    (owner, revision, policy, policy_hash, current_time, current_time),
                )
                self._connection.execute("COMMIT")
                return self.get(owner, revision)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("adaptive policy bootstrap failed.") from exc

    def start_shadow(
        self,
        *,
        owner_id: str,
        policy_id: str,
        policy_digest: str,
        now: float,
    ) -> AdaptivePolicyState:
        owner = normalize_owner_id(owner_id)
        policy = _identifier(policy_id, "policy_id")
        policy_hash = _digest(policy_digest, "policy_digest")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                baseline = self.current_promoted(owner)
                if baseline is None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("shadow registration requires a promoted baseline.")
                if baseline.policy_id == policy:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("shadow policy must differ from the promoted baseline.")
                revision = self._next_revision(owner)
                self._connection.execute(
                    """
                    INSERT INTO adaptive_policy_state (
                        owner_id, revision, policy_id, policy_digest, baseline_policy_id,
                        state, comparison_digest, shadow_metrics_digest, decision_digest,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'shadow', NULL, NULL, NULL, ?, ?)
                    """,
                    (
                        owner,
                        revision,
                        policy,
                        policy_hash,
                        baseline.policy_id,
                        current_time,
                        current_time,
                    ),
                )
                self._connection.execute("COMMIT")
                return self.get(owner, revision)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("shadow policy registration failed.") from exc

    def record_shadow_evidence(
        self,
        *,
        owner_id: str,
        revision: int,
        comparison: AdaptivePolicyComparison,
        shadow_metrics_digest: str,
        now: float,
    ) -> AdaptivePolicyState:
        owner = normalize_owner_id(owner_id)
        selected = _revision(revision)
        if not isinstance(comparison, AdaptivePolicyComparison):
            raise ValueError("comparison must be AdaptivePolicyComparison.")
        metrics = _digest(shadow_metrics_digest, "shadow_metrics_digest")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                current = self._get(owner, selected)
                if current is None or current.state != "shadow":
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("shadow evidence requires the exact shadow revision.")
                if (
                    comparison.candidate_policy_id != current.policy_id
                    or comparison.baseline_policy_id != current.baseline_policy_id
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("shadow comparison does not match policy identities.")
                if current.comparison_digest is not None and (
                    current.comparison_digest != comparison.comparison_digest
                    or current.shadow_metrics_digest != metrics
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("shadow evidence collision requires a new revision.")
                self._connection.execute(
                    """
                    UPDATE adaptive_policy_state
                    SET comparison_digest=?, shadow_metrics_digest=?, updated_at=?
                    WHERE owner_id=? AND revision=? AND state='shadow'
                    """,
                    (comparison.comparison_digest, metrics, current_time, owner, selected),
                )
                self._connection.execute("COMMIT")
                return self.get(owner, selected)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("shadow evidence update failed.") from exc

    def promote(
        self,
        *,
        owner_id: str,
        revision: int,
        decision: AdaptivePolicyDecision,
        now: float,
    ) -> AdaptivePolicyState:
        owner = normalize_owner_id(owner_id)
        selected = _revision(revision)
        if not isinstance(decision, AdaptivePolicyDecision) or decision.decision != "eligible":
            raise ValueError("promotion requires one eligible AdaptivePolicyDecision.")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                candidate = self._get(owner, selected)
                baseline = self.current_promoted(owner)
                if candidate is None or candidate.state != "shadow" or baseline is None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("promotion requires exact shadow and promoted revisions.")
                comparison = decision.comparison
                if (
                    candidate.comparison_digest is None
                    or candidate.shadow_metrics_digest is None
                    or candidate.comparison_digest != comparison.comparison_digest
                    or comparison.candidate_policy_id != candidate.policy_id
                    or comparison.baseline_policy_id != baseline.policy_id
                    or candidate.baseline_policy_id != baseline.policy_id
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("promotion decision does not match recorded shadow evidence.")
                self._connection.execute(
                    "UPDATE adaptive_policy_state SET state='superseded', updated_at=? "
                    "WHERE owner_id=? AND revision=? AND state='promoted'",
                    (current_time, owner, baseline.revision),
                )
                self._connection.execute(
                    """
                    UPDATE adaptive_policy_state
                    SET state='promoted', decision_digest=?, updated_at=?
                    WHERE owner_id=? AND revision=? AND state='shadow'
                    """,
                    (decision.decision_digest, current_time, owner, selected),
                )
                self._connection.execute("COMMIT")
                return self.get(owner, selected)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("adaptive policy promotion failed.") from exc

    def rollback(
        self,
        *,
        owner_id: str,
        revision: int,
        decision: AdaptivePolicyDecision,
        now: float,
    ) -> AdaptivePolicyState:
        owner = normalize_owner_id(owner_id)
        selected = _revision(revision)
        if not isinstance(decision, AdaptivePolicyDecision) or decision.decision != "rollback":
            raise ValueError("rollback requires one rollback AdaptivePolicyDecision.")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                candidate = self._get(owner, selected)
                if candidate is None or candidate.state != "promoted" or candidate.baseline_policy_id is None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("rollback requires the exact promoted candidate revision.")
                comparison = decision.comparison
                if (
                    comparison.candidate_policy_id != candidate.policy_id
                    or comparison.baseline_policy_id != candidate.baseline_policy_id
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("rollback decision does not match policy identities.")
                row = self._connection.execute(
                    """
                    SELECT owner_id, revision, policy_id, policy_digest, baseline_policy_id,
                           state, comparison_digest, shadow_metrics_digest, decision_digest,
                           created_at, updated_at
                    FROM adaptive_policy_state
                    WHERE owner_id=? AND policy_id=? AND state='superseded'
                    ORDER BY revision DESC LIMIT 1
                    """,
                    (owner, candidate.baseline_policy_id),
                ).fetchone()
                baseline = self._state(row)
                if baseline is None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("rollback baseline revision is unavailable.")
                self._connection.execute(
                    """
                    UPDATE adaptive_policy_state
                    SET state='rolled_back', comparison_digest=?, decision_digest=?, updated_at=?
                    WHERE owner_id=? AND revision=? AND state='promoted'
                    """,
                    (
                        comparison.comparison_digest,
                        decision.decision_digest,
                        current_time,
                        owner,
                        selected,
                    ),
                )
                self._connection.execute(
                    "UPDATE adaptive_policy_state SET state='promoted', updated_at=? "
                    "WHERE owner_id=? AND revision=? AND state='superseded'",
                    (current_time, owner, baseline.revision),
                )
                self._connection.execute("COMMIT")
                return self.get(owner, selected)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("adaptive policy rollback failed.") from exc


__all__ = ["AdaptivePolicyState", "AdaptivePolicyStateStore"]
