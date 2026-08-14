"""Canary deployment decisions bound to immutable candidate and known-good artifact sets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from tools.recovery_control import (
    CanaryAction,
    CanaryObservation,
    CanaryThresholds,
    evaluate_canary,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest_items(values: tuple[tuple[str, str], ...], label: str) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, digest in values:
        selected_name = str(name).strip()
        selected_digest = str(digest).lower().strip()
        if not selected_name or selected_name in seen:
            raise ValueError(f"{label} artifact names must be non-empty and unique")
        if len(selected_digest) != 64 or any(ch not in "0123456789abcdef" for ch in selected_digest):
            raise ValueError(f"{label} artifact digests must be SHA-256 hex")
        seen.add(selected_name)
        normalized.append((selected_name, selected_digest))
    if not normalized:
        raise ValueError(f"{label} artifact set must not be empty")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class DeploymentCandidate:
    candidate_id: str
    candidate_artifacts: tuple[tuple[str, str], ...]
    known_good_artifacts: tuple[tuple[str, str], ...]
    evidence_generated_at: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        object.__setattr__(
            self,
            "candidate_artifacts",
            _digest_items(self.candidate_artifacts, "candidate"),
        )
        object.__setattr__(
            self,
            "known_good_artifacts",
            _digest_items(self.known_good_artifacts, "known-good"),
        )


@dataclass(frozen=True)
class DeploymentDecision:
    decision_id: str
    candidate_id: str
    action: CanaryAction
    reason_codes: tuple[str, ...]
    candidate_artifacts: tuple[tuple[str, str], ...]
    rollback_artifacts: tuple[tuple[str, str], ...]
    evaluated_at: float


def evaluate_deployment_canary(
    candidate: DeploymentCandidate,
    observation: CanaryObservation,
    *,
    thresholds: CanaryThresholds | None = None,
    now: float | None = None,
    max_evidence_age_seconds: float = 300.0,
) -> DeploymentDecision:
    evaluated_at = float(time.time() if now is None else now)
    max_age = float(max_evidence_age_seconds)
    if max_age < 0.0:
        raise ValueError("max_evidence_age_seconds must not be negative")
    age = evaluated_at - float(candidate.evidence_generated_at)
    if age < 0.0:
        action = CanaryAction.HOLD
        reasons = ("canary_evidence_from_future",)
    elif age > max_age:
        action = CanaryAction.HOLD
        reasons = ("canary_evidence_stale",)
    else:
        decision = evaluate_canary(observation, thresholds)
        action = decision.action
        reasons = decision.reason_codes
    rollback = candidate.known_good_artifacts if action == CanaryAction.ROLLBACK else ()
    payload = {
        "candidate_id": candidate.candidate_id,
        "action": action.value,
        "reason_codes": reasons,
        "candidate_artifacts": candidate.candidate_artifacts,
        "rollback_artifacts": rollback,
        "evaluated_at": evaluated_at,
    }
    decision_id = hashlib.sha256(_canonical(payload)).hexdigest()
    return DeploymentDecision(decision_id=decision_id, **payload)


@dataclass(frozen=True)
class DeploymentJournalEntry:
    sequence: int
    decision_id: str
    candidate_id: str
    action: str
    actor: str
    timestamp: float
    previous_hash: str
    record_hash: str


class DeploymentJournal:
    """SQLite append-only hash chain for canary promotion/rollback decisions."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS deployment_journal ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL UNIQUE,"
                "candidate_id TEXT NOT NULL, action TEXT NOT NULL, actor TEXT NOT NULL,"
                "timestamp REAL NOT NULL, previous_hash TEXT NOT NULL, record_hash TEXT NOT NULL UNIQUE)"
            )

    @staticmethod
    def _entry(row: tuple[object, ...]) -> DeploymentJournalEntry:
        return DeploymentJournalEntry(
            sequence=int(row[0]),
            decision_id=str(row[1]),
            candidate_id=str(row[2]),
            action=str(row[3]),
            actor=str(row[4]),
            timestamp=float(row[5]),
            previous_hash=str(row[6]),
            record_hash=str(row[7]),
        )

    def append(self, decision: DeploymentDecision, *, actor: str) -> DeploymentJournalEntry:
        selected_actor = str(actor).strip()
        if not selected_actor or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected_actor):
            raise ValueError("actor is invalid")
        with sqlite3.connect(str(self.path), timeout=10.0, isolation_level="IMMEDIATE") as connection:
            existing = connection.execute(
                "SELECT sequence,decision_id,candidate_id,action,actor,timestamp,previous_hash,record_hash "
                "FROM deployment_journal WHERE decision_id=?",
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                entry = self._entry(existing)
                if entry.candidate_id != decision.candidate_id or entry.action != decision.action.value:
                    raise ValueError("journal replay does not match deployment decision")
                return entry
            previous = connection.execute(
                "SELECT sequence,record_hash FROM deployment_journal ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 1 if previous is None else int(previous[0]) + 1
            previous_hash = "0" * 64 if previous is None else str(previous[1])
            timestamp = float(self._clock())
            payload = {
                "sequence": sequence,
                "decision_id": decision.decision_id,
                "candidate_id": decision.candidate_id,
                "action": decision.action.value,
                "actor": selected_actor,
                "timestamp": timestamp,
                "previous_hash": previous_hash,
            }
            record_hash = hashlib.sha256(_canonical(payload)).hexdigest()
            connection.execute(
                "INSERT INTO deployment_journal(sequence,decision_id,candidate_id,action,actor,"
                "timestamp,previous_hash,record_hash) VALUES(?,?,?,?,?,?,?,?)",
                (
                    sequence,
                    decision.decision_id,
                    decision.candidate_id,
                    decision.action.value,
                    selected_actor,
                    timestamp,
                    previous_hash,
                    record_hash,
                ),
            )
            return DeploymentJournalEntry(record_hash=record_hash, **payload)

    def entries(self) -> tuple[DeploymentJournalEntry, ...]:
        with sqlite3.connect(str(self.path)) as connection:
            rows = connection.execute(
                "SELECT sequence,decision_id,candidate_id,action,actor,timestamp,previous_hash,record_hash "
                "FROM deployment_journal ORDER BY sequence"
            ).fetchall()
        return tuple(self._entry(row) for row in rows)

    def verify_chain(self) -> bool:
        previous_hash = "0" * 64
        for expected_sequence, entry in enumerate(self.entries(), start=1):
            if entry.sequence != expected_sequence or entry.previous_hash != previous_hash:
                return False
            payload = asdict(entry)
            record_hash = str(payload.pop("record_hash"))
            if hashlib.sha256(_canonical(payload)).hexdigest() != record_hash:
                return False
            previous_hash = record_hash
        return True


__all__ = [
    "DeploymentCandidate",
    "DeploymentDecision",
    "DeploymentJournal",
    "DeploymentJournalEntry",
    "evaluate_deployment_canary",
]
