"""Tamper-evident durable journal for feedback-driven model promotions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from tools.feedback_promotion import PromotionDecision

_ACTIONS = {"eligible", "rejected", "promoted", "rolled_back"}
_ALLOWED_PREDECESSORS = {
    "eligible": {None},
    "rejected": {None, "eligible"},
    "promoted": {"eligible"},
    "rolled_back": {"promoted"},
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


@dataclass(frozen=True)
class PromotionJournalEntry:
    sequence: int
    decision_id: str
    batch_id: str
    owner_id: str
    baseline_version: str
    candidate_version: str
    action: str
    actor: str
    timestamp: float
    previous_hash: str
    record_hash: str


@dataclass(frozen=True)
class JournalVerification:
    valid: bool
    entry_count: int
    first_invalid_sequence: int | None = None


class PromotionJournal:
    """SQLite-backed append-only hash chain with promotion transition validation."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS promotion_journal ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
                "decision_id TEXT NOT NULL, batch_id TEXT NOT NULL, owner_id TEXT NOT NULL,"
                "baseline_version TEXT NOT NULL, candidate_version TEXT NOT NULL,"
                "action TEXT NOT NULL, actor TEXT NOT NULL, timestamp REAL NOT NULL,"
                "previous_hash TEXT NOT NULL, record_hash TEXT NOT NULL UNIQUE,"
                "UNIQUE(decision_id, action))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotion_journal_decision "
                "ON promotion_journal(decision_id, sequence)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=10.0, isolation_level="IMMEDIATE")

    @staticmethod
    def _payload(
        *,
        sequence: int,
        decision: PromotionDecision,
        action: str,
        actor: str,
        timestamp: float,
        previous_hash: str,
    ) -> dict[str, object]:
        return {
            "sequence": sequence,
            "decision_id": decision.decision_id,
            "batch_id": decision.batch_id,
            "owner_id": decision.owner_id,
            "baseline_version": decision.baseline_version,
            "candidate_version": decision.candidate_version,
            "action": action,
            "actor": actor,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
        }

    @staticmethod
    def _entry(row: tuple[object, ...]) -> PromotionJournalEntry:
        return PromotionJournalEntry(
            sequence=int(row[0]),
            decision_id=str(row[1]),
            batch_id=str(row[2]),
            owner_id=str(row[3]),
            baseline_version=str(row[4]),
            candidate_version=str(row[5]),
            action=str(row[6]),
            actor=str(row[7]),
            timestamp=float(row[8]),
            previous_hash=str(row[9]),
            record_hash=str(row[10]),
        )

    def append(
        self, *, decision: PromotionDecision, action: str, actor: str
    ) -> PromotionJournalEntry:
        selected_action = _text(action, "action", 64)
        if selected_action not in _ACTIONS:
            raise ValueError("promotion action is unsupported.")
        selected_actor = _text(actor, "actor")
        if selected_action == "eligible" and not decision.eligible:
            raise ValueError("an ineligible decision cannot be marked eligible.")
        if selected_action == "promoted" and not decision.eligible:
            raise ValueError("an ineligible decision cannot be promoted.")

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT sequence,decision_id,batch_id,owner_id,baseline_version,candidate_version,"
                "action,actor,timestamp,previous_hash,record_hash FROM promotion_journal "
                "WHERE decision_id=? AND action=?",
                (decision.decision_id, selected_action),
            ).fetchone()
            if existing is not None:
                entry = self._entry(existing)
                expected_identity = (
                    decision.batch_id,
                    decision.owner_id,
                    decision.baseline_version,
                    decision.candidate_version,
                )
                actual_identity = (
                    entry.batch_id,
                    entry.owner_id,
                    entry.baseline_version,
                    entry.candidate_version,
                )
                if actual_identity != expected_identity:
                    raise ValueError("journal replay does not match the existing decision identity.")
                return entry

            previous_for_decision = connection.execute(
                "SELECT action FROM promotion_journal WHERE decision_id=? ORDER BY sequence DESC LIMIT 1",
                (decision.decision_id,),
            ).fetchone()
            prior_action = None if previous_for_decision is None else str(previous_for_decision[0])
            if prior_action not in _ALLOWED_PREDECESSORS[selected_action]:
                raise ValueError(
                    f"invalid promotion transition: {prior_action!r} -> {selected_action!r}."
                )

            last = connection.execute(
                "SELECT sequence,record_hash FROM promotion_journal ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 1 if last is None else int(last[0]) + 1
            previous_hash = "0" * 64 if last is None else str(last[1])
            timestamp = float(self._clock())
            payload = self._payload(
                sequence=sequence,
                decision=decision,
                action=selected_action,
                actor=selected_actor,
                timestamp=timestamp,
                previous_hash=previous_hash,
            )
            record_hash = _sha256(payload)
            connection.execute(
                "INSERT INTO promotion_journal(sequence,decision_id,batch_id,owner_id,"
                "baseline_version,candidate_version,action,actor,timestamp,previous_hash,record_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sequence,
                    decision.decision_id,
                    decision.batch_id,
                    decision.owner_id,
                    decision.baseline_version,
                    decision.candidate_version,
                    selected_action,
                    selected_actor,
                    timestamp,
                    previous_hash,
                    record_hash,
                ),
            )
            return PromotionJournalEntry(record_hash=record_hash, **payload)

    def entries(self) -> tuple[PromotionJournalEntry, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence,decision_id,batch_id,owner_id,baseline_version,candidate_version,"
                "action,actor,timestamp,previous_hash,record_hash FROM promotion_journal "
                "ORDER BY sequence"
            ).fetchall()
        return tuple(self._entry(row) for row in rows)

    def verify_chain(self) -> JournalVerification:
        entries = self.entries()
        previous_hash = "0" * 64
        expected_sequence = 1
        for entry in entries:
            if entry.sequence != expected_sequence or entry.previous_hash != previous_hash:
                return JournalVerification(False, len(entries), entry.sequence)
            payload = asdict(entry)
            record_hash = str(payload.pop("record_hash"))
            if _sha256(payload) != record_hash:
                return JournalVerification(False, len(entries), entry.sequence)
            previous_hash = record_hash
            expected_sequence += 1
        return JournalVerification(True, len(entries), None)


__all__ = ["JournalVerification", "PromotionJournal", "PromotionJournalEntry"]
