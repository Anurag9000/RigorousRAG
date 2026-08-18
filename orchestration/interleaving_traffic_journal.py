"""Owner-scoped durable exposure journal for interleaving mutual-exclusion groups."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path

from evaluation.interleaving_traffic import TrafficAssignment


def _text(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _time(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("now must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError("now must be finite and non-negative")
    return selected


class SQLiteInterleavingTrafficJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS interleaving_assignments (
                    assignment_sha256 TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    exclusion_group_id TEXT NOT NULL,
                    randomization_unit_sha256 TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS interleaving_exposure_slot_idx
                   ON interleaving_assignments(owner_id,exclusion_group_id,randomization_unit_sha256)
                   WHERE arm='interleaving'"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS interleaving_assignment_owner_idx ON interleaving_assignments(owner_id,spec_sha256,arm)"
            )

    def record_assignment(self, assignment: TrafficAssignment, *, now: float) -> str:
        if not isinstance(assignment, TrafficAssignment):
            raise ValueError("assignment must be TrafficAssignment")
        timestamp = _time(now)
        payload = json.dumps(asdict(assignment), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json FROM interleaving_assignments WHERE assignment_sha256=?",
                (assignment.assignment_sha256,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise RuntimeError("traffic assignment identity collision")
                return assignment.assignment_sha256
            if assignment.arm == "interleaving":
                conflict = connection.execute(
                    """SELECT spec_sha256,assignment_sha256 FROM interleaving_assignments
                       WHERE owner_id=? AND exclusion_group_id=? AND randomization_unit_sha256=?
                         AND arm='interleaving'""",
                    (assignment.owner_id, assignment.exclusion_group_id, assignment.randomization_unit_sha256),
                ).fetchone()
                if conflict is not None and conflict["spec_sha256"] != assignment.spec_sha256:
                    raise RuntimeError("randomization unit is already exposed to another experiment in the exclusion group")
            try:
                connection.execute(
                    "INSERT INTO interleaving_assignments(assignment_sha256,owner_id,exclusion_group_id,randomization_unit_sha256,spec_sha256,arm,payload_json,recorded_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        assignment.assignment_sha256,
                        assignment.owner_id,
                        assignment.exclusion_group_id,
                        assignment.randomization_unit_sha256,
                        assignment.spec_sha256,
                        assignment.arm,
                        payload,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("interleaving mutual-exclusion slot is already occupied") from exc
        return assignment.assignment_sha256

    def exposed_spec(
        self,
        *,
        owner_id: str,
        exclusion_group_id: str,
        randomization_unit_sha256: str,
    ) -> str | None:
        owner = _text(owner_id, "owner_id")
        group = _text(exclusion_group_id, "exclusion_group_id")
        unit = _text(randomization_unit_sha256, "randomization_unit_sha256", 64).lower()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT spec_sha256 FROM interleaving_assignments
                   WHERE owner_id=? AND exclusion_group_id=? AND randomization_unit_sha256=? AND arm='interleaving'""",
                (owner, group, unit),
            ).fetchone()
        return None if row is None else str(row["spec_sha256"])


__all__ = ["SQLiteInterleavingTrafficJournal"]
