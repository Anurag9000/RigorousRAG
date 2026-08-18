"""Owner-scoped durable journal for retrieval interleaving experiments.

The statistical interleaving modules are pure functions.  This journal supplies the
multi-tenant durability boundary needed by an online/offline experiment runner without
storing raw query text or document content.  Impressions and outcomes are append-only,
idempotent by content identity, and may only be read/exported through their owner-scoped
experiment id.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.retrieval_interleaving import InterleavedItem, InterleavingImpression, InterleavingOutcome, InterleavingSpec, RankedIdentity


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _identifier(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _timestamp(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


@dataclass(frozen=True)
class OwnerScopedInterleavingExperiment:
    owner_id: str
    spec: InterleavingSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _identifier(self.owner_id, "owner_id"))
        if not isinstance(self.spec, InterleavingSpec):
            raise ValueError("spec must be InterleavingSpec")

    @property
    def experiment_id(self) -> str:
        return _digest({
            "schema": "rigorousrag-owner-interleaving-experiment/v1",
            "owner_id": self.owner_id,
            "spec_sha256": self.spec.spec_sha256,
        })


@dataclass(frozen=True)
class InterleavingJournalExport:
    owner_id: str
    experiment_id: str
    spec: InterleavingSpec
    impressions: tuple[InterleavingImpression, ...]
    outcomes: tuple[InterleavingOutcome, ...]
    export_sha256: str


class SQLiteInterleavingJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS interleaving_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS interleaving_impressions (
                    experiment_id TEXT NOT NULL,
                    impression_sha256 TEXT PRIMARY KEY,
                    query_sha256 TEXT NOT NULL,
                    impression_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL,
                    UNIQUE(experiment_id, query_sha256, impression_index),
                    FOREIGN KEY(experiment_id) REFERENCES interleaving_experiments(experiment_id)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS interleaving_outcomes (
                    impression_sha256 TEXT PRIMARY KEY,
                    outcome_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL,
                    FOREIGN KEY(impression_sha256) REFERENCES interleaving_impressions(impression_sha256)
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS interleaving_owner_idx ON interleaving_experiments(owner_id, experiment_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS interleaving_impression_exp_idx ON interleaving_impressions(experiment_id, query_sha256, impression_index)")

    @staticmethod
    def _spec_json(spec: InterleavingSpec) -> str:
        return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _impression_json(impression: InterleavingImpression) -> str:
        return json.dumps({
            "spec_sha256": impression.spec_sha256,
            "query_sha256": impression.query_sha256,
            "impression_index": impression.impression_index,
            "ranking_a_sha256": impression.ranking_a_sha256,
            "ranking_b_sha256": impression.ranking_b_sha256,
            "items": [asdict(item) for item in impression.items],
            "impression_sha256": impression.impression_sha256,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _outcome_json(outcome: InterleavingOutcome) -> str:
        return json.dumps(asdict(outcome), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def ensure_experiment(self, experiment: OwnerScopedInterleavingExperiment, *, now: float) -> str:
        if not isinstance(experiment, OwnerScopedInterleavingExperiment):
            raise ValueError("experiment must be OwnerScopedInterleavingExperiment")
        timestamp = _timestamp(now, "now")
        payload = self._spec_json(experiment.spec)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner_id, spec_json, spec_sha256 FROM interleaving_experiments WHERE experiment_id=?", (experiment.experiment_id,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO interleaving_experiments(experiment_id,owner_id,spec_json,spec_sha256,created_at) VALUES(?,?,?,?,?)",
                    (experiment.experiment_id, experiment.owner_id, payload, experiment.spec.spec_sha256, timestamp),
                )
            elif row["owner_id"] != experiment.owner_id or row["spec_json"] != payload or row["spec_sha256"] != experiment.spec.spec_sha256:
                raise RuntimeError("durable interleaving experiment identity collision")
        return experiment.experiment_id

    def _assert_owner(self, connection: sqlite3.Connection, owner_id: str, experiment_id: str) -> sqlite3.Row:
        owner = _identifier(owner_id, "owner_id")
        row = connection.execute("SELECT owner_id, spec_json, spec_sha256 FROM interleaving_experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError("interleaving experiment not found")
        if row["owner_id"] != owner:
            raise PermissionError("interleaving experiment owner mismatch")
        return row

    def record_impression(self, *, owner_id: str, experiment_id: str, impression: InterleavingImpression, now: float) -> str:
        if not isinstance(impression, InterleavingImpression):
            raise ValueError("impression must be InterleavingImpression")
        timestamp = _timestamp(now, "now")
        payload = self._impression_json(impression)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            experiment = self._assert_owner(connection, owner_id, experiment_id)
            if impression.spec_sha256 != experiment["spec_sha256"]:
                raise ValueError("impression does not belong to durable experiment spec")
            row = connection.execute("SELECT experiment_id,payload_json FROM interleaving_impressions WHERE impression_sha256=?", (impression.impression_sha256,)).fetchone()
            if row is not None:
                if row["experiment_id"] != experiment_id or row["payload_json"] != payload:
                    raise RuntimeError("impression identity collision")
                return impression.impression_sha256
            try:
                connection.execute(
                    "INSERT INTO interleaving_impressions(experiment_id,impression_sha256,query_sha256,impression_index,payload_json,recorded_at) VALUES(?,?,?,?,?,?)",
                    (experiment_id, impression.impression_sha256, impression.query_sha256, impression.impression_index, payload, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("query/impression index already belongs to a different randomized impression") from exc
        return impression.impression_sha256

    def record_outcome(self, *, owner_id: str, experiment_id: str, outcome: InterleavingOutcome, now: float) -> str:
        if not isinstance(outcome, InterleavingOutcome):
            raise ValueError("outcome must be InterleavingOutcome")
        timestamp = _timestamp(now, "now")
        payload = self._outcome_json(outcome)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_owner(connection, owner_id, experiment_id)
            impression = connection.execute("SELECT experiment_id FROM interleaving_impressions WHERE impression_sha256=?", (outcome.impression_sha256,)).fetchone()
            if impression is None or impression["experiment_id"] != experiment_id:
                raise ValueError("outcome impression does not belong to owner-scoped experiment")
            row = connection.execute("SELECT outcome_sha256,payload_json FROM interleaving_outcomes WHERE impression_sha256=?", (outcome.impression_sha256,)).fetchone()
            if row is not None:
                if row["outcome_sha256"] != outcome.outcome_sha256 or row["payload_json"] != payload:
                    raise RuntimeError("outcome for impression is immutable and already recorded")
                return outcome.outcome_sha256
            connection.execute(
                "INSERT INTO interleaving_outcomes(impression_sha256,outcome_sha256,payload_json,recorded_at) VALUES(?,?,?,?)",
                (outcome.impression_sha256, outcome.outcome_sha256, payload, timestamp),
            )
        return outcome.outcome_sha256

    @staticmethod
    def _decode_spec(payload: str) -> InterleavingSpec:
        return InterleavingSpec(**json.loads(payload))

    @staticmethod
    def _decode_impression(payload: str) -> InterleavingImpression:
        raw = json.loads(payload)
        items = []
        for row in raw["items"]:
            item_raw = row["item"]
            items.append(InterleavedItem(row["position"], RankedIdentity(**item_raw), row["contributed_by"], row["source_rank"]))
        return InterleavingImpression(
            raw["spec_sha256"], raw["query_sha256"], raw["impression_index"], raw["ranking_a_sha256"], raw["ranking_b_sha256"], tuple(items), raw["impression_sha256"]
        )

    @staticmethod
    def _decode_outcome(payload: str) -> InterleavingOutcome:
        raw = json.loads(payload)
        return InterleavingOutcome(raw["impression_sha256"], tuple(raw["engaged_positions"]), raw["outcome_sha256"])

    def export_complete_evidence(self, *, owner_id: str, experiment_id: str) -> InterleavingJournalExport:
        owner = _identifier(owner_id, "owner_id")
        with self._connect() as connection:
            experiment = self._assert_owner(connection, owner, experiment_id)
            rows = connection.execute(
                """SELECT i.payload_json AS impression_json, o.payload_json AS outcome_json
                   FROM interleaving_impressions i
                   JOIN interleaving_outcomes o ON o.impression_sha256=i.impression_sha256
                   WHERE i.experiment_id=? ORDER BY i.query_sha256,i.impression_index,i.impression_sha256""",
                (experiment_id,),
            ).fetchall()
        impressions = tuple(self._decode_impression(row["impression_json"]) for row in rows)
        outcomes = tuple(self._decode_outcome(row["outcome_json"]) for row in rows)
        pairs = tuple((impression.impression_sha256, outcome.outcome_sha256) for impression, outcome in zip(impressions, outcomes))
        export_digest = _digest({
            "schema": "rigorousrag-interleaving-journal-export/v1",
            "owner_id": owner,
            "experiment_id": experiment_id,
            "spec_sha256": experiment["spec_sha256"],
            "pairs": pairs,
        })
        return InterleavingJournalExport(owner, experiment_id, self._decode_spec(experiment["spec_json"]), impressions, outcomes, export_digest)


__all__ = ["InterleavingJournalExport", "OwnerScopedInterleavingExperiment", "SQLiteInterleavingJournal"]
