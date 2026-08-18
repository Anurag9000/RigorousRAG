"""Durable active-learning to expert-adjudication coordination.

Selection remains a pure function in :mod:`evaluation.active_learning`.  This module
records immutable selection batches and materializes selected candidates into the existing
ExpertAdjudicationStore.  Case creation is idempotent and content-addressed, so a crash
after case creation but before the active-learning mapping can be recovered by retrying
the same batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.active_learning import ActiveLearningBatch, ActiveLearningCandidate
from evaluation.expert_adjudication import CasePolicy, ExpertAdjudicationStore, LabelSchema


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


@dataclass(frozen=True)
class ActiveLearningRoute:
    task_id: str
    schema: LabelSchema
    policy: CasePolicy = CasePolicy()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        if not isinstance(self.schema, LabelSchema):
            raise ValueError("schema must be LabelSchema")
        if not isinstance(self.policy, CasePolicy):
            raise ValueError("policy must be CasePolicy")

    @property
    def route_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-active-learning-route/v1",
                "task_id": self.task_id,
                "label_schema": asdict(self.schema),
                "case_policy": asdict(self.policy),
            }
        )


@dataclass(frozen=True)
class MaterializedActiveLearningCase:
    task_id: str
    item_sha256: str
    candidate_sha256: str
    route_sha256: str
    case_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        for name in ("item_sha256", "candidate_sha256", "route_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id", 1000))


@dataclass(frozen=True)
class ActiveLearningMaterializationReceipt:
    owner_id: str
    batch_sha256: str
    route_set_sha256: str
    cases: tuple[MaterializedActiveLearningCase, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "batch_sha256", _sha(self.batch_sha256, "batch_sha256"))
        object.__setattr__(self, "route_set_sha256", _sha(self.route_set_sha256, "route_set_sha256"))
        cases = tuple(self.cases)
        if len({(row.task_id, row.item_sha256) for row in cases}) != len(cases):
            raise ValueError("materialization receipt contains duplicate task/item identities")
        object.__setattr__(self, "cases", cases)
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match materialization receipt")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-active-learning-materialization/v1",
            "owner_id": self.owner_id,
            "batch_sha256": self.batch_sha256,
            "route_set_sha256": self.route_set_sha256,
            "cases": [asdict(row) for row in self.cases],
        }


class SQLiteActiveLearningJournal:
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
                """CREATE TABLE IF NOT EXISTS active_learning_batches (
                    batch_sha256 TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS active_learning_cases (
                    owner_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    item_sha256 TEXT NOT NULL,
                    candidate_sha256 TEXT NOT NULL,
                    batch_sha256 TEXT NOT NULL,
                    route_sha256 TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, task_id, item_sha256),
                    UNIQUE(case_id),
                    FOREIGN KEY(batch_sha256) REFERENCES active_learning_batches(batch_sha256)
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS active_learning_batch_owner_idx ON active_learning_batches(owner_id,batch_sha256)"
            )

    @staticmethod
    def _batch_json(batch: ActiveLearningBatch) -> str:
        return json.dumps(asdict(batch), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def record_batch(self, batch: ActiveLearningBatch, *, now: float) -> str:
        if not isinstance(batch, ActiveLearningBatch):
            raise ValueError("batch must be ActiveLearningBatch")
        timestamp = _time(now, "now")
        payload = self._batch_json(batch)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id,payload_json FROM active_learning_batches WHERE batch_sha256=?",
                (batch.batch_sha256,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO active_learning_batches(batch_sha256,owner_id,payload_json,recorded_at) VALUES(?,?,?,?)",
                    (batch.batch_sha256, batch.owner_id, payload, timestamp),
                )
            elif row["owner_id"] != batch.owner_id or row["payload_json"] != payload:
                raise RuntimeError("active-learning batch identity collision")
        return batch.batch_sha256

    def blocked_item_keys(self, *, owner_id: str) -> tuple[tuple[str, str], ...]:
        owner = _text(owner_id, "owner_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id,item_sha256 FROM active_learning_cases WHERE owner_id=? ORDER BY task_id,item_sha256",
                (owner,),
            ).fetchall()
        return tuple((row["task_id"], row["item_sha256"]) for row in rows)

    def record_case_mapping(
        self,
        *,
        owner_id: str,
        task_id: str,
        item_sha256: str,
        candidate_sha256: str,
        batch_sha256: str,
        route_sha256: str,
        case_id: str,
        now: float,
    ) -> MaterializedActiveLearningCase:
        owner = _text(owner_id, "owner_id")
        task = _text(task_id, "task_id")
        item = _sha(item_sha256, "item_sha256")
        candidate = _sha(candidate_sha256, "candidate_sha256")
        batch = _sha(batch_sha256, "batch_sha256")
        route = _sha(route_sha256, "route_sha256")
        case = _text(case_id, "case_id", 1000)
        timestamp = _time(now, "now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            batch_row = connection.execute(
                "SELECT owner_id FROM active_learning_batches WHERE batch_sha256=?", (batch,)
            ).fetchone()
            if batch_row is None or batch_row["owner_id"] != owner:
                raise ValueError("case mapping batch does not belong to owner")
            row = connection.execute(
                "SELECT candidate_sha256,batch_sha256,route_sha256,case_id FROM active_learning_cases WHERE owner_id=? AND task_id=? AND item_sha256=?",
                (owner, task, item),
            ).fetchone()
            if row is None:
                try:
                    connection.execute(
                        "INSERT INTO active_learning_cases(owner_id,task_id,item_sha256,candidate_sha256,batch_sha256,route_sha256,case_id,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (owner, task, item, candidate, batch, route, case, timestamp),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError("active-learning case mapping conflicts with an existing case") from exc
            elif (
                row["candidate_sha256"] != candidate
                or row["batch_sha256"] != batch
                or row["route_sha256"] != route
                or row["case_id"] != case
            ):
                raise RuntimeError("active-learning item is already mapped with different immutable intent")
        return MaterializedActiveLearningCase(task, item, candidate, route, case)


def _route_set_sha256(routes: Mapping[str, ActiveLearningRoute]) -> str:
    return _digest(
        {
            "schema": "rigorousrag-active-learning-route-set/v1",
            "routes": tuple(sorted((task_id, route.route_sha256) for task_id, route in routes.items())),
        }
    )


def materialize_active_learning_batch(
    batch: ActiveLearningBatch,
    candidates: Sequence[ActiveLearningCandidate],
    *,
    routes: Mapping[str, ActiveLearningRoute],
    adjudication_store: ExpertAdjudicationStore,
    journal: SQLiteActiveLearningJournal,
    now: float,
) -> ActiveLearningMaterializationReceipt:
    if not isinstance(batch, ActiveLearningBatch):
        raise ValueError("batch must be ActiveLearningBatch")
    if not isinstance(adjudication_store, ExpertAdjudicationStore):
        raise ValueError("adjudication_store must be ExpertAdjudicationStore")
    if not isinstance(journal, SQLiteActiveLearningJournal):
        raise ValueError("journal must be SQLiteActiveLearningJournal")
    timestamp = _time(now, "now")
    candidate_rows = tuple(candidates)
    if any(not isinstance(candidate, ActiveLearningCandidate) for candidate in candidate_rows):
        raise ValueError("candidates contains invalid values")
    candidate_by_sha = {candidate.candidate_sha256: candidate for candidate in candidate_rows}
    if len(candidate_by_sha) != len(candidate_rows):
        raise ValueError("candidates contains duplicate candidate identities")
    if any(candidate.owner_id != batch.owner_id for candidate in candidate_rows):
        raise ValueError("candidate owner differs from active-learning batch owner")
    route_map = dict(routes)
    if any(task_id != route.task_id for task_id, route in route_map.items()):
        raise ValueError("route mapping key must match route.task_id")

    journal.record_batch(batch, now=timestamp)
    materialized: list[MaterializedActiveLearningCase] = []
    for selected in batch.selected:
        candidate = candidate_by_sha.get(selected.candidate_sha256)
        if candidate is None:
            raise ValueError("selected candidate is missing from supplied candidate pool")
        if candidate.task_id != selected.task_id or candidate.item_sha256 != selected.item_sha256:
            raise ValueError("selected candidate identity differs from batch receipt")
        route = route_map.get(candidate.task_id)
        if route is None:
            raise ValueError(f"no adjudication route exists for task {candidate.task_id!r}")
        case = adjudication_store.create_case(
            owner_id=batch.owner_id,
            item_sha256=candidate.item_sha256,
            evidence_sha256=candidate.evidence_sha256s,
            schema=route.schema,
            policy=route.policy,
            now=timestamp,
        )
        materialized.append(
            journal.record_case_mapping(
                owner_id=batch.owner_id,
                task_id=candidate.task_id,
                item_sha256=candidate.item_sha256,
                candidate_sha256=candidate.candidate_sha256,
                batch_sha256=batch.batch_sha256,
                route_sha256=route.route_sha256,
                case_id=case.case.case_id,
                now=timestamp,
            )
        )

    rows = tuple(materialized)
    route_set = _route_set_sha256(route_map)
    payload = {
        "schema": "rigorousrag-active-learning-materialization/v1",
        "owner_id": batch.owner_id,
        "batch_sha256": batch.batch_sha256,
        "route_set_sha256": route_set,
        "cases": [asdict(row) for row in rows],
    }
    return ActiveLearningMaterializationReceipt(
        owner_id=batch.owner_id,
        batch_sha256=batch.batch_sha256,
        route_set_sha256=route_set,
        cases=rows,
        receipt_sha256=_digest(payload),
    )


__all__ = [
    "ActiveLearningMaterializationReceipt",
    "ActiveLearningRoute",
    "MaterializedActiveLearningCase",
    "SQLiteActiveLearningJournal",
    "materialize_active_learning_batch",
]
