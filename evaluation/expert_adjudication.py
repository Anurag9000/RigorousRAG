"""Durable expert adjudication and privacy-safe gold-label production.

The existing :mod:`evaluation.expert_review` module measures agreement on review data.
This module owns the missing lifecycle: immutable evidence-bound cases, independent
review assignments, monotonic-fenced reviewer claims, append-only judgments/corrections,
quorum/conflict policy, adjudicator escalation, immutable resolution receipts, reopened
correction rounds, and digest-only gold-label exports.

The control journal stores no raw query, answer, document, image, evidence or rationale
content. Callers retain those artifacts in their authoritative stores and bind them by
SHA-256.
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
from typing import Any, Sequence

from tools.security import normalize_owner_id

_HEX = frozenset("0123456789abcdef")
_ROLES = frozenset({"reviewer", "adjudicator"})
_CASE_STATES = frozenset({"open", "needs_adjudication", "resolved"})
_MAX_LABELS = 1_000
_MAX_EVIDENCE = 10_000
_MAX_LEASE_SECONDS = 7 * 24 * 60 * 60
_MAX_EXPORT_RECORDS = 1_000_000


def _text(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


def _unit(value: Any, label: str) -> float:
    selected = _timestamp(value, label)
    if selected > 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class LabelSchema:
    task_id: str
    schema_version: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", 300))
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version", 200))
        labels = tuple(_text(label, "label", 300) for label in self.labels)
        if not 2 <= len(labels) <= _MAX_LABELS or len(set(labels)) != len(labels):
            raise ValueError("labels must contain 2-1000 unique values")
        object.__setattr__(self, "labels", labels)

    @property
    def schema_sha256(self) -> str:
        return _digest({"contract": "rigorousrag-label-schema-v1", **asdict(self)})


@dataclass(frozen=True)
class AdjudicationPolicy:
    minimum_independent_reviews: int = 2
    automatic_consensus_fraction: float = 1.0
    minimum_adjudicator_confidence: float = 0.50
    allow_automatic_resolution: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_independent_reviews, bool)
            or not isinstance(self.minimum_independent_reviews, int)
            or not 2 <= self.minimum_independent_reviews <= 100
        ):
            raise ValueError("minimum_independent_reviews must be between 2 and 100")
        object.__setattr__(self, "automatic_consensus_fraction", _unit(self.automatic_consensus_fraction, "automatic_consensus_fraction"))
        if self.automatic_consensus_fraction <= 0.5:
            raise ValueError("automatic_consensus_fraction must exceed 0.5")
        object.__setattr__(self, "minimum_adjudicator_confidence", _unit(self.minimum_adjudicator_confidence, "minimum_adjudicator_confidence"))
        if not isinstance(self.allow_automatic_resolution, bool):
            raise ValueError("allow_automatic_resolution must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({"contract": "rigorousrag-adjudication-policy-v1", **asdict(self)})


@dataclass(frozen=True)
class AdjudicationCase:
    case_id: str
    owner_id: str
    item_sha256: str
    evidence_sha256: tuple[str, ...]
    schema: LabelSchema
    round_number: int
    parent_case_id: str | None = None
    reopen_reason_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _sha(self.case_id, "case_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "item_sha256", _sha(self.item_sha256, "item_sha256"))
        evidence = tuple(_sha(value, "evidence_sha256") for value in self.evidence_sha256)
        if not evidence or len(evidence) > _MAX_EVIDENCE or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_sha256 must be a non-empty unique bounded tuple")
        object.__setattr__(self, "evidence_sha256", tuple(sorted(evidence)))
        if not isinstance(self.schema, LabelSchema):
            raise ValueError("schema must be LabelSchema")
        if isinstance(self.round_number, bool) or not isinstance(self.round_number, int) or self.round_number < 1:
            raise ValueError("round_number must be positive")
        if self.parent_case_id is not None:
            object.__setattr__(self, "parent_case_id", _sha(self.parent_case_id, "parent_case_id"))
        if self.reopen_reason_sha256 is not None:
            object.__setattr__(self, "reopen_reason_sha256", _sha(self.reopen_reason_sha256, "reopen_reason_sha256"))
        if self.round_number == 1 and (self.parent_case_id is not None or self.reopen_reason_sha256 is not None):
            raise ValueError("first adjudication round may not have reopen lineage")
        if self.round_number > 1 and (self.parent_case_id is None or self.reopen_reason_sha256 is None):
            raise ValueError("reopened adjudication round requires parent and reason digests")


@dataclass(frozen=True)
class CaseRecord:
    case: AdjudicationCase
    state: str
    revision: int
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        if self.state not in _CASE_STATES:
            raise ValueError("invalid adjudication case state")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be non-negative")
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)


@dataclass(frozen=True)
class ReviewClaim:
    case_id: str
    reviewer_id: str
    role: str
    fencing_token: int
    expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _sha(self.case_id, "case_id"))
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 300))
        role = _text(self.role, "role", 32).lower()
        if role not in _ROLES:
            raise ValueError("unsupported review role")
        object.__setattr__(self, "role", role)
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expires_at"))


@dataclass(frozen=True)
class ExpertJudgment:
    judgment_id: str
    case_id: str
    reviewer_id: str
    role: str
    reviewer_revision: int
    label: str
    confidence: float
    rationale_sha256: str | None
    supersedes_judgment_id: str | None
    submitted_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "judgment_id", _sha(self.judgment_id, "judgment_id"))
        object.__setattr__(self, "case_id", _sha(self.case_id, "case_id"))
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 300))
        role = _text(self.role, "role", 32).lower()
        if role not in _ROLES:
            raise ValueError("unsupported review role")
        object.__setattr__(self, "role", role)
        if isinstance(self.reviewer_revision, bool) or not isinstance(self.reviewer_revision, int) or self.reviewer_revision < 1:
            raise ValueError("reviewer_revision must be positive")
        object.__setattr__(self, "label", _text(self.label, "label", 300))
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        if self.rationale_sha256 is not None:
            object.__setattr__(self, "rationale_sha256", _sha(self.rationale_sha256, "rationale_sha256"))
        if self.supersedes_judgment_id is not None:
            object.__setattr__(self, "supersedes_judgment_id", _sha(self.supersedes_judgment_id, "supersedes_judgment_id"))
        object.__setattr__(self, "submitted_at", _timestamp(self.submitted_at, "submitted_at"))


@dataclass(frozen=True)
class ResolutionReceipt:
    resolution_id: str
    case_id: str
    owner_id: str
    label: str
    method: str
    reviewer_count: int
    active_judgment_sha256: str
    schema_sha256: str
    policy_sha256: str
    resolved_at: float

    def __post_init__(self) -> None:
        for name in ("resolution_id", "case_id", "active_judgment_sha256", "schema_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "label", _text(self.label, "label", 300))
        method = _text(self.method, "method", 64)
        if method not in {"reviewer_consensus", "adjudicator"}:
            raise ValueError("invalid resolution method")
        object.__setattr__(self, "method", method)
        if isinstance(self.reviewer_count, bool) or not isinstance(self.reviewer_count, int) or self.reviewer_count < 2:
            raise ValueError("reviewer_count must be at least two")
        object.__setattr__(self, "resolved_at", _timestamp(self.resolved_at, "resolved_at"))


@dataclass(frozen=True)
class GoldLabelRecord:
    case_id: str
    item_sha256: str
    evidence_set_sha256: str
    label: str
    schema_sha256: str
    resolution_id: str
    round_number: int

    def __post_init__(self) -> None:
        for name in ("case_id", "item_sha256", "evidence_set_sha256", "schema_sha256", "resolution_id"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "label", _text(self.label, "label", 300))
        if isinstance(self.round_number, bool) or not isinstance(self.round_number, int) or self.round_number < 1:
            raise ValueError("round_number must be positive")


@dataclass(frozen=True)
class GoldLabelManifest:
    owner_id: str
    task_id: str
    records: tuple[GoldLabelRecord, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id", 300))
        if not self.records or len(self.records) > _MAX_EXPORT_RECORDS:
            raise ValueError("gold label records must be non-empty and bounded")
        if len({record.item_sha256 for record in self.records}) != len(self.records):
            raise ValueError("gold label manifest may contain only one current label per item")
        object.__setattr__(self, "manifest_sha256", _sha(self.manifest_sha256, "manifest_sha256"))


class ExpertAdjudicationStore:
    """SQLite-backed append-only judgments with case revision CAS and review fencing."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        selected = Path(os.fspath(path))
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        selected.parent.mkdir(parents=True, exist_ok=True)
        self.path = selected.absolute()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS adjudication_case (case_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,item_sha256 TEXT NOT NULL,evidence_json TEXT NOT NULL,schema_json TEXT NOT NULL,schema_sha256 TEXT NOT NULL,round_number INTEGER NOT NULL,parent_case_id TEXT,reopen_reason_sha256 TEXT,state TEXT NOT NULL,revision INTEGER NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL,UNIQUE(owner_id,item_sha256,round_number))"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS review_claim (case_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,role TEXT NOT NULL,fencing_token INTEGER NOT NULL DEFAULT 0,expires_at REAL,PRIMARY KEY(case_id,reviewer_id,role))"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS expert_judgment (judgment_id TEXT PRIMARY KEY,case_id TEXT NOT NULL,reviewer_id TEXT NOT NULL,role TEXT NOT NULL,reviewer_revision INTEGER NOT NULL,label TEXT NOT NULL,confidence REAL NOT NULL,rationale_sha256 TEXT,supersedes_judgment_id TEXT,submitted_at REAL NOT NULL,UNIQUE(case_id,reviewer_id,role,reviewer_revision))"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS adjudication_resolution (resolution_id TEXT PRIMARY KEY,case_id TEXT NOT NULL UNIQUE,owner_id TEXT NOT NULL,label TEXT NOT NULL,method TEXT NOT NULL,reviewer_count INTEGER NOT NULL,active_judgment_sha256 TEXT NOT NULL,schema_sha256 TEXT NOT NULL,policy_sha256 TEXT NOT NULL,resolved_at REAL NOT NULL)"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_adjudication_item ON adjudication_case(owner_id,item_sha256,round_number)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_judgment_case ON expert_judgment(case_id,reviewer_id,role,reviewer_revision)")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _schema_payload(schema: LabelSchema) -> str:
        return _canonical(asdict(schema)).decode("utf-8")

    @staticmethod
    def _case_from_row(row: tuple[Any, ...] | None) -> CaseRecord | None:
        if row is None:
            return None
        schema_raw = json.loads(row[4])
        schema = LabelSchema(schema_raw["task_id"], schema_raw["schema_version"], tuple(schema_raw["labels"]))
        case = AdjudicationCase(
            case_id=row[0], owner_id=row[1], item_sha256=row[2], evidence_sha256=tuple(json.loads(row[3])), schema=schema,
            round_number=int(row[6]), parent_case_id=row[7], reopen_reason_sha256=row[8],
        )
        if schema.schema_sha256 != row[5]:
            raise RuntimeError("persisted label schema digest is corrupt")
        return CaseRecord(case, str(row[9]), int(row[10]), float(row[11]), float(row[12]))

    def get_case(self, case_id: str) -> CaseRecord | None:
        selected = _sha(case_id, "case_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                (selected,),
            ).fetchone()
        return self._case_from_row(row)

    @staticmethod
    def _initial_case_id(owner: str, item: str, evidence: Sequence[str], schema: LabelSchema) -> str:
        return _digest({"contract": "rigorousrag-adjudication-case-v1", "owner_id": owner, "item_sha256": item, "evidence_sha256": sorted(evidence), "schema_sha256": schema.schema_sha256, "round_number": 1})

    def create_case(self, *, owner_id: str, item_sha256: str, evidence_sha256: Sequence[str], schema: LabelSchema, now: float) -> CaseRecord:
        owner = normalize_owner_id(owner_id)
        item = _sha(item_sha256, "item_sha256")
        evidence = tuple(sorted(_sha(value, "evidence_sha256") for value in evidence_sha256))
        if not evidence or len(evidence) > _MAX_EVIDENCE or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_sha256 must be non-empty, unique and bounded")
        if not isinstance(schema, LabelSchema):
            raise ValueError("schema must be LabelSchema")
        instant = _timestamp(now, "now")
        case_id = self._initial_case_id(owner, item, evidence, schema)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing_row = self._connection.execute(
                    "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                    (case_id,),
                ).fetchone()
                existing = self._case_from_row(existing_row)
                if existing is not None:
                    self._connection.execute("COMMIT")
                    return existing
                prior = self._connection.execute(
                    "SELECT case_id FROM adjudication_case WHERE owner_id=? AND item_sha256=? ORDER BY round_number DESC LIMIT 1",
                    (owner, item),
                ).fetchone()
                if prior is not None:
                    raise ValueError("item already has an adjudication lineage; reopen the latest resolved case instead")
                self._connection.execute(
                    "INSERT INTO adjudication_case VALUES(?,?,?,?,?,?,?,?,?,'open',0,?,?)",
                    (case_id, owner, item, _canonical(list(evidence)).decode("utf-8"), self._schema_payload(schema), schema.schema_sha256, 1, None, None, instant, instant),
                )
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        created = self.get_case(case_id)
        if created is None:
            raise RuntimeError("adjudication case disappeared after creation")
        return created

    def claim_review(self, case_id: str, *, reviewer_id: str, role: str, now: float, lease_seconds: float = 86_400.0) -> ReviewClaim:
        selected_case = _sha(case_id, "case_id")
        reviewer = _text(reviewer_id, "reviewer_id", 300)
        selected_role = _text(role, "role", 32).lower()
        if selected_role not in _ROLES:
            raise ValueError("unsupported review role")
        instant = _timestamp(now, "now")
        lease = _timestamp(lease_seconds, "lease_seconds")
        if not 0.0 < lease <= _MAX_LEASE_SECONDS:
            raise ValueError("lease_seconds is invalid")
        expiry = instant + lease
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                case_row = self._connection.execute(
                    "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                    (selected_case,),
                ).fetchone()
                case = self._case_from_row(case_row)
                if case is None or case.state == "resolved":
                    raise ValueError("review claim requires a live unresolved case")
                conflicting_role = "adjudicator" if selected_role == "reviewer" else "reviewer"
                if self._connection.execute(
                    "SELECT 1 FROM expert_judgment WHERE case_id=? AND reviewer_id=? AND role=? LIMIT 1",
                    (selected_case, reviewer, conflicting_role),
                ).fetchone() is not None:
                    raise ValueError("reviewer may not switch between reviewer and adjudicator on one case")
                if selected_role == "adjudicator":
                    live_other = self._connection.execute(
                        "SELECT reviewer_id FROM review_claim WHERE case_id=? AND role='adjudicator' AND reviewer_id<>? AND expires_at>? LIMIT 1",
                        (selected_case, reviewer, instant),
                    ).fetchone()
                    if live_other is not None:
                        raise RuntimeError("another adjudicator holds the live case claim")
                row = self._connection.execute(
                    "SELECT fencing_token,expires_at FROM review_claim WHERE case_id=? AND reviewer_id=? AND role=?",
                    (selected_case, reviewer, selected_role),
                ).fetchone()
                if row is not None and row[1] is not None and float(row[1]) > instant:
                    raise RuntimeError("reviewer already has a live claim")
                token = 1 if row is None else int(row[0]) + 1
                if row is None:
                    self._connection.execute(
                        "INSERT INTO review_claim(case_id,reviewer_id,role,fencing_token,expires_at) VALUES(?,?,?,?,?)",
                        (selected_case, reviewer, selected_role, token, expiry),
                    )
                else:
                    self._connection.execute(
                        "UPDATE review_claim SET fencing_token=?,expires_at=? WHERE case_id=? AND reviewer_id=? AND role=?",
                        (token, expiry, selected_case, reviewer, selected_role),
                    )
                self._connection.execute("COMMIT")
                return ReviewClaim(selected_case, reviewer, selected_role, token, expiry)
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _assert_claim(self, claim: ReviewClaim, *, now: float) -> None:
        instant = _timestamp(now, "now")
        with self._lock:
            row = self._connection.execute(
                "SELECT fencing_token,expires_at FROM review_claim WHERE case_id=? AND reviewer_id=? AND role=?",
                (claim.case_id, claim.reviewer_id, claim.role),
            ).fetchone()
        if row is None or int(row[0]) != claim.fencing_token or row[1] is None or float(row[1]) <= instant:
            raise RuntimeError("review claim is expired or fenced")

    @staticmethod
    def _judgment_from_row(row: tuple[Any, ...]) -> ExpertJudgment:
        return ExpertJudgment(
            judgment_id=row[0], case_id=row[1], reviewer_id=row[2], role=row[3], reviewer_revision=int(row[4]),
            label=row[5], confidence=float(row[6]), rationale_sha256=row[7], supersedes_judgment_id=row[8], submitted_at=float(row[9]),
        )

    def judgments(self, case_id: str) -> tuple[ExpertJudgment, ...]:
        selected = _sha(case_id, "case_id")
        with self._lock:
            rows = self._connection.execute(
                "SELECT judgment_id,case_id,reviewer_id,role,reviewer_revision,label,confidence,rationale_sha256,supersedes_judgment_id,submitted_at FROM expert_judgment WHERE case_id=? ORDER BY reviewer_id,role,reviewer_revision",
                (selected,),
            ).fetchall()
        return tuple(self._judgment_from_row(row) for row in rows)

    def active_judgments(self, case_id: str) -> tuple[ExpertJudgment, ...]:
        values = self.judgments(case_id)
        latest: dict[tuple[str, str], ExpertJudgment] = {}
        for value in values:
            key = (value.reviewer_id, value.role)
            if key not in latest or value.reviewer_revision > latest[key].reviewer_revision:
                latest[key] = value
        return tuple(sorted(latest.values(), key=lambda value: (value.role, value.reviewer_id)))

    def submit_judgment(
        self,
        claim: ReviewClaim,
        *,
        label: str,
        confidence: float,
        rationale_sha256: str | None,
        expected_case_revision: int,
        now: float,
        supersedes_judgment_id: str | None = None,
    ) -> ExpertJudgment:
        if not isinstance(claim, ReviewClaim):
            raise ValueError("claim must be ReviewClaim")
        selected_label = _text(label, "label", 300)
        selected_confidence = _unit(confidence, "confidence")
        rationale = None if rationale_sha256 is None else _sha(rationale_sha256, "rationale_sha256")
        supersedes = None if supersedes_judgment_id is None else _sha(supersedes_judgment_id, "supersedes_judgment_id")
        if isinstance(expected_case_revision, bool) or not isinstance(expected_case_revision, int) or expected_case_revision < 0:
            raise ValueError("expected_case_revision is invalid")
        instant = _timestamp(now, "now")
        self._assert_claim(claim, now=instant)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                case_row = self._connection.execute(
                    "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                    (claim.case_id,),
                ).fetchone()
                case = self._case_from_row(case_row)
                if case is None or case.state == "resolved":
                    raise ValueError("judgment requires an unresolved case")
                if case.revision != expected_case_revision:
                    raise RuntimeError("case revision changed before judgment submission")
                if selected_label not in case.case.schema.labels:
                    raise ValueError("judgment label is outside the governed schema")
                prior_row = self._connection.execute(
                    "SELECT judgment_id,case_id,reviewer_id,role,reviewer_revision,label,confidence,rationale_sha256,supersedes_judgment_id,submitted_at FROM expert_judgment WHERE case_id=? AND reviewer_id=? AND role=? ORDER BY reviewer_revision DESC LIMIT 1",
                    (claim.case_id, claim.reviewer_id, claim.role),
                ).fetchone()
                prior = None if prior_row is None else self._judgment_from_row(prior_row)
                if prior is None:
                    if supersedes is not None:
                        raise ValueError("first reviewer judgment may not supersede another judgment")
                    reviewer_revision = 1
                else:
                    if supersedes != prior.judgment_id:
                        raise ValueError("judgment correction must explicitly supersede the reviewer's current judgment")
                    reviewer_revision = prior.reviewer_revision + 1
                identity = {
                    "contract": "rigorousrag-expert-judgment-v1", "case_id": claim.case_id,
                    "reviewer_id": claim.reviewer_id, "role": claim.role, "reviewer_revision": reviewer_revision,
                    "label": selected_label, "confidence": selected_confidence, "rationale_sha256": rationale,
                    "supersedes_judgment_id": supersedes,
                }
                judgment_id = _digest(identity)
                existing = self._connection.execute(
                    "SELECT judgment_id,case_id,reviewer_id,role,reviewer_revision,label,confidence,rationale_sha256,supersedes_judgment_id,submitted_at FROM expert_judgment WHERE judgment_id=?",
                    (judgment_id,),
                ).fetchone()
                if existing is not None:
                    self._connection.execute("COMMIT")
                    return self._judgment_from_row(existing)
                self._connection.execute(
                    "INSERT INTO expert_judgment VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (judgment_id, claim.case_id, claim.reviewer_id, claim.role, reviewer_revision, selected_label, selected_confidence, rationale, supersedes, instant),
                )
                updated = self._connection.execute(
                    "UPDATE adjudication_case SET revision=revision+1,updated_at=? WHERE case_id=? AND revision=? AND state<>'resolved'",
                    (instant, claim.case_id, expected_case_revision),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("case revision compare-and-swap failed")
                self._connection.execute("COMMIT")
                return ExpertJudgment(judgment_id, claim.case_id, claim.reviewer_id, claim.role, reviewer_revision, selected_label, selected_confidence, rationale, supersedes, instant)
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _active_digest(values: Sequence[ExpertJudgment]) -> str:
        return _digest(
            {
                "contract": "rigorousrag-active-expert-judgments-v1",
                "judgments": [
                    {"judgment_id": value.judgment_id, "reviewer_id": value.reviewer_id, "role": value.role,
                     "reviewer_revision": value.reviewer_revision, "label": value.label, "confidence": value.confidence,
                     "rationale_sha256": value.rationale_sha256}
                    for value in values
                ],
            }
        )

    def reconcile_case(self, case_id: str, *, policy: AdjudicationPolicy, expected_case_revision: int, now: float) -> CaseRecord:
        selected = _sha(case_id, "case_id")
        if not isinstance(policy, AdjudicationPolicy):
            raise ValueError("policy must be AdjudicationPolicy")
        if isinstance(expected_case_revision, bool) or not isinstance(expected_case_revision, int) or expected_case_revision < 0:
            raise ValueError("expected_case_revision is invalid")
        instant = _timestamp(now, "now")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                case_row = self._connection.execute(
                    "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                    (selected,),
                ).fetchone()
                case = self._case_from_row(case_row)
                if case is None:
                    raise KeyError(selected)
                if case.revision != expected_case_revision:
                    raise RuntimeError("case revision changed before reconciliation")
                if case.state == "resolved":
                    self._connection.execute("COMMIT")
                    return case
                judgment_rows = self._connection.execute(
                    "SELECT judgment_id,case_id,reviewer_id,role,reviewer_revision,label,confidence,rationale_sha256,supersedes_judgment_id,submitted_at FROM expert_judgment WHERE case_id=? ORDER BY reviewer_id,role,reviewer_revision",
                    (selected,),
                ).fetchall()
                latest: dict[tuple[str, str], ExpertJudgment] = {}
                for row in judgment_rows:
                    value = self._judgment_from_row(row)
                    key = (value.reviewer_id, value.role)
                    if key not in latest or value.reviewer_revision > latest[key].reviewer_revision:
                        latest[key] = value
                active = tuple(sorted(latest.values(), key=lambda value: (value.role, value.reviewer_id)))
                reviewers = [value for value in active if value.role == "reviewer"]
                adjudicators = [value for value in active if value.role == "adjudicator"]
                if len({value.reviewer_id for value in reviewers}) != len(reviewers):
                    raise RuntimeError("active reviewer identities are not independent")
                new_state = "open"
                resolution_label: str | None = None
                method: str | None = None
                if len(reviewers) >= policy.minimum_independent_reviews:
                    counts: dict[str, int] = {}
                    for value in reviewers:
                        counts[value.label] = counts.get(value.label, 0) + 1
                    maximum = max(counts.values())
                    leaders = sorted(label for label, count in counts.items() if count == maximum)
                    fraction = maximum / len(reviewers)
                    if policy.allow_automatic_resolution and len(leaders) == 1 and fraction >= policy.automatic_consensus_fraction:
                        resolution_label = leaders[0]
                        method = "reviewer_consensus"
                    elif adjudicators:
                        if len(adjudicators) != 1:
                            raise RuntimeError("at most one active adjudicator is allowed")
                        adjudicator = adjudicators[0]
                        if adjudicator.confidence >= policy.minimum_adjudicator_confidence:
                            resolution_label = adjudicator.label
                            method = "adjudicator"
                        else:
                            new_state = "needs_adjudication"
                    else:
                        new_state = "needs_adjudication"
                if resolution_label is not None and method is not None:
                    active_digest = self._active_digest(active)
                    payload = {
                        "contract": "rigorousrag-adjudication-resolution-v1", "case_id": selected,
                        "owner_id": case.case.owner_id, "label": resolution_label, "method": method,
                        "reviewer_count": len(reviewers), "active_judgment_sha256": active_digest,
                        "schema_sha256": case.case.schema.schema_sha256, "policy_sha256": policy.policy_sha256,
                    }
                    resolution_id = _digest(payload)
                    self._connection.execute(
                        "INSERT OR IGNORE INTO adjudication_resolution VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (resolution_id, selected, case.case.owner_id, resolution_label, method, len(reviewers), active_digest, case.case.schema.schema_sha256, policy.policy_sha256, instant),
                    )
                    new_state = "resolved"
                if new_state != case.state:
                    updated = self._connection.execute(
                        "UPDATE adjudication_case SET state=?,revision=revision+1,updated_at=? WHERE case_id=? AND revision=?",
                        (new_state, instant, selected, expected_case_revision),
                    )
                    if updated.rowcount != 1:
                        raise RuntimeError("case reconciliation compare-and-swap failed")
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        current = self.get_case(selected)
        if current is None:
            raise RuntimeError("adjudication case disappeared after reconciliation")
        return current

    def resolution(self, case_id: str) -> ResolutionReceipt | None:
        selected = _sha(case_id, "case_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT resolution_id,case_id,owner_id,label,method,reviewer_count,active_judgment_sha256,schema_sha256,policy_sha256,resolved_at FROM adjudication_resolution WHERE case_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            return None
        return ResolutionReceipt(row[0], row[1], row[2], row[3], row[4], int(row[5]), row[6], row[7], row[8], float(row[9]))

    def reopen_resolved_case(self, case_id: str, *, reason_sha256: str, actor_id: str, now: float) -> CaseRecord:
        parent_id = _sha(case_id, "case_id")
        reason = _sha(reason_sha256, "reason_sha256")
        actor = _text(actor_id, "actor_id", 300)
        instant = _timestamp(now, "now")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                parent_row = self._connection.execute(
                    "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                    (parent_id,),
                ).fetchone()
                parent = self._case_from_row(parent_row)
                resolution_row = self._connection.execute("SELECT 1 FROM adjudication_resolution WHERE case_id=?", (parent_id,)).fetchone()
                if parent is None or parent.state != "resolved" or resolution_row is None:
                    raise ValueError("only a resolved case can be reopened")
                existing_child = self._connection.execute(
                    "SELECT case_id FROM adjudication_case WHERE parent_case_id=? ORDER BY round_number DESC LIMIT 1",
                    (parent_id,),
                ).fetchone()
                if existing_child is not None:
                    child_row = self._connection.execute(
                        "SELECT case_id,owner_id,item_sha256,evidence_json,schema_json,schema_sha256,round_number,parent_case_id,reopen_reason_sha256,state,revision,created_at,updated_at FROM adjudication_case WHERE case_id=?",
                        (str(existing_child[0]),),
                    ).fetchone()
                    child = self._case_from_row(child_row)
                    self._connection.execute("COMMIT")
                    if child is None:
                        raise RuntimeError("reopened child case disappeared")
                    return child
                round_number = parent.case.round_number + 1
                child_id = _digest(
                    {"contract": "rigorousrag-adjudication-reopen-v1", "parent_case_id": parent_id,
                     "round_number": round_number, "reason_sha256": reason,
                     "actor_sha256": hashlib.sha256(actor.encode("utf-8")).hexdigest()}
                )
                self._connection.execute(
                    "INSERT INTO adjudication_case VALUES(?,?,?,?,?,?,?,?,?,'open',0,?,?)",
                    (child_id, parent.case.owner_id, parent.case.item_sha256, _canonical(list(parent.case.evidence_sha256)).decode("utf-8"), self._schema_payload(parent.case.schema), parent.case.schema.schema_sha256, round_number, parent_id, reason, instant, instant),
                )
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        child = self.get_case(child_id)
        if child is None:
            raise RuntimeError("reopened adjudication case disappeared")
        return child

    def _latest_cases_for_owner(self, owner_id: str) -> tuple[CaseRecord, ...]:
        owner = normalize_owner_id(owner_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT c.case_id,c.owner_id,c.item_sha256,c.evidence_json,c.schema_json,c.schema_sha256,c.round_number,c.parent_case_id,c.reopen_reason_sha256,c.state,c.revision,c.created_at,c.updated_at FROM adjudication_case c JOIN (SELECT item_sha256,MAX(round_number) AS round_number FROM adjudication_case WHERE owner_id=? GROUP BY item_sha256) latest ON c.item_sha256=latest.item_sha256 AND c.round_number=latest.round_number WHERE c.owner_id=? ORDER BY c.item_sha256",
                (owner, owner),
            ).fetchall()
        return tuple(value for row in rows if (value := self._case_from_row(row)) is not None)

    def build_gold_manifest(self, *, owner_id: str, task_id: str) -> GoldLabelManifest:
        owner = normalize_owner_id(owner_id)
        task = _text(task_id, "task_id", 300)
        records: list[GoldLabelRecord] = []
        for case in self._latest_cases_for_owner(owner):
            if case.case.schema.task_id != task or case.state != "resolved":
                continue
            resolution = self.resolution(case.case.case_id)
            if resolution is None:
                raise RuntimeError("resolved adjudication case has no immutable resolution receipt")
            evidence_set = _digest({"contract": "rigorousrag-gold-evidence-set-v1", "evidence_sha256": list(case.case.evidence_sha256)})
            records.append(
                GoldLabelRecord(case.case.case_id, case.case.item_sha256, evidence_set, resolution.label, case.case.schema.schema_sha256, resolution.resolution_id, case.case.round_number)
            )
        records.sort(key=lambda value: value.item_sha256)
        if not records:
            raise ValueError("no current resolved gold labels exist for owner/task")
        payload = {"contract": "rigorousrag-gold-label-manifest-v1", "owner_id": owner, "task_id": task, "records": [asdict(value) for value in records]}
        return GoldLabelManifest(owner, task, tuple(records), _digest(payload))


def write_gold_manifest(path: str | os.PathLike[str], manifest: GoldLabelManifest) -> Path:
    """Atomically export digest-only current gold labels; no raw review content is emitted."""

    if not isinstance(manifest, GoldLabelManifest):
        raise ValueError("manifest must be GoldLabelManifest")
    destination = Path(os.fspath(path))
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "rigorousrag-gold-label-manifest-v1", "owner_id": manifest.owner_id,
        "task_id": manifest.task_id, "records": [asdict(value) for value in manifest.records],
        "manifest_sha256": manifest.manifest_sha256,
    }
    encoded = _canonical(payload) + b"\n"
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


__all__ = [
    "AdjudicationCase", "AdjudicationPolicy", "CaseRecord", "ExpertAdjudicationStore",
    "ExpertJudgment", "GoldLabelManifest", "GoldLabelRecord", "LabelSchema",
    "ResolutionReceipt", "ReviewClaim", "write_gold_manifest",
]
