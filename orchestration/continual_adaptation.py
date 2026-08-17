"""Durable drift -> build -> benchmark -> promotion orchestration.

RigorousRAG already provides distribution-drift decisions, reproducible training lineage,
verified benchmark datasets, feedback-driven promotion gates, continual-learning metrics,
and a tamper-evident promotion journal.  This module binds those primitives into one
crash-resumable workflow without performing hidden training, dataset acquisition, model
loading, or background scheduling.

External build, benchmark and publication systems are injected.  Every side-effecting
request carries a stable workflow id and must be idempotent.  SQLite persists only
identifiers, digests, bounded metric/decision payloads, stage state and fencing claims;
raw queries, documents, examples and model tensors never enter this journal.
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
from typing import Any, Mapping, Protocol

from tools.continual_promotion import (
    ContinualEvidence,
    ContinualPromotionDecision,
    ContinualPromotionPolicy,
    evaluate_continual_promotion,
)
from tools.feedback_promotion import (
    CandidateMetrics,
    FeedbackBatchManifest,
    PromotionDecision,
)
from tools.index_drift import IndexAdaptationDecision
from tools.security import normalize_owner_id
from tools.training_lineage import TrainingLineage, TrainingOutcome, TrainingRequest

_HEX = frozenset("0123456789abcdef")
_STATES = frozenset(
    {
        "detected",
        "build_requested",
        "build_ready",
        "benchmark_requested",
        "benchmark_ready",
        "decision_ready",
        "held",
        "stable_held",
        "promoted",
        "rolled_back",
        "failed",
    }
)
_TERMINAL = frozenset({"held", "stable_held", "promoted", "rolled_back", "failed"})
_MAX_LEASE_SECONDS = 86_400.0
_MAX_JSON_BYTES = 2_000_000


class ContinualValidationError(RuntimeError):
    """Permanent evidence/identity mismatch that must fail the workflow closed."""


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
        raise ValueError(f"{label} must be a finite non-negative timestamp")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite non-negative timestamp") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be a finite non-negative timestamp")
    return selected


def _lease(value: Any) -> float:
    selected = _timestamp(value, "lease_seconds")
    if not 0.0 < selected <= _MAX_LEASE_SECONDS:
        raise ValueError("lease_seconds must be between zero and 86400")
    return selected


def _canonical(value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("continual workflow payload exceeds storage bound")
    return encoded


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _policy_digest(policy: ContinualPromotionPolicy) -> str:
    return _digest({"contract": "rigorousrag-continual-promotion-policy-v1", **asdict(policy)})


@dataclass(frozen=True)
class ContinualWorkflowSpec:
    owner_id: str
    baseline_version: str
    candidate_version: str
    drift_evidence_sha256: str
    adaptation_policy_sha256: str
    adaptation_decision: IndexAdaptationDecision
    training_request: TrainingRequest
    feedback_batch: FeedbackBatchManifest
    benchmark_contract_sha256: str
    continual_policy: ContinualPromotionPolicy = ContinualPromotionPolicy()

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "baseline_version", _text(self.baseline_version, "baseline_version", 500))
        object.__setattr__(self, "candidate_version", _text(self.candidate_version, "candidate_version", 500))
        if self.baseline_version == self.candidate_version:
            raise ValueError("candidate_version must differ from baseline_version")
        for name in ("drift_evidence_sha256", "adaptation_policy_sha256", "benchmark_contract_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.adaptation_decision, IndexAdaptationDecision):
            raise ValueError("adaptation_decision must be IndexAdaptationDecision")
        if not isinstance(self.training_request, TrainingRequest):
            raise ValueError("training_request must be TrainingRequest")
        if not isinstance(self.feedback_batch, FeedbackBatchManifest):
            raise ValueError("feedback_batch must be FeedbackBatchManifest")
        if self.feedback_batch.owner_id != self.owner_id:
            raise ValueError("feedback batch crosses owner boundary")
        if not isinstance(self.continual_policy, ContinualPromotionPolicy):
            raise ValueError("continual_policy must be ContinualPromotionPolicy")

    @property
    def continual_policy_sha256(self) -> str:
        return _policy_digest(self.continual_policy)

    @property
    def spec_sha256(self) -> str:
        payload = {
            "contract": "rigorousrag-continual-workflow-spec-v1",
            "owner_id": self.owner_id,
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "drift_evidence_sha256": self.drift_evidence_sha256,
            "adaptation_policy_sha256": self.adaptation_policy_sha256,
            "adaptation_decision": asdict(self.adaptation_decision),
            "training_request_sha256": self.training_request.request_sha256,
            "feedback_batch_id": self.feedback_batch.batch_id,
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "continual_policy_sha256": self.continual_policy_sha256,
        }
        return _digest(payload)

    @property
    def workflow_id(self) -> str:
        return _digest({"contract": "rigorousrag-continual-workflow-id-v1", "spec_sha256": self.spec_sha256})


@dataclass(frozen=True)
class BenchmarkEvidence:
    workflow_id: str
    benchmark_receipt_sha256: str
    verified_dataset_manifest_sha256: str
    experiment_sha256: str
    candidate_artifact_sha256: str
    base_decision: PromotionDecision
    continual_evidence: ContinualEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _sha(self.workflow_id, "workflow_id"))
        for name in (
            "benchmark_receipt_sha256",
            "verified_dataset_manifest_sha256",
            "experiment_sha256",
            "candidate_artifact_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.base_decision, PromotionDecision):
            raise ValueError("base_decision must be PromotionDecision")
        if not isinstance(self.continual_evidence, ContinualEvidence):
            raise ValueError("continual_evidence must be ContinualEvidence")

    @property
    def evidence_sha256(self) -> str:
        return _digest(
            {
                "contract": "rigorousrag-continual-benchmark-evidence-v1",
                "workflow_id": self.workflow_id,
                "benchmark_receipt_sha256": self.benchmark_receipt_sha256,
                "verified_dataset_manifest_sha256": self.verified_dataset_manifest_sha256,
                "experiment_sha256": self.experiment_sha256,
                "candidate_artifact_sha256": self.candidate_artifact_sha256,
                "base_decision_id": self.base_decision.decision_id,
                "continual_evidence": asdict(self.continual_evidence),
            }
        )


@dataclass(frozen=True)
class PromotionReceipt:
    workflow_id: str
    previous_version: str
    promoted_version: str
    candidate_artifact_sha256: str
    decision_id: str
    publication_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _sha(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "previous_version", _text(self.previous_version, "previous_version", 500))
        object.__setattr__(self, "promoted_version", _text(self.promoted_version, "promoted_version", 500))
        for name in ("candidate_artifact_sha256", "decision_id", "publication_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))


@dataclass(frozen=True)
class RollbackReceipt:
    workflow_id: str
    rolled_back_from_version: str
    restored_version: str
    promotion_publication_sha256: str
    rollback_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _sha(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "rolled_back_from_version", _text(self.rolled_back_from_version, "rolled_back_from_version", 500))
        object.__setattr__(self, "restored_version", _text(self.restored_version, "restored_version", 500))
        object.__setattr__(self, "promotion_publication_sha256", _sha(self.promotion_publication_sha256, "promotion_publication_sha256"))
        object.__setattr__(self, "rollback_sha256", _sha(self.rollback_sha256, "rollback_sha256"))


class CandidateBuildBackend(Protocol):
    """Build backend; repeated calls with one workflow id must be idempotent."""

    def build(self, request: TrainingRequest, *, workflow_id: str, fencing_token: int) -> TrainingOutcome: ...


class BenchmarkBackend(Protocol):
    """Benchmark backend; it must evaluate the exact candidate artifact supplied."""

    def evaluate(
        self,
        spec: ContinualWorkflowSpec,
        lineage: TrainingLineage,
        *,
        workflow_id: str,
        fencing_token: int,
    ) -> BenchmarkEvidence: ...


class PromotionBackend(Protocol):
    """Publication backend with expected-baseline CAS semantics and independent rollback."""

    def promote(
        self,
        spec: ContinualWorkflowSpec,
        lineage: TrainingLineage,
        evidence: BenchmarkEvidence,
        decision: ContinualPromotionDecision,
        *,
        workflow_id: str,
        expected_baseline_version: str,
        fencing_token: int,
    ) -> PromotionReceipt: ...

    def rollback(
        self,
        spec: ContinualWorkflowSpec,
        promotion: PromotionReceipt,
        *,
        workflow_id: str,
        expected_current_version: str,
        fencing_token: int,
    ) -> RollbackReceipt: ...


@dataclass(frozen=True)
class WorkflowClaim:
    workflow_id: str
    worker_id: str
    fencing_token: int
    lease_expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _sha(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id", 256))
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        object.__setattr__(self, "lease_expires_at", _timestamp(self.lease_expires_at, "lease_expires_at"))


@dataclass(frozen=True)
class ContinualWorkflowRecord:
    workflow_id: str
    owner_id: str
    spec_sha256: str
    state: str
    revision: int
    created_at: float
    updated_at: float
    build_payload: Mapping[str, Any] | None = None
    benchmark_payload: Mapping[str, Any] | None = None
    decision_payload: Mapping[str, Any] | None = None
    promotion_payload: Mapping[str, Any] | None = None
    rollback_payload: Mapping[str, Any] | None = None
    failure_type: str | None = None
    terminal_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _sha(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "spec_sha256", _sha(self.spec_sha256, "spec_sha256"))
        if self.state not in _STATES:
            raise ValueError("invalid continual workflow state")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be non-negative")
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.failure_type is not None:
            object.__setattr__(self, "failure_type", _text(self.failure_type, "failure_type", 300))
        if self.terminal_receipt_sha256 is not None:
            object.__setattr__(self, "terminal_receipt_sha256", _sha(self.terminal_receipt_sha256, "terminal_receipt_sha256"))

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL


class SQLiteContinualWorkflowStore:
    """Durable workflow state with revision CAS and monotonic-fenced executor claims."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        selected = Path(os.fspath(path))
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        selected.parent.mkdir(parents=True, exist_ok=True)
        self.path = selected.absolute()
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS continual_workflow (
                    workflow_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    build_json TEXT,
                    benchmark_json TEXT,
                    decision_json TEXT,
                    promotion_json TEXT,
                    rollback_json TEXT,
                    failure_type TEXT,
                    terminal_receipt_sha256 TEXT
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS continual_workflow_claim (
                    workflow_id TEXT PRIMARY KEY,
                    worker_id TEXT,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL
                )
                """
            )
        except sqlite3.Error as exc:
            raise RuntimeError("continual workflow store initialization failed") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _json(value: Mapping[str, Any] | None) -> str | None:
        return None if value is None else _canonical(dict(value)).decode("utf-8")

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise RuntimeError("continual workflow payload is corrupt") from exc
        if not isinstance(parsed, Mapping):
            raise RuntimeError("continual workflow payload is corrupt")
        return dict(parsed)

    @classmethod
    def _record(cls, row: tuple[Any, ...] | None) -> ContinualWorkflowRecord | None:
        if row is None:
            return None
        return ContinualWorkflowRecord(
            workflow_id=row[0], owner_id=row[1], spec_sha256=row[2], state=row[3], revision=int(row[4]),
            created_at=float(row[5]), updated_at=float(row[6]), build_payload=cls._mapping(row[7]),
            benchmark_payload=cls._mapping(row[8]), decision_payload=cls._mapping(row[9]),
            promotion_payload=cls._mapping(row[10]), rollback_payload=cls._mapping(row[11]),
            failure_type=row[12], terminal_receipt_sha256=row[13],
        )

    def _get(self, workflow_id: str) -> ContinualWorkflowRecord | None:
        row = self._connection.execute(
            "SELECT workflow_id,owner_id,spec_sha256,state,revision,created_at,updated_at,build_json,benchmark_json,decision_json,promotion_json,rollback_json,failure_type,terminal_receipt_sha256 FROM continual_workflow WHERE workflow_id=?",
            (workflow_id,),
        ).fetchone()
        return self._record(row)

    def get(self, workflow_id: str) -> ContinualWorkflowRecord | None:
        selected = _sha(workflow_id, "workflow_id")
        with self._lock:
            try:
                return self._get(selected)
            except sqlite3.Error as exc:
                raise RuntimeError("continual workflow read failed") from exc

    def ensure(self, spec: ContinualWorkflowSpec, *, now: float) -> ContinualWorkflowRecord:
        if not isinstance(spec, ContinualWorkflowSpec):
            raise ValueError("spec must be ContinualWorkflowSpec")
        instant = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                current = self._get(spec.workflow_id)
                if current is not None:
                    self._connection.execute("COMMIT")
                    if current.owner_id != spec.owner_id or current.spec_sha256 != spec.spec_sha256:
                        raise RuntimeError("continual workflow identity collision")
                    return current
                self._connection.execute(
                    "INSERT INTO continual_workflow(workflow_id,owner_id,spec_sha256,state,revision,created_at,updated_at) VALUES(?,?,?,'detected',0,?,?)",
                    (spec.workflow_id, spec.owner_id, spec.spec_sha256, instant, instant),
                )
                self._connection.execute(
                    "INSERT INTO continual_workflow_claim(workflow_id,worker_id,fencing_token,lease_expires_at) VALUES(?,NULL,0,NULL)",
                    (spec.workflow_id,),
                )
                self._connection.execute("COMMIT")
                created = self._get(spec.workflow_id)
                if created is None:
                    raise RuntimeError("continual workflow disappeared after creation")
                return created
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("continual workflow creation failed") from exc

    def claim(self, workflow_id: str, *, worker_id: str, now: float, lease_seconds: float) -> WorkflowClaim:
        selected = _sha(workflow_id, "workflow_id")
        worker = _text(worker_id, "worker_id", 256)
        instant = _timestamp(now, "now")
        expiry = instant + _lease(lease_seconds)
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if self._get(selected) is None:
                    raise KeyError(selected)
                row = self._connection.execute(
                    "SELECT worker_id,fencing_token,lease_expires_at FROM continual_workflow_claim WHERE workflow_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("continual workflow claim row missing")
                current_worker, token, current_expiry = row
                if current_worker is not None and current_expiry is not None and float(current_expiry) > instant:
                    raise RuntimeError("continual workflow already has a live executor")
                next_token = int(token) + 1
                if next_token > 2**63 - 1:
                    raise RuntimeError("continual workflow fencing token exhausted")
                self._connection.execute(
                    "UPDATE continual_workflow_claim SET worker_id=?,fencing_token=?,lease_expires_at=? WHERE workflow_id=?",
                    (worker, next_token, expiry, selected),
                )
                self._connection.execute("COMMIT")
                return WorkflowClaim(selected, worker, next_token, expiry)
            except (KeyError, RuntimeError):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("continual workflow claim failed") from exc

    def assert_claim(self, claim: WorkflowClaim, *, now: float) -> None:
        if not isinstance(claim, WorkflowClaim):
            raise ValueError("claim must be WorkflowClaim")
        instant = _timestamp(now, "now")
        with self._lock:
            row = self._connection.execute(
                "SELECT worker_id,fencing_token,lease_expires_at FROM continual_workflow_claim WHERE workflow_id=?",
                (claim.workflow_id,),
            ).fetchone()
        if row is None or row[0] != claim.worker_id or int(row[1]) != claim.fencing_token or row[2] is None or float(row[2]) <= instant:
            raise RuntimeError("continual workflow executor lease is stale or fenced")

    def renew(self, claim: WorkflowClaim, *, now: float, lease_seconds: float) -> WorkflowClaim:
        instant = _timestamp(now, "now")
        self.assert_claim(claim, now=instant)
        expiry = instant + _lease(lease_seconds)
        with self._lock:
            updated = self._connection.execute(
                "UPDATE continual_workflow_claim SET lease_expires_at=? WHERE workflow_id=? AND worker_id=? AND fencing_token=? AND lease_expires_at>?",
                (expiry, claim.workflow_id, claim.worker_id, claim.fencing_token, instant),
            )
            if updated.rowcount != 1:
                raise RuntimeError("continual workflow lease renewal was fenced")
        return WorkflowClaim(claim.workflow_id, claim.worker_id, claim.fencing_token, expiry)

    def release(self, claim: WorkflowClaim, *, now: float) -> None:
        instant = _timestamp(now, "now")
        self.assert_claim(claim, now=instant)
        with self._lock:
            updated = self._connection.execute(
                "UPDATE continual_workflow_claim SET worker_id=NULL,lease_expires_at=NULL WHERE workflow_id=? AND worker_id=? AND fencing_token=?",
                (claim.workflow_id, claim.worker_id, claim.fencing_token),
            )
            if updated.rowcount != 1:
                raise RuntimeError("continual workflow lease release was fenced")

    def transition(
        self,
        claim: WorkflowClaim,
        *,
        expected_state: str,
        expected_revision: int,
        new_state: str,
        now: float,
        build_payload: Mapping[str, Any] | None = None,
        benchmark_payload: Mapping[str, Any] | None = None,
        decision_payload: Mapping[str, Any] | None = None,
        promotion_payload: Mapping[str, Any] | None = None,
        rollback_payload: Mapping[str, Any] | None = None,
        failure_type: str | None = None,
        terminal_receipt_sha256: str | None = None,
    ) -> ContinualWorkflowRecord:
        if expected_state not in _STATES or new_state not in _STATES:
            raise ValueError("invalid continual workflow transition state")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision is invalid")
        instant = _timestamp(now, "now")
        self.assert_claim(claim, now=instant)
        failure = None if failure_type is None else _text(failure_type, "failure_type", 300)
        terminal = None if terminal_receipt_sha256 is None else _sha(terminal_receipt_sha256, "terminal_receipt_sha256")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT state,revision FROM continual_workflow WHERE workflow_id=?",
                    (claim.workflow_id,),
                ).fetchone()
                if row is None or row[0] != expected_state or int(row[1]) != expected_revision:
                    raise RuntimeError("continual workflow state changed since observation")
                updated = self._connection.execute(
                    """
                    UPDATE continual_workflow SET state=?,revision=revision+1,updated_at=?,
                        build_json=COALESCE(?,build_json), benchmark_json=COALESCE(?,benchmark_json),
                        decision_json=COALESCE(?,decision_json), promotion_json=COALESCE(?,promotion_json),
                        rollback_json=COALESCE(?,rollback_json), failure_type=COALESCE(?,failure_type),
                        terminal_receipt_sha256=COALESCE(?,terminal_receipt_sha256)
                    WHERE workflow_id=? AND state=? AND revision=?
                    """,
                    (
                        new_state, instant, self._json(build_payload), self._json(benchmark_payload),
                        self._json(decision_payload), self._json(promotion_payload), self._json(rollback_payload),
                        failure, terminal, claim.workflow_id, expected_state, expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("continual workflow transition lost compare-and-swap")
                self._connection.execute("COMMIT")
                current = self._get(claim.workflow_id)
                if current is None:
                    raise RuntimeError("continual workflow disappeared after transition")
                return current
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("continual workflow transition failed") from exc


def _training_payload(lineage: TrainingLineage) -> Mapping[str, Any]:
    return {
        "output_artifact_sha256": lineage.outcome.output_artifact_sha256,
        "evaluation_sha256": list(lineage.outcome.evaluation_sha256),
        "provider_run_ref": lineage.outcome.provider_run_ref,
        "lineage_sha256": lineage.lineage_sha256,
    }


def _lineage_from_payload(spec: ContinualWorkflowSpec, payload: Mapping[str, Any] | None) -> TrainingLineage:
    if payload is None:
        raise ContinualValidationError("build payload is missing")
    try:
        outcome = TrainingOutcome(
            output_artifact_sha256=payload["output_artifact_sha256"],
            evaluation_sha256=tuple(payload.get("evaluation_sha256", ())),
            provider_run_ref=payload.get("provider_run_ref", "local"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("build payload is corrupt") from exc
    lineage = TrainingLineage.bind(spec.training_request, outcome)
    if lineage.lineage_sha256 != payload.get("lineage_sha256"):
        raise ContinualValidationError("training lineage digest changed after persistence")
    return lineage


def _promotion_decision_payload(value: PromotionDecision) -> Mapping[str, Any]:
    return asdict(value)


def _promotion_decision_from_payload(payload: Mapping[str, Any]) -> PromotionDecision:
    try:
        return PromotionDecision(
            decision_id=payload["decision_id"], eligible=bool(payload["eligible"]), reason_codes=tuple(payload["reason_codes"]),
            owner_id=payload["owner_id"], batch_id=payload["batch_id"], baseline_version=payload["baseline_version"],
            candidate_version=payload["candidate_version"], baseline=CandidateMetrics(**payload["baseline"]),
            candidate=CandidateMetrics(**payload["candidate"]), quality_delta=float(payload["quality_delta"]),
            latency_ratio=float(payload["latency_ratio"]), cost_ratio=float(payload["cost_ratio"]),
            policy_fingerprint=payload["policy_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted base promotion decision is corrupt") from exc


def _benchmark_payload(value: BenchmarkEvidence) -> Mapping[str, Any]:
    return {
        "workflow_id": value.workflow_id,
        "benchmark_receipt_sha256": value.benchmark_receipt_sha256,
        "verified_dataset_manifest_sha256": value.verified_dataset_manifest_sha256,
        "experiment_sha256": value.experiment_sha256,
        "candidate_artifact_sha256": value.candidate_artifact_sha256,
        "base_decision": _promotion_decision_payload(value.base_decision),
        "continual_evidence": asdict(value.continual_evidence),
        "evidence_sha256": value.evidence_sha256,
    }


def _benchmark_from_payload(payload: Mapping[str, Any] | None) -> BenchmarkEvidence:
    if payload is None:
        raise ContinualValidationError("benchmark payload is missing")
    try:
        value = BenchmarkEvidence(
            workflow_id=payload["workflow_id"], benchmark_receipt_sha256=payload["benchmark_receipt_sha256"],
            verified_dataset_manifest_sha256=payload["verified_dataset_manifest_sha256"], experiment_sha256=payload["experiment_sha256"],
            candidate_artifact_sha256=payload["candidate_artifact_sha256"], base_decision=_promotion_decision_from_payload(payload["base_decision"]),
            continual_evidence=ContinualEvidence(**payload["continual_evidence"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted benchmark evidence is corrupt") from exc
    if value.evidence_sha256 != payload.get("evidence_sha256"):
        raise ContinualValidationError("benchmark evidence digest changed after persistence")
    return value


def _continual_decision_payload(value: ContinualPromotionDecision) -> Mapping[str, Any]:
    return {
        "decision_id": value.decision_id,
        "eligible": value.eligible,
        "reason_codes": list(value.reason_codes),
        "base_promotion_decision_id": value.base_promotion_decision_id,
        "adapter_version": value.adapter_version,
        "evidence": asdict(value.evidence),
        "policy_fingerprint": value.policy_fingerprint,
    }


def _continual_decision_from_payload(payload: Mapping[str, Any] | None) -> ContinualPromotionDecision:
    if payload is None:
        raise ContinualValidationError("continual decision payload is missing")
    try:
        return ContinualPromotionDecision(
            decision_id=payload["decision_id"], eligible=bool(payload["eligible"]), reason_codes=tuple(payload["reason_codes"]),
            base_promotion_decision_id=payload["base_promotion_decision_id"], adapter_version=payload["adapter_version"],
            evidence=ContinualEvidence(**payload["evidence"]), policy_fingerprint=payload["policy_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted continual decision is corrupt") from exc


def _promotion_payload(value: PromotionReceipt) -> Mapping[str, Any]:
    return asdict(value)


def _promotion_from_payload(payload: Mapping[str, Any] | None) -> PromotionReceipt:
    if payload is None:
        raise ContinualValidationError("promotion receipt is missing")
    try:
        return PromotionReceipt(**payload)
    except (TypeError, ValueError) as exc:
        raise ContinualValidationError("promotion receipt is corrupt") from exc


def _terminal_digest(spec: ContinualWorkflowSpec, state: str, evidence: Mapping[str, Any]) -> str:
    return _digest({"contract": "rigorousrag-continual-terminal-v1", "workflow_id": spec.workflow_id, "state": state, "evidence": dict(evidence)})


def _validate_benchmark(spec: ContinualWorkflowSpec, lineage: TrainingLineage, evidence: BenchmarkEvidence) -> None:
    if evidence.workflow_id != spec.workflow_id:
        raise ContinualValidationError("benchmark evidence belongs to another workflow")
    if evidence.candidate_artifact_sha256 != lineage.outcome.output_artifact_sha256:
        raise ContinualValidationError("benchmark evaluated a different candidate artifact")
    decision = evidence.base_decision
    if decision.owner_id != spec.owner_id or decision.batch_id != spec.feedback_batch.batch_id:
        raise ContinualValidationError("benchmark promotion decision crosses owner/feedback identity")
    if decision.baseline_version != spec.baseline_version or decision.candidate_version != spec.candidate_version:
        raise ContinualValidationError("benchmark promotion decision version identity differs from workflow")


def _validate_promotion(spec: ContinualWorkflowSpec, lineage: TrainingLineage, decision: ContinualPromotionDecision, receipt: PromotionReceipt) -> None:
    if receipt.workflow_id != spec.workflow_id:
        raise ContinualValidationError("promotion receipt belongs to another workflow")
    if receipt.previous_version != spec.baseline_version or receipt.promoted_version != spec.candidate_version:
        raise ContinualValidationError("promotion receipt version identity differs from workflow")
    if receipt.candidate_artifact_sha256 != lineage.outcome.output_artifact_sha256:
        raise ContinualValidationError("promotion published a different candidate artifact")
    if receipt.decision_id != decision.decision_id:
        raise ContinualValidationError("promotion receipt decision differs from governed decision")


def _fail_closed(
    store: SQLiteContinualWorkflowStore,
    claim: WorkflowClaim,
    spec: ContinualWorkflowSpec,
    record: ContinualWorkflowRecord,
    error: ContinualValidationError,
    *,
    now: float,
) -> ContinualWorkflowRecord:
    failure_type = error.__class__.__name__
    receipt = _terminal_digest(spec, "failed", {"failure_type": failure_type, "state": record.state, "revision": record.revision})
    return store.transition(
        claim, expected_state=record.state, expected_revision=record.revision, new_state="failed", now=now,
        failure_type=failure_type, terminal_receipt_sha256=receipt,
    )


def advance_continual_workflow(
    spec: ContinualWorkflowSpec,
    *,
    store: SQLiteContinualWorkflowStore,
    build_backend: CandidateBuildBackend,
    benchmark_backend: BenchmarkBackend,
    promotion_backend: PromotionBackend,
    worker_id: str,
    now: float,
    lease_seconds: float = 3_600.0,
    max_steps: int = 16,
) -> ContinualWorkflowRecord:
    """Advance one workflow until terminal or ``max_steps`` is reached.

    Backend exceptions are intentionally *not* converted into terminal failure: the row
    remains in its requested state so the stable idempotency key can be retried.  Only
    authority/evidence mismatches represented by :class:`ContinualValidationError` are
    sealed as fail-closed terminal records.
    """

    if not isinstance(spec, ContinualWorkflowSpec):
        raise ValueError("spec must be ContinualWorkflowSpec")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 1_000:
        raise ValueError("max_steps is invalid")
    instant = _timestamp(now, "now")
    lease = _lease(lease_seconds)
    store.ensure(spec, now=instant)
    claim = store.claim(spec.workflow_id, worker_id=worker_id, now=instant, lease_seconds=lease)
    try:
        for _ in range(max_steps):
            claim = store.renew(claim, now=instant, lease_seconds=lease)
            record = store.get(spec.workflow_id)
            if record is None:
                raise RuntimeError("continual workflow disappeared")
            if record.owner_id != spec.owner_id or record.spec_sha256 != spec.spec_sha256:
                raise RuntimeError("continual workflow persisted identity differs from supplied spec")
            if record.terminal:
                return record
            try:
                if record.state == "detected":
                    if spec.adaptation_decision.action == "stable":
                        terminal = _terminal_digest(spec, "stable_held", {"drift_evidence_sha256": spec.drift_evidence_sha256})
                        return store.transition(claim, expected_state="detected", expected_revision=record.revision, new_state="stable_held", now=instant, terminal_receipt_sha256=terminal)
                    record = store.transition(claim, expected_state="detected", expected_revision=record.revision, new_state="build_requested", now=instant)
                    continue

                if record.state == "build_requested":
                    outcome = build_backend.build(spec.training_request, workflow_id=spec.workflow_id, fencing_token=claim.fencing_token)
                    if not isinstance(outcome, TrainingOutcome):
                        raise ContinualValidationError("build backend returned an invalid training outcome")
                    lineage = TrainingLineage.bind(spec.training_request, outcome)
                    record = store.transition(claim, expected_state="build_requested", expected_revision=record.revision, new_state="build_ready", now=instant, build_payload=_training_payload(lineage))
                    continue

                if record.state == "build_ready":
                    _lineage_from_payload(spec, record.build_payload)
                    record = store.transition(claim, expected_state="build_ready", expected_revision=record.revision, new_state="benchmark_requested", now=instant)
                    continue

                if record.state == "benchmark_requested":
                    lineage = _lineage_from_payload(spec, record.build_payload)
                    evidence = benchmark_backend.evaluate(spec, lineage, workflow_id=spec.workflow_id, fencing_token=claim.fencing_token)
                    if not isinstance(evidence, BenchmarkEvidence):
                        raise ContinualValidationError("benchmark backend returned invalid evidence")
                    _validate_benchmark(spec, lineage, evidence)
                    record = store.transition(claim, expected_state="benchmark_requested", expected_revision=record.revision, new_state="benchmark_ready", now=instant, benchmark_payload=_benchmark_payload(evidence))
                    continue

                if record.state == "benchmark_ready":
                    evidence = _benchmark_from_payload(record.benchmark_payload)
                    decision = evaluate_continual_promotion(base=evidence.base_decision, evidence=evidence.continual_evidence, policy=spec.continual_policy)
                    if decision.policy_fingerprint != spec.continual_policy_sha256:
                        # The older continual gate hashes only ``asdict(policy)``. Preserve
                        # its native fingerprint as authority; the spec digest separately
                        # binds our namespaced policy digest.
                        native = _digest(asdict(spec.continual_policy))
                        if decision.policy_fingerprint != native:
                            raise ContinualValidationError("continual promotion policy fingerprint mismatch")
                    record = store.transition(claim, expected_state="benchmark_ready", expected_revision=record.revision, new_state="decision_ready", now=instant, decision_payload=_continual_decision_payload(decision))
                    continue

                if record.state == "decision_ready":
                    lineage = _lineage_from_payload(spec, record.build_payload)
                    evidence = _benchmark_from_payload(record.benchmark_payload)
                    decision = _continual_decision_from_payload(record.decision_payload)
                    if decision.base_promotion_decision_id != evidence.base_decision.decision_id:
                        raise ContinualValidationError("continual decision no longer binds benchmark base decision")
                    if not decision.eligible:
                        terminal = _terminal_digest(spec, "held", {"decision_id": decision.decision_id, "reason_codes": list(decision.reason_codes), "benchmark_evidence_sha256": evidence.evidence_sha256})
                        return store.transition(claim, expected_state="decision_ready", expected_revision=record.revision, new_state="held", now=instant, terminal_receipt_sha256=terminal)
                    receipt = promotion_backend.promote(
                        spec, lineage, evidence, decision, workflow_id=spec.workflow_id,
                        expected_baseline_version=spec.baseline_version, fencing_token=claim.fencing_token,
                    )
                    if not isinstance(receipt, PromotionReceipt):
                        raise ContinualValidationError("promotion backend returned an invalid receipt")
                    _validate_promotion(spec, lineage, decision, receipt)
                    terminal = _terminal_digest(spec, "promoted", {"decision_id": decision.decision_id, "publication_sha256": receipt.publication_sha256, "candidate_artifact_sha256": receipt.candidate_artifact_sha256})
                    return store.transition(claim, expected_state="decision_ready", expected_revision=record.revision, new_state="promoted", now=instant, promotion_payload=_promotion_payload(receipt), terminal_receipt_sha256=terminal)

                raise ContinualValidationError(f"unsupported nonterminal workflow state: {record.state}")
            except ContinualValidationError as exc:
                return _fail_closed(store, claim, spec, record, exc, now=instant)
        final = store.get(spec.workflow_id)
        if final is None:
            raise RuntimeError("continual workflow disappeared")
        return final
    finally:
        try:
            store.release(claim, now=instant)
        except RuntimeError:
            # A lease may have expired or been fenced while an injected backend ran.
            # Never mask the primary workflow result/error with release cleanup.
            pass


def rollback_promoted_workflow(
    spec: ContinualWorkflowSpec,
    *,
    store: SQLiteContinualWorkflowStore,
    promotion_backend: PromotionBackend,
    worker_id: str,
    now: float,
    lease_seconds: float = 3_600.0,
) -> ContinualWorkflowRecord:
    """Append an independent rollback after a previously promoted workflow."""

    instant = _timestamp(now, "now")
    lease = _lease(lease_seconds)
    record = store.get(spec.workflow_id)
    if record is None or record.state != "promoted":
        raise ValueError("only a promoted continual workflow can be rolled back")
    promotion = _promotion_from_payload(record.promotion_payload)
    claim = store.claim(spec.workflow_id, worker_id=worker_id, now=instant, lease_seconds=lease)
    try:
        claim = store.renew(claim, now=instant, lease_seconds=lease)
        current = store.get(spec.workflow_id)
        if current is None or current.state != "promoted" or current.revision != record.revision:
            raise RuntimeError("promotion state changed before rollback")
        receipt = promotion_backend.rollback(
            spec, promotion, workflow_id=spec.workflow_id,
            expected_current_version=spec.candidate_version, fencing_token=claim.fencing_token,
        )
        if not isinstance(receipt, RollbackReceipt):
            raise ContinualValidationError("rollback backend returned invalid receipt")
        if (
            receipt.workflow_id != spec.workflow_id
            or receipt.rolled_back_from_version != spec.candidate_version
            or receipt.restored_version != spec.baseline_version
            or receipt.promotion_publication_sha256 != promotion.publication_sha256
        ):
            raise ContinualValidationError("rollback receipt does not restore the governed baseline")
        terminal = _terminal_digest(spec, "rolled_back", {"promotion_publication_sha256": promotion.publication_sha256, "rollback_sha256": receipt.rollback_sha256})
        return store.transition(
            claim, expected_state="promoted", expected_revision=current.revision, new_state="rolled_back", now=instant,
            rollback_payload=asdict(receipt), terminal_receipt_sha256=terminal,
        )
    finally:
        try:
            store.release(claim, now=instant)
        except RuntimeError:
            pass


__all__ = [
    "BenchmarkBackend",
    "BenchmarkEvidence",
    "CandidateBuildBackend",
    "ContinualValidationError",
    "ContinualWorkflowRecord",
    "ContinualWorkflowSpec",
    "PromotionBackend",
    "PromotionReceipt",
    "RollbackReceipt",
    "SQLiteContinualWorkflowStore",
    "WorkflowClaim",
    "advance_continual_workflow",
    "rollback_promoted_workflow",
]
