"""Durable drift -> build -> benchmark -> promotion orchestration.

Existing RigorousRAG modules provide drift decisions, reproducible training lineage,
benchmark evidence, feedback promotion gates, continual-learning safeguards and a
promotion journal.  This module binds those pieces into one crash-resumable workflow.
It starts no trainer, downloader, model process, benchmark process or background loop.

Injected side-effect backends receive a stable workflow id plus a monotonic fencing
token and must be idempotent.  SQLite stores only identifiers, digests, bounded metrics,
decisions, receipts and stage state--never raw examples, queries, documents or tensors.
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

from tools.continual_promotion import ContinualEvidence, ContinualPromotionDecision, ContinualPromotionPolicy, evaluate_continual_promotion
from tools.feedback_promotion import CandidateMetrics, FeedbackBatchManifest, PromotionDecision
from tools.index_drift import IndexAdaptationDecision
from tools.security import normalize_owner_id
from tools.training_lineage import TrainingLineage, TrainingOutcome, TrainingRequest

_HEX = frozenset("0123456789abcdef")
_STATES = frozenset({"detected", "build_requested", "build_ready", "benchmark_requested", "benchmark_ready", "decision_ready", "held", "stable_held", "promoted", "rolled_back", "failed"})
_TERMINAL = frozenset({"held", "stable_held", "promoted", "rolled_back", "failed"})
_MAX_LEASE_SECONDS = 86_400.0
_MAX_JSON_BYTES = 2_000_000


class ContinualValidationError(RuntimeError):
    """Permanent evidence/authority mismatch that must terminate fail-closed."""


def _text(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label, 64).lower()
    if len(result) != 64 or any(ch not in _HEX for ch in result):
        raise ValueError(f"{label} must be SHA-256")
    return result


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _lease(value: Any) -> float:
    result = _time(value, "lease_seconds")
    if not 0 < result <= _MAX_LEASE_SECONDS:
        raise ValueError("lease_seconds must be between zero and 86400")
    return result


def _canonical(value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("continual workflow payload exceeds storage bound")
    return encoded


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
        return _digest({"contract": "rigorousrag-continual-promotion-policy-v1", **asdict(self.continual_policy)})

    @property
    def spec_sha256(self) -> str:
        return _digest({
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
        })

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
        for name in ("benchmark_receipt_sha256", "verified_dataset_manifest_sha256", "experiment_sha256", "candidate_artifact_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.base_decision, PromotionDecision):
            raise ValueError("base_decision must be PromotionDecision")
        if not isinstance(self.continual_evidence, ContinualEvidence):
            raise ValueError("continual_evidence must be ContinualEvidence")

    @property
    def evidence_sha256(self) -> str:
        return _digest({
            "contract": "rigorousrag-continual-benchmark-evidence-v1",
            "workflow_id": self.workflow_id,
            "benchmark_receipt_sha256": self.benchmark_receipt_sha256,
            "verified_dataset_manifest_sha256": self.verified_dataset_manifest_sha256,
            "experiment_sha256": self.experiment_sha256,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "base_decision_id": self.base_decision.decision_id,
            "continual_evidence": asdict(self.continual_evidence),
        })


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
    def build(self, request: TrainingRequest, *, workflow_id: str, fencing_token: int) -> TrainingOutcome: ...


class BenchmarkBackend(Protocol):
    def evaluate(self, spec: ContinualWorkflowSpec, lineage: TrainingLineage, *, workflow_id: str, fencing_token: int) -> BenchmarkEvidence: ...


class PromotionBackend(Protocol):
    def promote(self, spec: ContinualWorkflowSpec, lineage: TrainingLineage, evidence: BenchmarkEvidence, decision: ContinualPromotionDecision, *, workflow_id: str, expected_baseline_version: str, fencing_token: int) -> PromotionReceipt: ...
    def rollback(self, spec: ContinualWorkflowSpec, promotion: PromotionReceipt, *, workflow_id: str, expected_current_version: str, fencing_token: int) -> RollbackReceipt: ...


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
        object.__setattr__(self, "lease_expires_at", _time(self.lease_expires_at, "lease_expires_at"))


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
        created, updated = _time(self.created_at, "created_at"), _time(self.updated_at, "updated_at")
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
    """Durable state with revision CAS and claims revalidated in the mutation transaction."""

    _COLUMNS = "workflow_id,owner_id,spec_sha256,state,revision,created_at,updated_at,build_json,benchmark_json,decision_json,promotion_json,rollback_json,failure_type,terminal_receipt_sha256"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        selected = Path(os.fspath(path))
        self.path = (selected if selected.is_absolute() else Path.cwd() / selected).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS continual_workflow (workflow_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,spec_sha256 TEXT NOT NULL,state TEXT NOT NULL,revision INTEGER NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL,build_json TEXT,benchmark_json TEXT,decision_json TEXT,promotion_json TEXT,rollback_json TEXT,failure_type TEXT,terminal_receipt_sha256 TEXT)")
        self._connection.execute("CREATE TABLE IF NOT EXISTS continual_workflow_claim (workflow_id TEXT PRIMARY KEY,worker_id TEXT,fencing_token INTEGER NOT NULL DEFAULT 0,lease_expires_at REAL)")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _dump(value: Mapping[str, Any] | None) -> str | None:
        return None if value is None else _canonical(dict(value)).decode("utf-8")

    @staticmethod
    def _load(value: Any) -> Mapping[str, Any] | None:
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
        return ContinualWorkflowRecord(row[0], row[1], row[2], row[3], int(row[4]), float(row[5]), float(row[6]), cls._load(row[7]), cls._load(row[8]), cls._load(row[9]), cls._load(row[10]), cls._load(row[11]), row[12], row[13])

    def _get_locked(self, workflow_id: str) -> ContinualWorkflowRecord | None:
        return self._record(self._connection.execute(f"SELECT {self._COLUMNS} FROM continual_workflow WHERE workflow_id=?", (workflow_id,)).fetchone())

    def get(self, workflow_id: str) -> ContinualWorkflowRecord | None:
        selected = _sha(workflow_id, "workflow_id")
        with self._lock:
            return self._get_locked(selected)

    def ensure(self, spec: ContinualWorkflowSpec, *, now: float) -> ContinualWorkflowRecord:
        if not isinstance(spec, ContinualWorkflowSpec):
            raise ValueError("spec must be ContinualWorkflowSpec")
        instant = _time(now, "now")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._get_locked(spec.workflow_id)
                if current is None:
                    self._connection.execute("INSERT INTO continual_workflow(workflow_id,owner_id,spec_sha256,state,revision,created_at,updated_at) VALUES(?,?,?,'detected',0,?,?)", (spec.workflow_id, spec.owner_id, spec.spec_sha256, instant, instant))
                    self._connection.execute("INSERT INTO continual_workflow_claim(workflow_id,worker_id,fencing_token,lease_expires_at) VALUES(?,NULL,0,NULL)", (spec.workflow_id,))
                    current = self._get_locked(spec.workflow_id)
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        if current is None:
            raise RuntimeError("continual workflow disappeared after ensure")
        if current.owner_id != spec.owner_id or current.spec_sha256 != spec.spec_sha256:
            raise RuntimeError("continual workflow identity collision")
        return current

    def _assert_claim_locked(self, claim: WorkflowClaim, instant: float) -> None:
        row = self._connection.execute("SELECT worker_id,fencing_token,lease_expires_at FROM continual_workflow_claim WHERE workflow_id=?", (claim.workflow_id,)).fetchone()
        if row is None or row[0] != claim.worker_id or int(row[1]) != claim.fencing_token or row[2] is None or float(row[2]) <= instant:
            raise RuntimeError("continual workflow executor lease is stale or fenced")

    def assert_claim(self, claim: WorkflowClaim, *, now: float) -> None:
        instant = _time(now, "now")
        with self._lock:
            self._assert_claim_locked(claim, instant)

    def claim(self, workflow_id: str, *, worker_id: str, now: float, lease_seconds: float) -> WorkflowClaim:
        selected, worker, instant, duration = _sha(workflow_id, "workflow_id"), _text(worker_id, "worker_id", 256), _time(now, "now"), _lease(lease_seconds)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self._get_locked(selected) is None:
                    raise KeyError(selected)
                row = self._connection.execute("SELECT worker_id,fencing_token,lease_expires_at FROM continual_workflow_claim WHERE workflow_id=?", (selected,)).fetchone()
                if row is None:
                    raise RuntimeError("continual workflow claim row missing")
                if row[0] is not None and row[2] is not None and float(row[2]) > instant:
                    raise RuntimeError("continual workflow already has a live executor")
                token = int(row[1]) + 1
                if token > 2**63 - 1:
                    raise RuntimeError("continual workflow fencing token exhausted")
                expiry = instant + duration
                self._connection.execute("UPDATE continual_workflow_claim SET worker_id=?,fencing_token=?,lease_expires_at=? WHERE workflow_id=?", (worker, token, expiry, selected))
                self._connection.execute("COMMIT")
                return WorkflowClaim(selected, worker, token, expiry)
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def renew(self, claim: WorkflowClaim, *, now: float, lease_seconds: float) -> WorkflowClaim:
        instant, duration = _time(now, "now"), _lease(lease_seconds)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_claim_locked(claim, instant)
                expiry = instant + duration
                self._connection.execute("UPDATE continual_workflow_claim SET lease_expires_at=? WHERE workflow_id=? AND worker_id=? AND fencing_token=?", (expiry, claim.workflow_id, claim.worker_id, claim.fencing_token))
                self._connection.execute("COMMIT")
                return WorkflowClaim(claim.workflow_id, claim.worker_id, claim.fencing_token, expiry)
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def release(self, claim: WorkflowClaim, *, now: float) -> None:
        instant = _time(now, "now")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._assert_claim_locked(claim, instant)
                self._connection.execute("UPDATE continual_workflow_claim SET worker_id=NULL,lease_expires_at=NULL WHERE workflow_id=? AND worker_id=? AND fencing_token=?", (claim.workflow_id, claim.worker_id, claim.fencing_token))
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def transition(self, claim: WorkflowClaim, *, expected_state: str, expected_revision: int, new_state: str, now: float, build_payload: Mapping[str, Any] | None = None, benchmark_payload: Mapping[str, Any] | None = None, decision_payload: Mapping[str, Any] | None = None, promotion_payload: Mapping[str, Any] | None = None, rollback_payload: Mapping[str, Any] | None = None, failure_type: str | None = None, terminal_receipt_sha256: str | None = None) -> ContinualWorkflowRecord:
        if expected_state not in _STATES or new_state not in _STATES:
            raise ValueError("invalid workflow transition state")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision is invalid")
        instant = _time(now, "now")
        failure = None if failure_type is None else _text(failure_type, "failure_type", 300)
        terminal = None if terminal_receipt_sha256 is None else _sha(terminal_receipt_sha256, "terminal_receipt_sha256")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                # Critical invariant: claim and state CAS are observed in one DB write
                # transaction, closing the check-then-takeover race.
                self._assert_claim_locked(claim, instant)
                row = self._connection.execute("SELECT state,revision FROM continual_workflow WHERE workflow_id=?", (claim.workflow_id,)).fetchone()
                if row is None or row[0] != expected_state or int(row[1]) != expected_revision:
                    raise RuntimeError("continual workflow state changed since observation")
                changed = self._connection.execute("UPDATE continual_workflow SET state=?,revision=revision+1,updated_at=?,build_json=COALESCE(?,build_json),benchmark_json=COALESCE(?,benchmark_json),decision_json=COALESCE(?,decision_json),promotion_json=COALESCE(?,promotion_json),rollback_json=COALESCE(?,rollback_json),failure_type=COALESCE(?,failure_type),terminal_receipt_sha256=COALESCE(?,terminal_receipt_sha256) WHERE workflow_id=? AND state=? AND revision=?", (new_state, instant, self._dump(build_payload), self._dump(benchmark_payload), self._dump(decision_payload), self._dump(promotion_payload), self._dump(rollback_payload), failure, terminal, claim.workflow_id, expected_state, expected_revision))
                if changed.rowcount != 1:
                    raise RuntimeError("continual workflow transition lost compare-and-swap")
                current = self._get_locked(claim.workflow_id)
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        if current is None:
            raise RuntimeError("continual workflow disappeared after transition")
        return current


def _training_payload(lineage: TrainingLineage) -> Mapping[str, Any]:
    return {"output_artifact_sha256": lineage.outcome.output_artifact_sha256, "evaluation_sha256": list(lineage.outcome.evaluation_sha256), "provider_run_ref": lineage.outcome.provider_run_ref, "lineage_sha256": lineage.lineage_sha256}


def _lineage(spec: ContinualWorkflowSpec, payload: Mapping[str, Any] | None) -> TrainingLineage:
    if payload is None:
        raise ContinualValidationError("build payload is missing")
    try:
        outcome = TrainingOutcome(payload["output_artifact_sha256"], tuple(payload.get("evaluation_sha256", ())), payload.get("provider_run_ref", "local"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("build payload is corrupt") from exc
    result = TrainingLineage.bind(spec.training_request, outcome)
    if result.lineage_sha256 != payload.get("lineage_sha256"):
        raise ContinualValidationError("training lineage digest changed after persistence")
    return result


def _base_payload(value: PromotionDecision) -> Mapping[str, Any]:
    return asdict(value)


def _base(payload: Mapping[str, Any]) -> PromotionDecision:
    try:
        return PromotionDecision(payload["decision_id"], bool(payload["eligible"]), tuple(payload["reason_codes"]), payload["owner_id"], payload["batch_id"], payload["baseline_version"], payload["candidate_version"], CandidateMetrics(**payload["baseline"]), CandidateMetrics(**payload["candidate"]), float(payload["quality_delta"]), float(payload["latency_ratio"]), float(payload["cost_ratio"]), payload["policy_fingerprint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted base promotion decision is corrupt") from exc


def _benchmark_payload(value: BenchmarkEvidence) -> Mapping[str, Any]:
    return {"workflow_id": value.workflow_id, "benchmark_receipt_sha256": value.benchmark_receipt_sha256, "verified_dataset_manifest_sha256": value.verified_dataset_manifest_sha256, "experiment_sha256": value.experiment_sha256, "candidate_artifact_sha256": value.candidate_artifact_sha256, "base_decision": _base_payload(value.base_decision), "continual_evidence": asdict(value.continual_evidence), "evidence_sha256": value.evidence_sha256}


def _benchmark(payload: Mapping[str, Any] | None) -> BenchmarkEvidence:
    if payload is None:
        raise ContinualValidationError("benchmark payload is missing")
    try:
        result = BenchmarkEvidence(payload["workflow_id"], payload["benchmark_receipt_sha256"], payload["verified_dataset_manifest_sha256"], payload["experiment_sha256"], payload["candidate_artifact_sha256"], _base(payload["base_decision"]), ContinualEvidence(**payload["continual_evidence"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted benchmark evidence is corrupt") from exc
    if result.evidence_sha256 != payload.get("evidence_sha256"):
        raise ContinualValidationError("benchmark evidence digest changed after persistence")
    return result


def _decision_payload(value: ContinualPromotionDecision) -> Mapping[str, Any]:
    return {"decision_id": value.decision_id, "eligible": value.eligible, "reason_codes": list(value.reason_codes), "base_promotion_decision_id": value.base_promotion_decision_id, "adapter_version": value.adapter_version, "evidence": asdict(value.evidence), "policy_fingerprint": value.policy_fingerprint}


def _decision(payload: Mapping[str, Any] | None) -> ContinualPromotionDecision:
    if payload is None:
        raise ContinualValidationError("continual decision payload is missing")
    try:
        return ContinualPromotionDecision(payload["decision_id"], bool(payload["eligible"]), tuple(payload["reason_codes"]), payload["base_promotion_decision_id"], payload["adapter_version"], ContinualEvidence(**payload["evidence"]), payload["policy_fingerprint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinualValidationError("persisted continual decision is corrupt") from exc


def _promotion(payload: Mapping[str, Any] | None) -> PromotionReceipt:
    if payload is None:
        raise ContinualValidationError("promotion receipt is missing")
    try:
        return PromotionReceipt(**payload)
    except (TypeError, ValueError) as exc:
        raise ContinualValidationError("promotion receipt is corrupt") from exc


def _terminal(spec: ContinualWorkflowSpec, state: str, evidence: Mapping[str, Any]) -> str:
    return _digest({"contract": "rigorousrag-continual-terminal-v1", "workflow_id": spec.workflow_id, "state": state, "evidence": dict(evidence)})


def _validate_benchmark(spec: ContinualWorkflowSpec, lineage: TrainingLineage, evidence: BenchmarkEvidence) -> None:
    if evidence.workflow_id != spec.workflow_id:
        raise ContinualValidationError("benchmark evidence belongs to another workflow")
    if evidence.candidate_artifact_sha256 != lineage.outcome.output_artifact_sha256:
        raise ContinualValidationError("benchmark evaluated a different candidate artifact")
    decision = evidence.base_decision
    if decision.owner_id != spec.owner_id or decision.batch_id != spec.feedback_batch.batch_id:
        raise ContinualValidationError("benchmark decision crosses owner/feedback identity")
    if decision.baseline_version != spec.baseline_version or decision.candidate_version != spec.candidate_version:
        raise ContinualValidationError("benchmark decision version identity differs from workflow")


def _validate_promotion(spec: ContinualWorkflowSpec, lineage: TrainingLineage, decision: ContinualPromotionDecision, receipt: PromotionReceipt) -> None:
    if receipt.workflow_id != spec.workflow_id or receipt.previous_version != spec.baseline_version or receipt.promoted_version != spec.candidate_version:
        raise ContinualValidationError("promotion receipt workflow/version identity differs")
    if receipt.candidate_artifact_sha256 != lineage.outcome.output_artifact_sha256 or receipt.decision_id != decision.decision_id:
        raise ContinualValidationError("promotion receipt artifact/decision identity differs")


def _fail(store: SQLiteContinualWorkflowStore, claim: WorkflowClaim, spec: ContinualWorkflowSpec, record: ContinualWorkflowRecord, error: ContinualValidationError, now: float) -> ContinualWorkflowRecord:
    receipt = _terminal(spec, "failed", {"failure_type": error.__class__.__name__, "state": record.state, "revision": record.revision})
    return store.transition(claim, expected_state=record.state, expected_revision=record.revision, new_state="failed", now=now, failure_type=error.__class__.__name__, terminal_receipt_sha256=receipt)


def advance_continual_workflow(spec: ContinualWorkflowSpec, *, store: SQLiteContinualWorkflowStore, build_backend: CandidateBuildBackend, benchmark_backend: BenchmarkBackend, promotion_backend: PromotionBackend, worker_id: str, now: float, lease_seconds: float = 3_600.0, max_steps: int = 16) -> ContinualWorkflowRecord:
    """Advance synchronously; backend errors stay retryable, identity errors fail closed."""
    if not isinstance(spec, ContinualWorkflowSpec):
        raise ValueError("spec must be ContinualWorkflowSpec")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 1_000:
        raise ValueError("max_steps is invalid")
    instant, duration = _time(now, "now"), _lease(lease_seconds)
    store.ensure(spec, now=instant)
    claim = store.claim(spec.workflow_id, worker_id=worker_id, now=instant, lease_seconds=duration)
    try:
        for _ in range(max_steps):
            claim = store.renew(claim, now=instant, lease_seconds=duration)
            record = store.get(spec.workflow_id)
            if record is None or record.owner_id != spec.owner_id or record.spec_sha256 != spec.spec_sha256:
                raise RuntimeError("continual workflow persisted identity differs from supplied spec")
            if record.terminal:
                return record
            try:
                if record.state == "detected":
                    if spec.adaptation_decision.action == "stable":
                        return store.transition(claim, expected_state="detected", expected_revision=record.revision, new_state="stable_held", now=instant, terminal_receipt_sha256=_terminal(spec, "stable_held", {"drift_evidence_sha256": spec.drift_evidence_sha256}))
                    store.transition(claim, expected_state="detected", expected_revision=record.revision, new_state="build_requested", now=instant)
                    continue
                if record.state == "build_requested":
                    outcome = build_backend.build(spec.training_request, workflow_id=spec.workflow_id, fencing_token=claim.fencing_token)
                    if not isinstance(outcome, TrainingOutcome):
                        raise ContinualValidationError("build backend returned invalid training outcome")
                    lineage = TrainingLineage.bind(spec.training_request, outcome)
                    store.transition(claim, expected_state="build_requested", expected_revision=record.revision, new_state="build_ready", now=instant, build_payload=_training_payload(lineage))
                    continue
                if record.state == "build_ready":
                    _lineage(spec, record.build_payload)
                    store.transition(claim, expected_state="build_ready", expected_revision=record.revision, new_state="benchmark_requested", now=instant)
                    continue
                if record.state == "benchmark_requested":
                    lineage = _lineage(spec, record.build_payload)
                    evidence = benchmark_backend.evaluate(spec, lineage, workflow_id=spec.workflow_id, fencing_token=claim.fencing_token)
                    if not isinstance(evidence, BenchmarkEvidence):
                        raise ContinualValidationError("benchmark backend returned invalid evidence")
                    _validate_benchmark(spec, lineage, evidence)
                    store.transition(claim, expected_state="benchmark_requested", expected_revision=record.revision, new_state="benchmark_ready", now=instant, benchmark_payload=_benchmark_payload(evidence))
                    continue
                if record.state == "benchmark_ready":
                    evidence = _benchmark(record.benchmark_payload)
                    decision = evaluate_continual_promotion(base=evidence.base_decision, evidence=evidence.continual_evidence, policy=spec.continual_policy)
                    if decision.policy_fingerprint != _digest(asdict(spec.continual_policy)):
                        raise ContinualValidationError("continual promotion policy fingerprint mismatch")
                    store.transition(claim, expected_state="benchmark_ready", expected_revision=record.revision, new_state="decision_ready", now=instant, decision_payload=_decision_payload(decision))
                    continue
                if record.state == "decision_ready":
                    lineage, evidence, decision = _lineage(spec, record.build_payload), _benchmark(record.benchmark_payload), _decision(record.decision_payload)
                    if decision.base_promotion_decision_id != evidence.base_decision.decision_id:
                        raise ContinualValidationError("continual decision no longer binds benchmark decision")
                    if not decision.eligible:
                        return store.transition(claim, expected_state="decision_ready", expected_revision=record.revision, new_state="held", now=instant, terminal_receipt_sha256=_terminal(spec, "held", {"decision_id": decision.decision_id, "reason_codes": list(decision.reason_codes), "benchmark_evidence_sha256": evidence.evidence_sha256}))
                    receipt = promotion_backend.promote(spec, lineage, evidence, decision, workflow_id=spec.workflow_id, expected_baseline_version=spec.baseline_version, fencing_token=claim.fencing_token)
                    if not isinstance(receipt, PromotionReceipt):
                        raise ContinualValidationError("promotion backend returned invalid receipt")
                    _validate_promotion(spec, lineage, decision, receipt)
                    return store.transition(claim, expected_state="decision_ready", expected_revision=record.revision, new_state="promoted", now=instant, promotion_payload=asdict(receipt), terminal_receipt_sha256=_terminal(spec, "promoted", {"decision_id": decision.decision_id, "publication_sha256": receipt.publication_sha256, "candidate_artifact_sha256": receipt.candidate_artifact_sha256}))
                raise ContinualValidationError(f"unsupported nonterminal workflow state: {record.state}")
            except ContinualValidationError as exc:
                return _fail(store, claim, spec, record, exc, instant)
        final = store.get(spec.workflow_id)
        if final is None:
            raise RuntimeError("continual workflow disappeared")
        return final
    finally:
        try:
            store.release(claim, now=instant)
        except RuntimeError:
            pass


def rollback_promoted_workflow(spec: ContinualWorkflowSpec, *, store: SQLiteContinualWorkflowStore, promotion_backend: PromotionBackend, worker_id: str, now: float, lease_seconds: float = 3_600.0) -> ContinualWorkflowRecord:
    instant, duration = _time(now, "now"), _lease(lease_seconds)
    observed = store.get(spec.workflow_id)
    if observed is None or observed.state != "promoted":
        raise ValueError("only a promoted workflow can be rolled back")
    promotion = _promotion(observed.promotion_payload)
    claim = store.claim(spec.workflow_id, worker_id=worker_id, now=instant, lease_seconds=duration)
    try:
        claim = store.renew(claim, now=instant, lease_seconds=duration)
        current = store.get(spec.workflow_id)
        if current is None or current.state != "promoted" or current.revision != observed.revision:
            raise RuntimeError("promotion state changed before rollback")
        receipt = promotion_backend.rollback(spec, promotion, workflow_id=spec.workflow_id, expected_current_version=spec.candidate_version, fencing_token=claim.fencing_token)
        if not isinstance(receipt, RollbackReceipt):
            raise ContinualValidationError("rollback backend returned invalid receipt")
        if receipt.workflow_id != spec.workflow_id or receipt.rolled_back_from_version != spec.candidate_version or receipt.restored_version != spec.baseline_version or receipt.promotion_publication_sha256 != promotion.publication_sha256:
            raise ContinualValidationError("rollback receipt does not restore governed baseline")
        return store.transition(claim, expected_state="promoted", expected_revision=current.revision, new_state="rolled_back", now=instant, rollback_payload=asdict(receipt), terminal_receipt_sha256=_terminal(spec, "rolled_back", {"promotion_publication_sha256": promotion.publication_sha256, "rollback_sha256": receipt.rollback_sha256}))
    finally:
        try:
            store.release(claim, now=instant)
        except RuntimeError:
            pass


__all__ = ["BenchmarkBackend", "BenchmarkEvidence", "CandidateBuildBackend", "ContinualValidationError", "ContinualWorkflowRecord", "ContinualWorkflowSpec", "PromotionBackend", "PromotionReceipt", "RollbackReceipt", "SQLiteContinualWorkflowStore", "WorkflowClaim", "advance_continual_workflow", "rollback_promoted_workflow"]
