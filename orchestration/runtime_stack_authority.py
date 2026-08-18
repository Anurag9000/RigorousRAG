"""Governed promotion, serving authority and monotonic rollback for runtime RAG stacks.

This layer is intentionally separate from vector/index migration. It governs which
already-compatible model/retriever/reranker/calibrator/fusion/planner stack may serve for
an owner/service/domain. Promotion consumes typed immutable evidence digests; rollback
creates a new authority revision and fencing token rather than rewinding history.
No models, datasets, providers, or deployment APIs are loaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_COMPONENT_KINDS = frozenset(
    {
        "dense_retriever",
        "sparse_retriever",
        "late_interaction_retriever",
        "reranker",
        "generator",
        "semantic_support",
        "query_router",
        "query_plan_ranker",
        "calibrator",
        "fusion_policy",
        "tokenizer",
        "document_model",
    }
)
_EVIDENCE_KINDS = frozenset(
    {
        "offline_quality",
        "semantic_citation_quality",
        "calibration_qualification",
        "calibration_currentness",
        "interleaving_quality",
        "resource_budget",
        "security_review",
        "license_governance",
        "compatibility",
        "dataset_governance",
        "operator_review",
    }
)
_HEX = frozenset("0123456789abcdef")
_MAX_DECISION_TTL_SECONDS = 30 * 24 * 60 * 60


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
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


def _git_revision(value: Any) -> str:
    selected = _text(value, "source_revision", 64).lower()
    if len(selected) not in (40, 64) or any(ch not in _HEX for ch in selected):
        raise ValueError("source_revision must be a 40- or 64-character hexadecimal Git object id")
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


def _positive_float(value: Any, label: str, *, maximum: float | None = None) -> float:
    selected = _time(value, label)
    if selected <= 0.0 or (maximum is not None and selected > maximum):
        raise ValueError(f"{label} must be positive" + (f" and <= {maximum}" if maximum is not None else ""))
    return selected


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _component_kind(value: Any) -> str:
    selected = _text(value, "component kind", 100).lower()
    if selected not in _COMPONENT_KINDS:
        raise ValueError(f"unsupported runtime component kind {selected!r}")
    return selected


def _evidence_kind(value: Any) -> str:
    selected = _text(value, "evidence kind", 100).lower()
    if selected not in _EVIDENCE_KINDS:
        raise ValueError(f"unsupported promotion evidence kind {selected!r}")
    return selected


@dataclass(frozen=True)
class RuntimeComponent:
    kind: str
    component_id: str
    artifact_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _component_kind(self.kind))
        object.__setattr__(self, "component_id", _text(self.component_id, "component_id", 500))
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, "artifact_sha256"))
        object.__setattr__(self, "contract_sha256", _sha(self.contract_sha256, "contract_sha256"))

    @property
    def component_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-runtime-component/v1", **asdict(self)})


@dataclass(frozen=True)
class RuntimeStackArtifact:
    stack_id: str
    components: tuple[RuntimeComponent, ...]
    retrieval_contract_sha256: str
    generation_contract_sha256: str
    compatibility_sha256: str
    source_revision: str
    stack_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_id", _text(self.stack_id, "stack_id", 500))
        components = tuple(self.components)
        if not components or len(components) > 100 or any(not isinstance(value, RuntimeComponent) for value in components):
            raise ValueError("components must be a non-empty bounded RuntimeComponent sequence")
        keys = [(value.kind, value.component_id) for value in components]
        if len(keys) != len(set(keys)):
            raise ValueError("runtime components must be unique by kind/component_id")
        components = tuple(sorted(components, key=lambda value: (value.kind, value.component_id)))
        object.__setattr__(self, "components", components)
        for name in ("retrieval_contract_sha256", "generation_contract_sha256", "compatibility_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        expected = _digest(self._payload())
        provided = _sha(self.stack_sha256, "stack_sha256")
        if expected != provided:
            raise ValueError("stack_sha256 does not match runtime stack artifact")
        object.__setattr__(self, "stack_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-runtime-stack/v1",
            "stack_id": self.stack_id,
            "components": [asdict(value) for value in self.components],
            "retrieval_contract_sha256": self.retrieval_contract_sha256,
            "generation_contract_sha256": self.generation_contract_sha256,
            "compatibility_sha256": self.compatibility_sha256,
            "source_revision": self.source_revision,
        }

    @classmethod
    def build(
        cls,
        *,
        stack_id: str,
        components: Sequence[RuntimeComponent],
        retrieval_contract_sha256: str,
        generation_contract_sha256: str,
        compatibility_sha256: str,
        source_revision: str,
    ) -> "RuntimeStackArtifact":
        selected = tuple(components)
        if not selected or any(not isinstance(value, RuntimeComponent) for value in selected):
            raise ValueError("components must contain RuntimeComponent values")
        ordered = tuple(sorted(selected, key=lambda value: (value.kind, value.component_id)))
        payload = {
            "schema": "rigorousrag-runtime-stack/v1",
            "stack_id": _text(stack_id, "stack_id", 500),
            "components": [asdict(value) for value in ordered],
            "retrieval_contract_sha256": _sha(retrieval_contract_sha256, "retrieval_contract_sha256"),
            "generation_contract_sha256": _sha(generation_contract_sha256, "generation_contract_sha256"),
            "compatibility_sha256": _sha(compatibility_sha256, "compatibility_sha256"),
            "source_revision": _git_revision(source_revision),
        }
        return cls(
            stack_id=payload["stack_id"],
            components=ordered,
            retrieval_contract_sha256=payload["retrieval_contract_sha256"],
            generation_contract_sha256=payload["generation_contract_sha256"],
            compatibility_sha256=payload["compatibility_sha256"],
            source_revision=payload["source_revision"],
            stack_sha256=_digest(payload),
        )


@dataclass(frozen=True)
class RuntimePromotionEvidence:
    kind: str
    evidence_sha256: str
    stack_sha256: str
    valid_from: float
    expires_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _evidence_kind(self.kind))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        object.__setattr__(self, "stack_sha256", _sha(self.stack_sha256, "stack_sha256"))
        start = _time(self.valid_from, "valid_from")
        object.__setattr__(self, "valid_from", start)
        if self.expires_at is not None:
            end = _time(self.expires_at, "expires_at")
            if end <= start:
                raise ValueError("expires_at must be later than valid_from")
            object.__setattr__(self, "expires_at", end)

    @property
    def row_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-runtime-promotion-evidence/v1", **asdict(self)})

    def current_at(self, now: float) -> bool:
        instant = _time(now, "now")
        return self.valid_from <= instant and (self.expires_at is None or instant < self.expires_at)


@dataclass(frozen=True)
class RuntimePromotionPolicy:
    policy_id: str
    required_evidence_kinds: tuple[str, ...]
    require_compatibility_digest_match: bool = True
    decision_ttl_seconds: float = 3600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        kinds = tuple(sorted({_evidence_kind(value) for value in self.required_evidence_kinds}))
        if not kinds:
            raise ValueError("required_evidence_kinds must be non-empty")
        object.__setattr__(self, "required_evidence_kinds", kinds)
        if not isinstance(self.require_compatibility_digest_match, bool):
            raise ValueError("require_compatibility_digest_match must be boolean")
        object.__setattr__(
            self,
            "decision_ttl_seconds",
            _positive_float(self.decision_ttl_seconds, "decision_ttl_seconds", maximum=_MAX_DECISION_TTL_SECONDS),
        )

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-runtime-promotion-policy/v2", **asdict(self)})


@dataclass(frozen=True)
class RuntimePromotionDecision:
    stack_sha256: str
    policy_sha256: str
    evidence_row_sha256s: tuple[tuple[str, str], ...]
    eligible: bool
    reason_codes: tuple[str, ...]
    decided_at: float
    valid_until: float
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_sha256", _sha(self.stack_sha256, "stack_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        rows = tuple(sorted((_evidence_kind(kind), _sha(digest, "evidence row sha256")) for kind, digest in self.evidence_row_sha256s))
        if len({kind for kind, _ in rows}) != len(rows):
            raise ValueError("promotion decision may contain at most one evidence row per kind")
        object.__setattr__(self, "evidence_row_sha256s", rows)
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        reasons = tuple(sorted({_text(value, "reason code", 200) for value in self.reason_codes}))
        if self.eligible and reasons:
            raise ValueError("eligible promotion decision may not contain failure reasons")
        if not self.eligible and not reasons:
            raise ValueError("ineligible promotion decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        decided = _time(self.decided_at, "decided_at")
        valid_until = _time(self.valid_until, "valid_until")
        if valid_until <= decided:
            raise ValueError("valid_until must be later than decided_at")
        object.__setattr__(self, "decided_at", decided)
        object.__setattr__(self, "valid_until", valid_until)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match runtime promotion decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-runtime-promotion-decision/v2",
            "stack_sha256": self.stack_sha256,
            "policy_sha256": self.policy_sha256,
            "evidence_row_sha256s": self.evidence_row_sha256s,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
            "decided_at": self.decided_at,
            "valid_until": self.valid_until,
        }

    def current_at(self, now: float) -> bool:
        instant = _time(now, "now")
        return self.decided_at <= instant < self.valid_until


def decide_runtime_promotion(
    stack: RuntimeStackArtifact,
    *,
    evidence: Sequence[RuntimePromotionEvidence],
    policy: RuntimePromotionPolicy,
    now: float,
    current_compatibility_sha256: str | None = None,
) -> RuntimePromotionDecision:
    if not isinstance(stack, RuntimeStackArtifact):
        raise ValueError("stack must be RuntimeStackArtifact")
    if not isinstance(policy, RuntimePromotionPolicy):
        raise ValueError("policy must be RuntimePromotionPolicy")
    instant = _time(now, "now")
    rows = tuple(evidence)
    if any(not isinstance(value, RuntimePromotionEvidence) for value in rows):
        raise ValueError("evidence contains invalid values")
    if any(value.stack_sha256 != stack.stack_sha256 for value in rows):
        raise ValueError("promotion evidence is bound to a different runtime stack")
    by_kind: dict[str, RuntimePromotionEvidence] = {}
    for value in rows:
        if value.kind in by_kind:
            raise ValueError("promotion evidence contains duplicate kinds")
        by_kind[value.kind] = value

    reasons: list[str] = []
    selected_rows: list[tuple[str, str]] = []
    for kind in policy.required_evidence_kinds:
        value = by_kind.get(kind)
        if value is None:
            reasons.append(f"missing_required_evidence:{kind}")
            continue
        selected_rows.append((kind, value.row_sha256))
        if not value.current_at(instant):
            reasons.append(f"stale_or_not_yet_valid_evidence:{kind}")
    for kind, value in sorted(by_kind.items()):
        if kind not in policy.required_evidence_kinds:
            selected_rows.append((kind, value.row_sha256))
    if policy.require_compatibility_digest_match:
        if current_compatibility_sha256 is None:
            reasons.append("current_compatibility_digest_missing")
        elif _sha(current_compatibility_sha256, "current_compatibility_sha256") != stack.compatibility_sha256:
            reasons.append("runtime_stack_compatibility_mismatch")

    payload = {
        "schema": "rigorousrag-runtime-promotion-decision/v2",
        "stack_sha256": stack.stack_sha256,
        "policy_sha256": policy.policy_sha256,
        "evidence_row_sha256s": tuple(sorted(selected_rows)),
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
        "decided_at": instant,
        "valid_until": instant + policy.decision_ttl_seconds,
    }
    return RuntimePromotionDecision(**payload, decision_sha256=_digest(payload))


@dataclass(frozen=True)
class RuntimeAuthorityRecord:
    owner_id: str
    service_id: str
    domain_id: str
    stack_sha256: str
    authority_revision: int
    fencing_token: int
    action: str
    authority_evidence_sha256: str
    updated_at: float

    def __post_init__(self) -> None:
        for name in ("owner_id", "service_id", "domain_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "stack_sha256", _sha(self.stack_sha256, "stack_sha256"))
        object.__setattr__(self, "authority_revision", _positive_int(self.authority_revision, "authority_revision"))
        object.__setattr__(self, "fencing_token", _positive_int(self.fencing_token, "fencing_token"))
        if self.action not in {"promote", "rollback"}:
            raise ValueError("authority action is invalid")
        object.__setattr__(self, "authority_evidence_sha256", _sha(self.authority_evidence_sha256, "authority_evidence_sha256"))
        object.__setattr__(self, "updated_at", _time(self.updated_at, "updated_at"))


@dataclass(frozen=True)
class RuntimeRollbackRequest:
    owner_id: str
    service_id: str
    domain_id: str
    target_authority_revision: int
    reason_sha256: str
    actor_sha256: str
    requested_at: float

    def __post_init__(self) -> None:
        for name in ("owner_id", "service_id", "domain_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "target_authority_revision", _positive_int(self.target_authority_revision, "target_authority_revision"))
        object.__setattr__(self, "reason_sha256", _sha(self.reason_sha256, "reason_sha256"))
        object.__setattr__(self, "actor_sha256", _sha(self.actor_sha256, "actor_sha256"))
        object.__setattr__(self, "requested_at", _time(self.requested_at, "requested_at"))

    @property
    def request_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-runtime-rollback-request/v1", **asdict(self)})


class SQLiteRuntimeStackAuthorityStore:
    """Immutable stack registry plus CAS serving authority with monotonic rollback."""

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
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runtime_stack_registry (
                    stack_sha256 TEXT PRIMARY KEY,
                    stack_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runtime_stack_authority (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    stack_sha256 TEXT NOT NULL,
                    authority_revision INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    authority_evidence_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id,service_id,domain_id)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runtime_stack_history (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    authority_revision INTEGER NOT NULL,
                    stack_sha256 TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    authority_evidence_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id,service_id,domain_id,authority_revision)
                )"""
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> RuntimeAuthorityRecord:
        return RuntimeAuthorityRecord(
            row["owner_id"],
            row["service_id"],
            row["domain_id"],
            row["stack_sha256"],
            int(row["authority_revision"]),
            int(row["fencing_token"]),
            row["action"],
            row["authority_evidence_sha256"],
            float(row["updated_at"]),
        )

    @staticmethod
    def _stack_json(stack: RuntimeStackArtifact) -> str:
        return _canonical({**stack._payload(), "stack_sha256": stack.stack_sha256}).decode("utf-8")

    @staticmethod
    def _decode_stack(raw: str, expected_sha256: str) -> RuntimeStackArtifact:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise RuntimeError("persisted runtime stack is invalid")
        components = tuple(RuntimeComponent(**row) for row in value.get("components", ()))
        stack = RuntimeStackArtifact(
            stack_id=value["stack_id"],
            components=components,
            retrieval_contract_sha256=value["retrieval_contract_sha256"],
            generation_contract_sha256=value["generation_contract_sha256"],
            compatibility_sha256=value["compatibility_sha256"],
            source_revision=value["source_revision"],
            stack_sha256=value["stack_sha256"],
        )
        if stack.stack_sha256 != _sha(expected_sha256, "expected stack sha256"):
            raise RuntimeError("persisted runtime stack digest is corrupt")
        return stack

    def register_stack(self, stack: RuntimeStackArtifact) -> RuntimeStackArtifact:
        if not isinstance(stack, RuntimeStackArtifact):
            raise ValueError("stack must be RuntimeStackArtifact")
        encoded = self._stack_json(stack)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT stack_json FROM runtime_stack_registry WHERE stack_sha256=?", (stack.stack_sha256,)).fetchone()
            if row is not None:
                if row["stack_json"] != encoded:
                    raise RuntimeError("runtime stack identity collision")
                return stack
            connection.execute("INSERT INTO runtime_stack_registry(stack_sha256,stack_json) VALUES(?,?)", (stack.stack_sha256, encoded))
        return stack

    def get_stack(self, stack_sha256: str) -> RuntimeStackArtifact | None:
        selected = _sha(stack_sha256, "stack_sha256")
        with self._connect() as connection:
            row = connection.execute("SELECT stack_json FROM runtime_stack_registry WHERE stack_sha256=?", (selected,)).fetchone()
        return None if row is None else self._decode_stack(row["stack_json"], selected)

    def current(self, *, owner_id: str, service_id: str, domain_id: str) -> RuntimeAuthorityRecord | None:
        owner, service, domain = (_text(owner_id, "owner_id"), _text(service_id, "service_id"), _text(domain_id, "domain_id"))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_stack_authority WHERE owner_id=? AND service_id=? AND domain_id=?",
                (owner, service, domain),
            ).fetchone()
        return None if row is None else self._record(row)

    def history(self, *, owner_id: str, service_id: str, domain_id: str, limit: int = 100) -> tuple[RuntimeAuthorityRecord, ...]:
        owner, service, domain = (_text(owner_id, "owner_id"), _text(service_id, "service_id"), _text(domain_id, "domain_id"))
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10,000")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM runtime_stack_history WHERE owner_id=? AND service_id=? AND domain_id=?
                   ORDER BY authority_revision DESC LIMIT ?""",
                (owner, service, domain, limit),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def promote(
        self,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        stack: RuntimeStackArtifact,
        decision: RuntimePromotionDecision,
        expected_authority_revision: int | None,
        now: float,
    ) -> RuntimeAuthorityRecord:
        owner, service, domain = (_text(owner_id, "owner_id"), _text(service_id, "service_id"), _text(domain_id, "domain_id"))
        if not isinstance(stack, RuntimeStackArtifact) or not isinstance(decision, RuntimePromotionDecision):
            raise ValueError("stack/decision types are invalid")
        if not decision.eligible or decision.stack_sha256 != stack.stack_sha256:
            raise ValueError("runtime promotion requires an eligible decision for the exact stack")
        if expected_authority_revision is not None:
            _positive_int(expected_authority_revision, "expected_authority_revision", allow_zero=True)
        timestamp = _time(now, "now")
        if not decision.current_at(timestamp):
            raise ValueError("runtime promotion decision is stale or not yet valid")
        encoded = self._stack_json(stack)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            registered = connection.execute("SELECT stack_json FROM runtime_stack_registry WHERE stack_sha256=?", (stack.stack_sha256,)).fetchone()
            if registered is None:
                connection.execute("INSERT INTO runtime_stack_registry(stack_sha256,stack_json) VALUES(?,?)", (stack.stack_sha256, encoded))
            elif registered["stack_json"] != encoded:
                raise RuntimeError("runtime stack registry identity collision")
            row = connection.execute(
                "SELECT * FROM runtime_stack_authority WHERE owner_id=? AND service_id=? AND domain_id=?",
                (owner, service, domain),
            ).fetchone()
            if row is None:
                if expected_authority_revision not in (None, 0):
                    raise RuntimeError("runtime stack bootstrap CAS failed")
                revision, fence = 1, 1
            else:
                current = self._record(row)
                if expected_authority_revision is None or expected_authority_revision != current.authority_revision:
                    raise RuntimeError("runtime stack promotion CAS failed")
                if current.stack_sha256 == stack.stack_sha256:
                    return current
                revision, fence = current.authority_revision + 1, current.fencing_token + 1
            record = RuntimeAuthorityRecord(owner, service, domain, stack.stack_sha256, revision, fence, "promote", decision.decision_sha256, timestamp)
            if row is None:
                connection.execute(
                    "INSERT INTO runtime_stack_authority VALUES(?,?,?,?,?,?,?,?,?)",
                    (owner, service, domain, record.stack_sha256, revision, fence, record.action, record.authority_evidence_sha256, timestamp),
                )
            else:
                current = self._record(row)
                changed = connection.execute(
                    """UPDATE runtime_stack_authority SET stack_sha256=?,authority_revision=?,fencing_token=?,action=?,authority_evidence_sha256=?,updated_at=?
                       WHERE owner_id=? AND service_id=? AND domain_id=? AND authority_revision=? AND fencing_token=?""",
                    (record.stack_sha256, revision, fence, record.action, record.authority_evidence_sha256, timestamp, owner, service, domain, expected_authority_revision, current.fencing_token),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("runtime stack promotion lost CAS race")
            connection.execute(
                "INSERT INTO runtime_stack_history VALUES(?,?,?,?,?,?,?,?,?)",
                (owner, service, domain, revision, record.stack_sha256, fence, record.action, record.authority_evidence_sha256, timestamp),
            )
            return record

    def rollback(
        self,
        request: RuntimeRollbackRequest,
        *,
        expected_authority_revision: int,
        current_compatibility_sha256: str,
        now: float,
    ) -> RuntimeAuthorityRecord:
        if not isinstance(request, RuntimeRollbackRequest):
            raise ValueError("request must be RuntimeRollbackRequest")
        expected = _positive_int(expected_authority_revision, "expected_authority_revision")
        compatibility = _sha(current_compatibility_sha256, "current_compatibility_sha256")
        timestamp = _time(now, "now")
        if request.requested_at > timestamp:
            raise ValueError("rollback request is future-dated")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runtime_stack_authority WHERE owner_id=? AND service_id=? AND domain_id=?",
                (request.owner_id, request.service_id, request.domain_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("cannot rollback before a runtime stack is promoted")
            current = self._record(row)
            if current.authority_revision != expected:
                raise RuntimeError("runtime rollback CAS failed")
            target_row = connection.execute(
                """SELECT * FROM runtime_stack_history WHERE owner_id=? AND service_id=? AND domain_id=? AND authority_revision=?""",
                (request.owner_id, request.service_id, request.domain_id, request.target_authority_revision),
            ).fetchone()
            if target_row is None:
                raise ValueError("rollback target authority revision does not exist")
            target = self._record(target_row)
            if target.authority_revision >= current.authority_revision:
                raise ValueError("rollback target must be a prior authority revision")
            registered = connection.execute("SELECT stack_json FROM runtime_stack_registry WHERE stack_sha256=?", (target.stack_sha256,)).fetchone()
            if registered is None:
                raise RuntimeError("rollback target stack is missing from the immutable registry")
            target_stack = self._decode_stack(registered["stack_json"], target.stack_sha256)
            if target_stack.compatibility_sha256 != compatibility:
                raise ValueError("rollback target stack is incompatible with the current serving environment")
            revision, fence = current.authority_revision + 1, current.fencing_token + 1
            record = RuntimeAuthorityRecord(
                request.owner_id,
                request.service_id,
                request.domain_id,
                target.stack_sha256,
                revision,
                fence,
                "rollback",
                request.request_sha256,
                timestamp,
            )
            changed = connection.execute(
                """UPDATE runtime_stack_authority SET stack_sha256=?,authority_revision=?,fencing_token=?,action='rollback',authority_evidence_sha256=?,updated_at=?
                   WHERE owner_id=? AND service_id=? AND domain_id=? AND authority_revision=? AND fencing_token=?""",
                (record.stack_sha256, revision, fence, request.request_sha256, timestamp, request.owner_id, request.service_id, request.domain_id, current.authority_revision, current.fencing_token),
            ).rowcount
            if changed != 1:
                raise RuntimeError("runtime rollback lost CAS race")
            connection.execute(
                "INSERT INTO runtime_stack_history VALUES(?,?,?,?,?,?,?,?,?)",
                (request.owner_id, request.service_id, request.domain_id, revision, record.stack_sha256, fence, "rollback", request.request_sha256, timestamp),
            )
            return record

    def assert_runtime_authority(
        self,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        stack_sha256: str,
        fencing_token: int,
    ) -> RuntimeStackArtifact:
        current = self.current(owner_id=owner_id, service_id=service_id, domain_id=domain_id)
        if current is None:
            raise RuntimeError("no runtime stack authority is established")
        if current.stack_sha256 != _sha(stack_sha256, "stack_sha256") or current.fencing_token != _positive_int(fencing_token, "fencing_token"):
            raise RuntimeError("stale or non-authoritative runtime stack")
        stack = self.get_stack(current.stack_sha256)
        if stack is None:
            raise RuntimeError("authoritative runtime stack is missing from registry")
        return stack


__all__ = [
    "RuntimeAuthorityRecord",
    "RuntimeComponent",
    "RuntimePromotionDecision",
    "RuntimePromotionEvidence",
    "RuntimePromotionPolicy",
    "RuntimeRollbackRequest",
    "RuntimeStackArtifact",
    "SQLiteRuntimeStackAuthorityStore",
    "decide_runtime_promotion",
]
