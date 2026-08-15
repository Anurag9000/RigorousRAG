"""Unified immutable model/policy/artifact promotion lifecycle.

The lifecycle coordinates candidate -> shadow -> canary -> production -> retired states
for retrievers, rerankers, routers, planners and extractors. It records immutable
artifact/evaluation identities and explicit rollback targets; it does not perform model
training or assume an evaluation report is sufficient for promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

_ALLOWED_STATES = ("candidate", "shadow", "canary", "production", "retired", "rejected")
_ALLOWED_KINDS = frozenset({"retriever", "reranker", "router", "planner", "extractor", "entailment", "multimodal", "policy", "other"})


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class LifecycleArtifact:
    artifact_id: str
    kind: str
    version: str
    artifact_sha256: str
    training_or_build_sha256: str
    dataset_sha256: str = ""
    config_sha256: str = ""
    parent_artifact_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported lifecycle artifact kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "version", _text(self.version, "version", 100))
        for name in ("artifact_sha256", "training_or_build_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("dataset_sha256", "config_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name, allow_empty=True))
        object.__setattr__(self, "parent_artifact_id", _text(self.parent_artifact_id, "parent_artifact_id", 256, allow_empty=True))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 500) for k, v in self.metadata.items()})

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class PromotionEvidence:
    report_sha256: str
    policy_sha256: str
    eligible: bool
    quality_score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_sha256", _digest(self.report_sha256, "report_sha256"))
        object.__setattr__(self, "policy_sha256", _digest(self.policy_sha256, "policy_sha256"))
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        if isinstance(self.quality_score, bool):
            raise ValueError("quality_score must be finite")
        score = float(self.quality_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("quality_score must lie in [0,1]")
        object.__setattr__(self, "quality_score", score)
        if len(self.reasons) > 100:
            raise ValueError("too many promotion reasons")
        object.__setattr__(self, "reasons", tuple(_text(item, "promotion reason", 500) for item in self.reasons))


@dataclass(frozen=True)
class LifecycleRecord:
    artifact: LifecycleArtifact
    state: str
    sequence: int
    created_at: float
    evidence: PromotionEvidence | None = None
    canary_fraction: float = 0.0
    rollback_artifact_id: str = ""
    supersedes: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, LifecycleArtifact):
            raise ValueError("artifact must be LifecycleArtifact")
        state = _text(self.state, "state", 32).lower()
        if state not in _ALLOWED_STATES:
            raise ValueError("unsupported lifecycle state")
        object.__setattr__(self, "state", state)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence is invalid")
        timestamp = float(self.created_at)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("created_at is invalid")
        object.__setattr__(self, "created_at", timestamp)
        fraction = float(self.canary_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("canary_fraction is invalid")
        if state == "canary" and not 0.0 < fraction < 1.0:
            raise ValueError("canary state requires a fraction strictly between 0 and 1")
        if state == "production" and fraction not in {0.0, 1.0}:
            raise ValueError("production canary_fraction must be 0 or 1")
        object.__setattr__(self, "canary_fraction", fraction)
        for name in ("rollback_artifact_id", "supersedes", "reason"):
            maximum = 1000 if name == "reason" else 256
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum, allow_empty=True))


class ArtifactLifecycleRegistry:
    """Append-only in-memory reference state machine; durable stores can mirror it."""

    def __init__(self) -> None:
        self._artifacts: dict[str, LifecycleArtifact] = {}
        self._history: dict[str, list[LifecycleRecord]] = {}
        self._production_by_kind: dict[str, str] = {}
        self._lock = threading.RLock()

    def register_candidate(self, artifact: LifecycleArtifact) -> LifecycleRecord:
        with self._lock:
            existing = self._artifacts.get(artifact.artifact_id)
            if existing is not None and existing != artifact:
                raise ValueError("artifact ID collision")
            if existing is not None:
                return self._history[artifact.artifact_id][-1]
            self._artifacts[artifact.artifact_id] = artifact
            record = LifecycleRecord(artifact, "candidate", 1, time.time())
            self._history[artifact.artifact_id] = [record]
            return record

    def current(self, artifact_id: str) -> LifecycleRecord:
        return self._history[_text(artifact_id, "artifact_id", 256)][-1]

    def _transition(self, artifact_id: str, state: str, *, evidence: PromotionEvidence | None = None, canary_fraction: float = 0.0, rollback_artifact_id: str = "", supersedes: str = "", reason: str = "") -> LifecycleRecord:
        with self._lock:
            current = self.current(artifact_id)
            allowed = {
                "candidate": {"shadow", "rejected", "retired"},
                "shadow": {"canary", "production", "rejected", "retired"},
                "canary": {"production", "shadow", "rejected", "retired"},
                "production": {"retired"},
                "rejected": {"retired"},
                "retired": set(),
            }
            if state not in allowed[current.state]:
                raise ValueError(f"invalid lifecycle transition {current.state}->{state}")
            if state in {"canary", "production"} and (evidence is None or not evidence.eligible):
                raise ValueError("canary/production promotion requires eligible evidence")
            if state == "canary" and not rollback_artifact_id:
                rollback_artifact_id = self._production_by_kind.get(current.artifact.kind, "")
            if state == "production":
                prior = self._production_by_kind.get(current.artifact.kind, "")
                if prior and prior != artifact_id:
                    rollback_artifact_id = rollback_artifact_id or prior
                    supersedes = supersedes or prior
                    prior_current = self.current(prior)
                    retirement = LifecycleRecord(prior_current.artifact, "retired", prior_current.sequence + 1, time.time(), prior_current.evidence, 0.0, prior_current.rollback_artifact_id, prior_current.supersedes, f"superseded_by:{artifact_id}")
                    self._history[prior].append(retirement)
                self._production_by_kind[current.artifact.kind] = artifact_id
            record = LifecycleRecord(current.artifact, state, current.sequence + 1, time.time(), evidence, canary_fraction, rollback_artifact_id, supersedes, reason)
            self._history[artifact_id].append(record)
            return record

    def shadow(self, artifact_id: str, *, evidence: PromotionEvidence | None = None) -> LifecycleRecord:
        return self._transition(artifact_id, "shadow", evidence=evidence)

    def canary(self, artifact_id: str, *, evidence: PromotionEvidence, fraction: float) -> LifecycleRecord:
        return self._transition(artifact_id, "canary", evidence=evidence, canary_fraction=fraction)

    def promote(self, artifact_id: str, *, evidence: PromotionEvidence) -> LifecycleRecord:
        return self._transition(artifact_id, "production", evidence=evidence, canary_fraction=1.0)

    def reject(self, artifact_id: str, *, reason: str) -> LifecycleRecord:
        return self._transition(artifact_id, "rejected", reason=reason)

    def rollback(self, artifact_id: str, *, evidence: PromotionEvidence, reason: str) -> LifecycleRecord:
        with self._lock:
            current = self.current(artifact_id)
            target = current.rollback_artifact_id
            if current.state not in {"canary", "production"} or not target:
                raise ValueError("artifact has no rollback target")
            if target not in self._artifacts:
                raise ValueError("rollback target is unavailable")
            if current.state == "production":
                retirement = self._transition(artifact_id, "retired", reason=f"rollback:{reason}")
                del retirement
                target_current = self.current(target)
                # Retired production artifacts can be restored only through an explicit new
                # production record with fresh promotion evidence.
                record = LifecycleRecord(target_current.artifact, "production", target_current.sequence + 1, time.time(), evidence, 1.0, artifact_id, target_current.supersedes, f"rollback_from:{artifact_id}:{reason}")
                self._history[target].append(record)
                self._production_by_kind[target_current.artifact.kind] = target
                return record
            return self._transition(artifact_id, "shadow", evidence=evidence, reason=f"canary_rollback:{reason}")

    def history(self, artifact_id: str) -> tuple[LifecycleRecord, ...]:
        return tuple(self._history[_text(artifact_id, "artifact_id", 256)])

    def production(self, kind: str) -> LifecycleArtifact | None:
        selected = _text(kind, "kind", 64).lower()
        artifact_id = self._production_by_kind.get(selected)
        return self._artifacts.get(artifact_id) if artifact_id else None


__all__ = [
    "ArtifactLifecycleRegistry", "LifecycleArtifact", "LifecycleRecord", "PromotionEvidence",
]
