"""Self-verifying promotion evidence for advanced RAG artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.advanced_rag_receipts import AdvancedEvaluationReceipt, qualify_advanced_artifact_with_receipt
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_artifacts import AdvancedArtifactManifest, AdvancedArtifactPromotionReceipt, MetricQualificationPolicy
from training.advanced_rag_manifest_integrity import assert_advanced_manifest_self_consistent

_MAX_BYTES = 16 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class AdvancedPromotionEvidence:
    artifact_sha256: str
    policy_sha256: str
    evaluation_receipt_sha256: str
    metrics_sha256: str
    promoted: bool
    reason_codes: tuple[str, ...]
    primitive_receipt_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "policy_sha256", "evaluation_receipt_sha256", "metrics_sha256", "primitive_receipt_sha256", "evidence_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.promoted, bool):
            raise ValueError("promoted must be boolean")
        reasons = tuple(sorted({str(value).strip() for value in self.reason_codes}))
        if any(not value for value in reasons):
            raise ValueError("promotion reason codes are invalid")
        if self.promoted and reasons:
            raise ValueError("promoted evidence may not contain failure reasons")
        if not self.promoted and not reasons:
            raise ValueError("blocked promotion evidence requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected_primitive = _digest({
            "schema": "rigorousrag-advanced-artifact-promotion/v1",
            "artifact_sha256": self.artifact_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "promoted": self.promoted,
            "reason_codes": list(self.reason_codes),
            "metrics_sha256": self.metrics_sha256,
        })
        if expected_primitive != self.primitive_receipt_sha256:
            raise ValueError("nested advanced artifact promotion receipt digest mismatch")
        if _digest(self._payload()) != self.evidence_sha256:
            raise ValueError("advanced promotion evidence digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-advanced-promotion-evidence/v1",
            "artifact_sha256": self.artifact_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "metrics_sha256": self.metrics_sha256,
            "promoted": self.promoted,
            "reason_codes": self.reason_codes,
            "primitive_receipt_sha256": self.primitive_receipt_sha256,
        }

    def primitive_receipt(self) -> AdvancedArtifactPromotionReceipt:
        return AdvancedArtifactPromotionReceipt(
            self.artifact_sha256,
            self.policy_sha256,
            self.evaluation_receipt_sha256,
            self.metrics_sha256,
            self.promoted,
            self.reason_codes,
            self.primitive_receipt_sha256,
        )


def build_advanced_promotion_evidence(manifest: AdvancedArtifactManifest, evaluation: AdvancedEvaluationReceipt, policy: MetricQualificationPolicy) -> AdvancedPromotionEvidence:
    assert_advanced_manifest_self_consistent(manifest)
    primitive = qualify_advanced_artifact_with_receipt(manifest, evaluation, policy)
    metrics_sha = _digest({str(key): float(value) for key, value in sorted(evaluation.metrics.items())})
    if primitive.metrics_sha256 != metrics_sha:
        raise ValueError("primitive promotion receipt metrics digest differs from evaluation metrics")
    unsigned = {
        "schema": "rigorousrag-advanced-promotion-evidence/v1",
        "artifact_sha256": primitive.artifact_sha256,
        "policy_sha256": primitive.policy_sha256,
        "evaluation_receipt_sha256": primitive.evaluation_receipt_sha256,
        "metrics_sha256": metrics_sha,
        "promoted": primitive.promoted,
        "reason_codes": tuple(sorted(primitive.reason_codes)),
        "primitive_receipt_sha256": primitive.receipt_sha256,
    }
    return AdvancedPromotionEvidence(
        artifact_sha256=primitive.artifact_sha256,
        policy_sha256=primitive.policy_sha256,
        evaluation_receipt_sha256=primitive.evaluation_receipt_sha256,
        metrics_sha256=metrics_sha,
        promoted=primitive.promoted,
        reason_codes=tuple(sorted(primitive.reason_codes)),
        primitive_receipt_sha256=primitive.receipt_sha256,
        evidence_sha256=_digest(unsigned),
    )


def write_advanced_promotion_evidence(path: str | Path, evidence: AdvancedPromotionEvidence) -> str:
    if not isinstance(evidence, AdvancedPromotionEvidence):
        raise ValueError("evidence must be AdvancedPromotionEvidence")
    destination = safe_advanced_path(path, label="promotion evidence destination", must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {**evidence._payload(), "evidence_sha256": evidence.evidence_sha256}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return evidence.evidence_sha256


def read_advanced_promotion_evidence(path: str | Path) -> AdvancedPromotionEvidence:
    source = safe_advanced_path(path, label="promotion evidence", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("promotion evidence must be a bounded regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("promotion evidence is not strict JSON") from exc
    required = {"schema", "artifact_sha256", "policy_sha256", "evaluation_receipt_sha256", "metrics_sha256", "promoted", "reason_codes", "primitive_receipt_sha256", "evidence_sha256"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload["schema"] != "rigorousrag-advanced-promotion-evidence/v1":
        raise ValueError("unsupported or malformed promotion evidence")
    return AdvancedPromotionEvidence(
        artifact_sha256=payload["artifact_sha256"],
        policy_sha256=payload["policy_sha256"],
        evaluation_receipt_sha256=payload["evaluation_receipt_sha256"],
        metrics_sha256=payload["metrics_sha256"],
        promoted=payload["promoted"],
        reason_codes=tuple(payload["reason_codes"]),
        primitive_receipt_sha256=payload["primitive_receipt_sha256"],
        evidence_sha256=payload["evidence_sha256"],
    )


__all__ = ["AdvancedPromotionEvidence", "build_advanced_promotion_evidence", "read_advanced_promotion_evidence", "write_advanced_promotion_evidence"]
