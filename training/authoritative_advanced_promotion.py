"""Production promotion evidence bound to verified result artifacts and exact policy values.

The compact ``AdvancedPromotionEvidence`` remains the primitive metric-policy decision. This
module wraps it with authoritative evaluation evidence and the exact normalized min/max policy
maps. Runtime assertion reconstructs the policy, re-verifies every v2 result artifact through
the evaluation evidence, and re-runs qualification; a caller cannot manufacture a promoted
boolean by merely making a self-consistent receipt hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_advanced_evaluation_verification import (
    verify_authoritative_advanced_evaluation_evidence,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_artifacts import AdvancedArtifactManifest, MetricQualificationPolicy
from training.advanced_rag_promotion_evidence import (
    AdvancedPromotionEvidence,
    build_advanced_promotion_evidence,
)

_MAX_BYTES = 32 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("authoritative promotion evidence destination must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _promotion_payload(value: AdvancedPromotionEvidence) -> Mapping[str, Any]:
    if not isinstance(value, AdvancedPromotionEvidence):
        raise ValueError("promotion must be AdvancedPromotionEvidence")
    return {
        "artifact_sha256": value.artifact_sha256,
        "policy_sha256": value.policy_sha256,
        "evaluation_receipt_sha256": value.evaluation_receipt_sha256,
        "metrics_sha256": value.metrics_sha256,
        "promoted": value.promoted,
        "reason_codes": list(value.reason_codes),
        "primitive_receipt_sha256": value.primitive_receipt_sha256,
        "evidence_sha256": value.evidence_sha256,
    }


def _promotion_from_payload(raw: Mapping[str, Any]) -> AdvancedPromotionEvidence:
    required = {"artifact_sha256", "policy_sha256", "evaluation_receipt_sha256", "metrics_sha256", "promoted", "reason_codes", "primitive_receipt_sha256", "evidence_sha256"}
    if set(raw) != required or not isinstance(raw.get("reason_codes"), list):
        raise ValueError("nested advanced promotion evidence fields are invalid")
    return AdvancedPromotionEvidence(
        artifact_sha256=raw["artifact_sha256"], policy_sha256=raw["policy_sha256"],
        evaluation_receipt_sha256=raw["evaluation_receipt_sha256"], metrics_sha256=raw["metrics_sha256"],
        promoted=raw["promoted"], reason_codes=tuple(raw["reason_codes"]),
        primitive_receipt_sha256=raw["primitive_receipt_sha256"], evidence_sha256=raw["evidence_sha256"],
    )


def _policy(minimum: Mapping[str, Any], maximum: Mapping[str, Any]) -> MetricQualificationPolicy:
    if not isinstance(minimum, Mapping) or not isinstance(maximum, Mapping):
        raise ValueError("authoritative promotion policy minimum/maximum must be mappings")
    return MetricQualificationPolicy(minimum=dict(minimum), maximum=dict(maximum))


@dataclass(frozen=True)
class AuthoritativeAdvancedPromotionEvidence:
    artifact_sha256: str
    authoritative_evaluation_evidence_path: str
    authoritative_evaluation_evidence_file_sha256: str
    authoritative_evaluation_evidence_sha256: str
    evaluation_receipt_sha256: str
    policy_minimum: Mapping[str, float]
    policy_maximum: Mapping[str, float]
    promotion: AdvancedPromotionEvidence
    evidence_sha256: str

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "authoritative_evaluation_evidence_file_sha256", "authoritative_evaluation_evidence_sha256", "evaluation_receipt_sha256", "evidence_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        path = safe_advanced_path(self.authoritative_evaluation_evidence_path, label="authoritative advanced evaluation evidence", must_exist=True, require_file=True)
        object.__setattr__(self, "authoritative_evaluation_evidence_path", str(path))
        policy = _policy(self.policy_minimum, self.policy_maximum)
        object.__setattr__(self, "policy_minimum", dict(policy.minimum))
        object.__setattr__(self, "policy_maximum", dict(policy.maximum))
        if not isinstance(self.promotion, AdvancedPromotionEvidence):
            raise ValueError("promotion must be AdvancedPromotionEvidence")
        if self.promotion.artifact_sha256 != self.artifact_sha256:
            raise ValueError("nested promotion is bound to a different artifact")
        if self.promotion.evaluation_receipt_sha256 != self.evaluation_receipt_sha256:
            raise ValueError("nested promotion is bound to a different evaluation receipt")
        if self.promotion.policy_sha256 != policy.policy_sha256:
            raise ValueError("nested promotion policy digest differs from embedded policy values")
        if _digest(self.unsigned()) != self.evidence_sha256:
            raise ValueError("authoritative advanced promotion evidence digest mismatch")

    @property
    def promoted(self) -> bool:
        return self.promotion.promoted

    @property
    def policy_sha256(self) -> str:
        return self.promotion.policy_sha256

    @property
    def metrics_sha256(self) -> str:
        return self.promotion.metrics_sha256

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.promotion.reason_codes

    @property
    def policy(self) -> MetricQualificationPolicy:
        return _policy(self.policy_minimum, self.policy_maximum)

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-advanced-promotion-evidence/v2",
            "artifact_sha256": self.artifact_sha256,
            "authoritative_evaluation_evidence_path": self.authoritative_evaluation_evidence_path,
            "authoritative_evaluation_evidence_file_sha256": self.authoritative_evaluation_evidence_file_sha256,
            "authoritative_evaluation_evidence_sha256": self.authoritative_evaluation_evidence_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "policy_minimum": dict(self.policy_minimum),
            "policy_maximum": dict(self.policy_maximum),
            "promotion": _promotion_payload(self.promotion),
        }


def _assert_manifest_lineage(manifest: AdvancedArtifactManifest, *, evaluation: Any) -> None:
    if not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("manifest must be AdvancedArtifactManifest")
    expected_kind = "grounded_generation" if manifest.kind == "grounded_generator" else "dynamic_rag_policy" if manifest.kind == "dynamic_rag_policy" else None
    if expected_kind is None:
        raise ValueError("unsupported advanced artifact kind")
    checks = {
        "kind": evaluation.kind == expected_kind,
        "checkpoint_digest": evaluation.checkpoint_digest == manifest.checkpoint_digest,
        "plan_sha256": evaluation.plan_sha256 == manifest.plan_sha256,
        "training_input_sha256": evaluation.training_input_sha256 == manifest.training_input_sha256,
        "training_config_sha256": evaluation.training_config_sha256 == manifest.training_config_sha256,
        "source_commit": evaluation.source_commit == manifest.source_commit,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError("authoritative evaluation differs from artifact lineage: " + ",".join(failures))
    if manifest.evaluation_receipt_sha256 is not None and manifest.evaluation_receipt_sha256 != evaluation.receipt_sha256:
        raise ValueError("artifact is bound to a different evaluation receipt")


def build_authoritative_advanced_promotion_evidence(
    manifest: AdvancedArtifactManifest,
    *,
    authoritative_evaluation_evidence_path: str | Path,
    policy: MetricQualificationPolicy,
) -> AuthoritativeAdvancedPromotionEvidence:
    if not isinstance(policy, MetricQualificationPolicy):
        raise ValueError("policy must be MetricQualificationPolicy")
    evidence_path = safe_advanced_path(authoritative_evaluation_evidence_path, label="authoritative advanced evaluation evidence", must_exist=True, require_file=True)
    evaluation, evaluation_evidence = verify_authoritative_advanced_evaluation_evidence(evidence_path)
    _assert_manifest_lineage(manifest, evaluation=evaluation)
    promotion = build_advanced_promotion_evidence(manifest, evaluation, policy)
    unsigned = {
        "schema": "rigorousrag-authoritative-advanced-promotion-evidence/v2",
        "artifact_sha256": manifest.artifact_sha256,
        "authoritative_evaluation_evidence_path": str(evidence_path),
        "authoritative_evaluation_evidence_file_sha256": _file_sha(evidence_path),
        "authoritative_evaluation_evidence_sha256": evaluation_evidence.evidence_sha256,
        "evaluation_receipt_sha256": evaluation.receipt_sha256,
        "policy_minimum": dict(policy.minimum),
        "policy_maximum": dict(policy.maximum),
        "promotion": _promotion_payload(promotion),
    }
    return AuthoritativeAdvancedPromotionEvidence(
        artifact_sha256=manifest.artifact_sha256,
        authoritative_evaluation_evidence_path=str(evidence_path),
        authoritative_evaluation_evidence_file_sha256=unsigned["authoritative_evaluation_evidence_file_sha256"],
        authoritative_evaluation_evidence_sha256=evaluation_evidence.evidence_sha256,
        evaluation_receipt_sha256=evaluation.receipt_sha256,
        policy_minimum=dict(policy.minimum),
        policy_maximum=dict(policy.maximum),
        promotion=promotion,
        evidence_sha256=_digest(unsigned),
    )


def assert_authoritative_advanced_promotion(manifest: AdvancedArtifactManifest, evidence: AuthoritativeAdvancedPromotionEvidence) -> None:
    if not isinstance(evidence, AuthoritativeAdvancedPromotionEvidence):
        raise ValueError("evidence must be AuthoritativeAdvancedPromotionEvidence")
    path = safe_advanced_path(evidence.authoritative_evaluation_evidence_path, label="authoritative advanced evaluation evidence", must_exist=True, require_file=True)
    if _file_sha(path) != evidence.authoritative_evaluation_evidence_file_sha256:
        raise ValueError("authoritative evaluation evidence bytes changed after promotion")
    evaluation, evaluation_evidence = verify_authoritative_advanced_evaluation_evidence(path)
    if evaluation_evidence.evidence_sha256 != evidence.authoritative_evaluation_evidence_sha256:
        raise ValueError("authoritative evaluation evidence identity changed after promotion")
    if evaluation.receipt_sha256 != evidence.evaluation_receipt_sha256:
        raise ValueError("advanced evaluation receipt changed after promotion")
    _assert_manifest_lineage(manifest, evaluation=evaluation)
    if evidence.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("authoritative promotion is bound to a different artifact")
    recomputed = build_advanced_promotion_evidence(manifest, evaluation, evidence.policy)
    if recomputed.evidence_sha256 != evidence.promotion.evidence_sha256:
        raise ValueError("promotion decision differs from independent policy recomputation")
    if recomputed.promoted != evidence.promotion.promoted or recomputed.reason_codes != evidence.promotion.reason_codes:
        raise ValueError("promotion status/reasons differ from independent policy recomputation")


def write_authoritative_advanced_promotion_evidence(path: str | Path, evidence: AuthoritativeAdvancedPromotionEvidence) -> None:
    if not isinstance(evidence, AuthoritativeAdvancedPromotionEvidence):
        raise ValueError("evidence must be AuthoritativeAdvancedPromotionEvidence")
    destination = safe_advanced_path(path, label="authoritative advanced promotion evidence", must_exist=False)
    if destination.exists():
        raise ValueError("authoritative promotion evidence destination must not already exist")
    _atomic(destination, _canonical({**evidence.unsigned(), "evidence_sha256": evidence.evidence_sha256}) + b"\n")


def read_authoritative_advanced_promotion_evidence(path: str | Path) -> AuthoritativeAdvancedPromotionEvidence:
    source = safe_advanced_path(path, label="authoritative advanced promotion evidence", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("authoritative promotion evidence exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("authoritative promotion evidence is not strict JSON") from exc
    required = {"schema", "artifact_sha256", "authoritative_evaluation_evidence_path", "authoritative_evaluation_evidence_file_sha256", "authoritative_evaluation_evidence_sha256", "evaluation_receipt_sha256", "policy_minimum", "policy_maximum", "promotion", "evidence_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-advanced-promotion-evidence/v2" or not isinstance(raw.get("promotion"), Mapping) or not isinstance(raw.get("policy_minimum"), Mapping) or not isinstance(raw.get("policy_maximum"), Mapping):
        raise ValueError("unsupported authoritative advanced promotion evidence schema")
    return AuthoritativeAdvancedPromotionEvidence(
        artifact_sha256=raw["artifact_sha256"],
        authoritative_evaluation_evidence_path=raw["authoritative_evaluation_evidence_path"],
        authoritative_evaluation_evidence_file_sha256=raw["authoritative_evaluation_evidence_file_sha256"],
        authoritative_evaluation_evidence_sha256=raw["authoritative_evaluation_evidence_sha256"],
        evaluation_receipt_sha256=raw["evaluation_receipt_sha256"],
        policy_minimum=raw["policy_minimum"],
        policy_maximum=raw["policy_maximum"],
        promotion=_promotion_from_payload(raw["promotion"]),
        evidence_sha256=raw["evidence_sha256"],
    )


__all__ = [
    "AuthoritativeAdvancedPromotionEvidence",
    "assert_authoritative_advanced_promotion",
    "build_authoritative_advanced_promotion_evidence",
    "read_authoritative_advanced_promotion_evidence",
    "write_authoritative_advanced_promotion_evidence",
]
