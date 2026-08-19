"""Bridge advanced-RAG promotion into the repository's signed artifact admission authority."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from security.artifact_attestation import ArtifactAdmissionPolicy, ArtifactAttestationStatement, AttestationVerifier, SignedAttestationEnvelope, decide_artifact_admission, verify_attestation
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_artifacts import AdvancedArtifactManifest, AdvancedArtifactPromotionReceipt, ArtifactAdmissionSink
from training.advanced_rag_manifest_integrity import assert_advanced_manifest_self_consistent
from training.advanced_rag_promotion_evidence import AdvancedPromotionEvidence


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
class AdvancedArtifactAdmissionReceipt:
    artifact_sha256: str
    promotion_receipt_sha256: str
    attestation_statement_sha256: str
    attestation_verification_sha256: str
    admission_policy_sha256: str
    admission_decision_sha256: str
    admitted: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "promotion_receipt_sha256", "attestation_statement_sha256", "attestation_verification_sha256", "admission_policy_sha256", "admission_decision_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.admitted, bool):
            raise ValueError("admitted must be boolean")
        if _digest(self._payload()) != self.receipt_sha256:
            raise ValueError("advanced artifact admission receipt digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-advanced-artifact-admission-receipt/v1",
            "artifact_sha256": self.artifact_sha256,
            "promotion_receipt_sha256": self.promotion_receipt_sha256,
            "attestation_statement_sha256": self.attestation_statement_sha256,
            "attestation_verification_sha256": self.attestation_verification_sha256,
            "admission_policy_sha256": self.admission_policy_sha256,
            "admission_decision_sha256": self.admission_decision_sha256,
            "admitted": self.admitted,
        }


def verify_advanced_artifact_attestation(
    manifest: AdvancedArtifactManifest,
    promotion: AdvancedArtifactPromotionReceipt,
    envelope: SignedAttestationEnvelope,
    verifier: AttestationVerifier,
    *,
    policy: ArtifactAdmissionPolicy,
    now: float,
    expected_dependency_lock_sha256: str,
) -> AdvancedArtifactAdmissionReceipt:
    """Lower-level bridge after the caller has already verified promotion evidence."""
    assert_advanced_manifest_self_consistent(manifest)
    if not isinstance(promotion, AdvancedArtifactPromotionReceipt):
        raise ValueError("promotion must be AdvancedArtifactPromotionReceipt")
    if not promotion.promoted or promotion.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("advanced artifact must pass matching metric promotion before attestation admission")
    if not isinstance(envelope, SignedAttestationEnvelope):
        raise ValueError("envelope must be SignedAttestationEnvelope")
    statement: ArtifactAttestationStatement = envelope.statement
    if statement.subject.artifact_type != "model":
        raise ValueError("advanced RAG inference artifact attestation must describe artifact_type=model")
    if statement.subject.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("attestation subject does not match promoted advanced artifact")
    verification = verify_attestation(envelope, verifier, now=now)
    decision = decide_artifact_admission(
        statement, verification, policy=policy, now=now,
        expected_artifact_sha256=manifest.artifact_sha256,
        expected_source_revision=manifest.source_commit,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
    )
    unsigned = {
        "schema": "rigorousrag-advanced-artifact-admission-receipt/v1",
        "artifact_sha256": manifest.artifact_sha256,
        "promotion_receipt_sha256": promotion.receipt_sha256,
        "attestation_statement_sha256": statement.statement_sha256,
        "attestation_verification_sha256": verification.verification_sha256,
        "admission_policy_sha256": policy.policy_sha256,
        "admission_decision_sha256": decision.decision_sha256,
        "admitted": decision.admitted,
    }
    return AdvancedArtifactAdmissionReceipt(**{key: value for key, value in unsigned.items() if key != "schema"}, receipt_sha256=_digest(unsigned))


def verify_advanced_artifact_attestation_with_evidence(
    manifest: AdvancedArtifactManifest,
    promotion_evidence: AdvancedPromotionEvidence,
    envelope: SignedAttestationEnvelope,
    verifier: AttestationVerifier,
    *,
    policy: ArtifactAdmissionPolicy,
    now: float,
    expected_dependency_lock_sha256: str,
) -> AdvancedArtifactAdmissionReceipt:
    """Authoritative bridge from self-verifying metric evidence to signed admission."""
    assert_advanced_manifest_self_consistent(manifest)
    if not isinstance(promotion_evidence, AdvancedPromotionEvidence):
        raise ValueError("promotion_evidence must be AdvancedPromotionEvidence")
    if not promotion_evidence.promoted or promotion_evidence.artifact_sha256 != manifest.artifact_sha256:
        raise ValueError("promotion evidence does not authorize this artifact")
    return verify_advanced_artifact_attestation(
        manifest,
        promotion_evidence.primitive_receipt(),
        envelope,
        verifier,
        policy=policy,
        now=now,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
    )


def admit_attested_advanced_artifact(
    directory: str | Path,
    manifest: AdvancedArtifactManifest,
    promotion: AdvancedArtifactPromotionReceipt,
    admission: AdvancedArtifactAdmissionReceipt,
    sink: ArtifactAdmissionSink,
) -> Any:
    """Final fail-closed handoff after metric promotion and signed supply-chain admission."""
    assert_advanced_manifest_self_consistent(manifest)
    if not promotion.promoted or not admission.admitted:
        raise ValueError("artifact must pass promotion and attestation admission")
    if not (manifest.artifact_sha256 == promotion.artifact_sha256 == admission.artifact_sha256):
        raise ValueError("artifact/promotion/admission identities differ")
    if admission.promotion_receipt_sha256 != promotion.receipt_sha256:
        raise ValueError("admission receipt is bound to a different promotion receipt")
    selected = safe_advanced_path(directory, label="advanced artifact directory", must_exist=True, require_directory=True)
    if selected.name != manifest.artifact_sha256:
        raise ValueError("advanced artifact directory must be the exact content-addressed export directory")
    return sink.admit(str(selected), artifact_sha256=manifest.artifact_sha256, promotion_receipt_sha256=admission.receipt_sha256)


def admit_attested_advanced_artifact_with_evidence(
    directory: str | Path,
    manifest: AdvancedArtifactManifest,
    promotion_evidence: AdvancedPromotionEvidence,
    admission: AdvancedArtifactAdmissionReceipt,
    sink: ArtifactAdmissionSink,
) -> Any:
    if not isinstance(promotion_evidence, AdvancedPromotionEvidence):
        raise ValueError("promotion_evidence must be AdvancedPromotionEvidence")
    return admit_attested_advanced_artifact(directory, manifest, promotion_evidence.primitive_receipt(), admission, sink)


__all__ = [
    "AdvancedArtifactAdmissionReceipt",
    "admit_attested_advanced_artifact",
    "admit_attested_advanced_artifact_with_evidence",
    "verify_advanced_artifact_attestation",
    "verify_advanced_artifact_attestation_with_evidence",
]
