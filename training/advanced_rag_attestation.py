"""Bridge advanced-RAG promotion into the repository's signed artifact admission authority."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from security.artifact_attestation import (
    ArtifactAdmissionPolicy,
    ArtifactAttestationStatement,
    AttestationVerifier,
    SignedAttestationEnvelope,
    decide_artifact_admission,
    verify_attestation,
)
from training.advanced_rag_artifacts import AdvancedArtifactManifest, AdvancedArtifactPromotionReceipt, ArtifactAdmissionSink


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
        for name in (
            "artifact_sha256",
            "promotion_receipt_sha256",
            "attestation_statement_sha256",
            "attestation_verification_sha256",
            "admission_policy_sha256",
            "admission_decision_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.admitted, bool):
            raise ValueError("admitted must be boolean")


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
    """Verify promotion + signed supply-chain evidence and produce one immutable receipt."""
    if not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("manifest must be AdvancedArtifactManifest")
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
        statement,
        verification,
        policy=policy,
        now=now,
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
    return AdvancedArtifactAdmissionReceipt(**unsigned, receipt_sha256=_digest(unsigned))


def admit_attested_advanced_artifact(
    directory: str | Path,
    manifest: AdvancedArtifactManifest,
    promotion: AdvancedArtifactPromotionReceipt,
    admission: AdvancedArtifactAdmissionReceipt,
    sink: ArtifactAdmissionSink,
) -> Any:
    """Final fail-closed handoff after both metric promotion and signed supply-chain admission."""
    if not promotion.promoted or not admission.admitted:
        raise ValueError("artifact must pass promotion and attestation admission")
    if not (manifest.artifact_sha256 == promotion.artifact_sha256 == admission.artifact_sha256):
        raise ValueError("artifact/promotion/admission identities differ")
    if admission.promotion_receipt_sha256 != promotion.receipt_sha256:
        raise ValueError("admission receipt is bound to a different promotion receipt")
    selected = Path(directory).expanduser().resolve(strict=True)
    if not selected.is_dir() or selected.is_symlink() or selected.name != manifest.artifact_sha256:
        raise ValueError("advanced artifact directory must be the exact content-addressed export directory")
    return sink.admit(
        str(selected),
        artifact_sha256=manifest.artifact_sha256,
        promotion_receipt_sha256=admission.receipt_sha256,
    )


__all__ = [
    "AdvancedArtifactAdmissionReceipt",
    "admit_attested_advanced_artifact",
    "verify_advanced_artifact_attestation",
]
