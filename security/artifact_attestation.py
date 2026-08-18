"""Verifier-neutral software/model artifact attestation and admission policy.

Cryptographic signature verification is injected through a trusted verifier adapter; this
module does not invent signing cryptography. It canonicalizes attestation statements,
binds verified evidence to exact artifact/source/SBOM/lock identities, and makes admission
fail closed when required predicates, builders, keys, source revisions, or freshness are
missing. The same contract can govern containers, local model trees and generated indexes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

_PREDICATE_TYPES = frozenset(
    {
        "build_provenance",
        "sbom",
        "dependency_lock",
        "license_review",
        "vulnerability_scan",
        "model_card",
        "dataset_card",
        "reproducibility_manifest",
    }
)
_ARTIFACT_TYPES = frozenset({"container", "python_package", "model", "tokenizer", "index", "dataset", "binary", "configuration"})
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 1000) -> str:
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


def _predicate_type(value: Any) -> str:
    selected = _text(value, "predicate_type", 100).lower()
    if selected not in _PREDICATE_TYPES:
        raise ValueError(f"unsupported attestation predicate_type {selected!r}")
    return selected


def _artifact_type(value: Any) -> str:
    selected = _text(value, "artifact_type", 100).lower()
    if selected not in _ARTIFACT_TYPES:
        raise ValueError(f"unsupported artifact_type {selected!r}")
    return selected


@dataclass(frozen=True)
class ArtifactSubject:
    artifact_id: str
    artifact_type: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 500))
        object.__setattr__(self, "artifact_type", _artifact_type(self.artifact_type))
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, "artifact_sha256"))

    @property
    def subject_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-artifact-subject/v1", **asdict(self)})


@dataclass(frozen=True)
class AttestationPredicate:
    predicate_type: str
    predicate_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_type", _predicate_type(self.predicate_type))
        object.__setattr__(self, "predicate_sha256", _sha(self.predicate_sha256, "predicate_sha256"))


@dataclass(frozen=True)
class ArtifactAttestationStatement:
    subject: ArtifactSubject
    builder_id: str
    source_revision: str
    build_config_sha256: str
    dependency_lock_sha256: str
    sbom_sha256: str
    predicates: tuple[AttestationPredicate, ...]
    produced_at: float
    statement_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ArtifactSubject):
            raise ValueError("subject must be ArtifactSubject")
        object.__setattr__(self, "builder_id", _text(self.builder_id, "builder_id", 500))
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        for name in ("build_config_sha256", "dependency_lock_sha256", "sbom_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        predicates = tuple(self.predicates)
        if any(not isinstance(value, AttestationPredicate) for value in predicates):
            raise ValueError("predicates contains invalid values")
        if len({value.predicate_type for value in predicates}) != len(predicates):
            raise ValueError("attestation may contain at most one predicate per type")
        object.__setattr__(self, "predicates", tuple(sorted(predicates, key=lambda value: value.predicate_type)))
        object.__setattr__(self, "produced_at", _time(self.produced_at, "produced_at"))
        expected = _digest(self._payload())
        provided = _sha(self.statement_sha256, "statement_sha256")
        if expected != provided:
            raise ValueError("statement_sha256 does not match attestation statement")
        object.__setattr__(self, "statement_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-artifact-attestation-statement/v1",
            "subject": asdict(self.subject),
            "builder_id": self.builder_id,
            "source_revision": self.source_revision,
            "build_config_sha256": self.build_config_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "sbom_sha256": self.sbom_sha256,
            "predicates": [asdict(value) for value in self.predicates],
            "produced_at": self.produced_at,
        }

    @classmethod
    def build(
        cls,
        *,
        subject: ArtifactSubject,
        builder_id: str,
        source_revision: str,
        build_config_sha256: str,
        dependency_lock_sha256: str,
        sbom_sha256: str,
        predicates: Sequence[AttestationPredicate],
        produced_at: float,
    ) -> "ArtifactAttestationStatement":
        selected_predicates = tuple(sorted(tuple(predicates), key=lambda value: value.predicate_type))
        payload = {
            "schema": "rigorousrag-artifact-attestation-statement/v1",
            "subject": asdict(subject),
            "builder_id": _text(builder_id, "builder_id", 500),
            "source_revision": _git_revision(source_revision),
            "build_config_sha256": _sha(build_config_sha256, "build_config_sha256"),
            "dependency_lock_sha256": _sha(dependency_lock_sha256, "dependency_lock_sha256"),
            "sbom_sha256": _sha(sbom_sha256, "sbom_sha256"),
            "predicates": [asdict(value) for value in selected_predicates],
            "produced_at": _time(produced_at, "produced_at"),
        }
        return cls(
            subject,
            payload["builder_id"],
            payload["source_revision"],
            payload["build_config_sha256"],
            payload["dependency_lock_sha256"],
            payload["sbom_sha256"],
            selected_predicates,
            payload["produced_at"],
            _digest(payload),
        )


@dataclass(frozen=True)
class SignedAttestationEnvelope:
    statement: ArtifactAttestationStatement
    key_id: str
    signature: bytes
    signature_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.statement, ArtifactAttestationStatement):
            raise ValueError("statement must be ArtifactAttestationStatement")
        object.__setattr__(self, "key_id", _text(self.key_id, "key_id", 500))
        if not isinstance(self.signature, bytes) or not self.signature or len(self.signature) > 1_000_000:
            raise ValueError("signature must be bounded non-empty bytes")
        object.__setattr__(self, "signature_format", _text(self.signature_format, "signature_format", 100).lower())


@dataclass(frozen=True)
class VerifiedAttestation:
    statement_sha256: str
    subject_sha256: str
    key_id: str
    verifier_id: str
    verifier_version_sha256: str
    verification_evidence_sha256: str
    verified_at: float

    def __post_init__(self) -> None:
        for name in ("statement_sha256", "subject_sha256", "verifier_version_sha256", "verification_evidence_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "key_id", _text(self.key_id, "key_id", 500))
        object.__setattr__(self, "verifier_id", _text(self.verifier_id, "verifier_id", 500))
        object.__setattr__(self, "verified_at", _time(self.verified_at, "verified_at"))

    @property
    def verification_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-verified-attestation/v1", **asdict(self)})


class AttestationVerifier(Protocol):
    def verify(self, envelope: SignedAttestationEnvelope, *, now: float) -> VerifiedAttestation: ...


def verify_attestation(
    envelope: SignedAttestationEnvelope,
    verifier: AttestationVerifier,
    *,
    now: float,
) -> VerifiedAttestation:
    if not isinstance(envelope, SignedAttestationEnvelope):
        raise ValueError("envelope must be SignedAttestationEnvelope")
    result = verifier.verify(envelope, now=_time(now, "now"))
    if not isinstance(result, VerifiedAttestation):
        raise RuntimeError("attestation verifier returned an invalid result")
    if result.statement_sha256 != envelope.statement.statement_sha256:
        raise RuntimeError("attestation verifier changed statement identity")
    if result.subject_sha256 != envelope.statement.subject.subject_sha256:
        raise RuntimeError("attestation verifier changed subject identity")
    if result.key_id != envelope.key_id:
        raise RuntimeError("attestation verifier changed signing key identity")
    return result


@dataclass(frozen=True)
class ArtifactAdmissionPolicy:
    policy_id: str
    required_predicate_types: tuple[str, ...]
    trusted_builder_ids: tuple[str, ...]
    trusted_key_ids: tuple[str, ...]
    trusted_verifier_ids: tuple[str, ...]
    maximum_attestation_age_seconds: float = 30 * 24 * 60 * 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        predicates = tuple(sorted({_predicate_type(value) for value in self.required_predicate_types}))
        builders = tuple(sorted({_text(value, "trusted builder id", 500) for value in self.trusted_builder_ids}))
        keys = tuple(sorted({_text(value, "trusted key id", 500) for value in self.trusted_key_ids}))
        verifiers = tuple(sorted({_text(value, "trusted verifier id", 500) for value in self.trusted_verifier_ids}))
        if not predicates or not builders or not keys or not verifiers:
            raise ValueError("admission policy trust/evidence sets must be non-empty")
        object.__setattr__(self, "required_predicate_types", predicates)
        object.__setattr__(self, "trusted_builder_ids", builders)
        object.__setattr__(self, "trusted_key_ids", keys)
        object.__setattr__(self, "trusted_verifier_ids", verifiers)
        age = _time(self.maximum_attestation_age_seconds, "maximum_attestation_age_seconds")
        if age <= 0.0:
            raise ValueError("maximum_attestation_age_seconds must be positive")
        object.__setattr__(self, "maximum_attestation_age_seconds", age)

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-artifact-admission-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class ArtifactAdmissionDecision:
    subject_sha256: str
    statement_sha256: str
    verification_sha256: str
    policy_sha256: str
    admitted: bool
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        for name in ("subject_sha256", "statement_sha256", "verification_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.admitted, bool):
            raise ValueError("admitted must be boolean")
        reasons = tuple(sorted({_text(value, "reason code", 200) for value in self.reason_codes}))
        if self.admitted and reasons:
            raise ValueError("admitted decision may not contain failure reasons")
        if not self.admitted and not reasons:
            raise ValueError("blocked admission decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match artifact admission decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-artifact-admission-decision/v1",
            "subject_sha256": self.subject_sha256,
            "statement_sha256": self.statement_sha256,
            "verification_sha256": self.verification_sha256,
            "policy_sha256": self.policy_sha256,
            "admitted": self.admitted,
            "reason_codes": self.reason_codes,
        }


def decide_artifact_admission(
    statement: ArtifactAttestationStatement,
    verification: VerifiedAttestation,
    *,
    policy: ArtifactAdmissionPolicy,
    now: float,
    expected_artifact_sha256: str,
    expected_source_revision: str,
    expected_dependency_lock_sha256: str,
) -> ArtifactAdmissionDecision:
    if not isinstance(statement, ArtifactAttestationStatement) or not isinstance(verification, VerifiedAttestation):
        raise ValueError("statement/verification types are invalid")
    if not isinstance(policy, ArtifactAdmissionPolicy):
        raise ValueError("policy must be ArtifactAdmissionPolicy")
    instant = _time(now, "now")
    if verification.statement_sha256 != statement.statement_sha256 or verification.subject_sha256 != statement.subject.subject_sha256:
        raise ValueError("verification is bound to a different attestation statement/subject")
    reasons: list[str] = []
    if statement.subject.artifact_sha256 != _sha(expected_artifact_sha256, "expected_artifact_sha256"):
        reasons.append("artifact_digest_mismatch")
    if statement.source_revision != _git_revision(expected_source_revision):
        reasons.append("source_revision_mismatch")
    if statement.dependency_lock_sha256 != _sha(expected_dependency_lock_sha256, "expected_dependency_lock_sha256"):
        reasons.append("dependency_lock_mismatch")
    if statement.builder_id not in policy.trusted_builder_ids:
        reasons.append("builder_not_trusted")
    if verification.key_id not in policy.trusted_key_ids:
        reasons.append("signing_key_not_trusted")
    if verification.verifier_id not in policy.trusted_verifier_ids:
        reasons.append("verifier_not_trusted")
    if verification.verified_at > instant:
        reasons.append("verification_is_future_dated")
    if statement.produced_at > instant or instant - statement.produced_at > policy.maximum_attestation_age_seconds:
        reasons.append("attestation_is_stale_or_future_dated")
    present_predicates = {value.predicate_type for value in statement.predicates}
    for predicate in policy.required_predicate_types:
        if predicate not in present_predicates:
            reasons.append(f"missing_required_predicate:{predicate}")
    payload = {
        "schema": "rigorousrag-artifact-admission-decision/v1",
        "subject_sha256": statement.subject.subject_sha256,
        "statement_sha256": statement.statement_sha256,
        "verification_sha256": verification.verification_sha256,
        "policy_sha256": policy.policy_sha256,
        "admitted": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return ArtifactAdmissionDecision(**payload, decision_sha256=_digest(payload))


__all__ = [
    "ArtifactAdmissionDecision",
    "ArtifactAdmissionPolicy",
    "ArtifactAttestationStatement",
    "ArtifactSubject",
    "AttestationPredicate",
    "AttestationVerifier",
    "SignedAttestationEnvelope",
    "VerifiedAttestation",
    "decide_artifact_admission",
    "verify_attestation",
]
