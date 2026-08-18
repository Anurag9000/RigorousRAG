"""Bind verified local HF artifact trees to governed supply-chain admission decisions.

``models.local_hf_adapters.LocalArtifactBinding`` proves that local bytes match declared
model/tokenizer tree digests. ``security.artifact_attestation`` proves that an exact
artifact digest was admitted under a trusted builder/key/verifier policy. This module
requires both proofs before an adapter may be constructed, preventing digest-correct but
unadmitted local artifacts from silently entering serving code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from models.local_hf_adapters import LocalArtifactBinding, artifact_tree_digest
from security.artifact_attestation import ArtifactAdmissionDecision, ArtifactAttestationStatement

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class AdmittedArtifactProof:
    statement: ArtifactAttestationStatement
    decision: ArtifactAdmissionDecision

    def __post_init__(self) -> None:
        if not isinstance(self.statement, ArtifactAttestationStatement):
            raise ValueError("statement must be ArtifactAttestationStatement")
        if not isinstance(self.decision, ArtifactAdmissionDecision):
            raise ValueError("decision must be ArtifactAdmissionDecision")
        if not self.decision.admitted:
            raise ValueError("artifact admission decision is not admitted")
        if self.decision.statement_sha256 != self.statement.statement_sha256:
            raise ValueError("artifact admission decision is bound to a different statement")
        if self.decision.subject_sha256 != self.statement.subject.subject_sha256:
            raise ValueError("artifact admission decision is bound to a different subject")

    @property
    def proof_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-admitted-artifact-proof/v1",
                "statement_sha256": self.statement.statement_sha256,
                "admission_decision_sha256": self.decision.decision_sha256,
            }
        )

    @property
    def artifact_sha256(self) -> str:
        return self.statement.subject.artifact_sha256


@dataclass(frozen=True)
class AdmittedLocalArtifactBinding:
    binding: LocalArtifactBinding
    model_proof: AdmittedArtifactProof
    tokenizer_proof: AdmittedArtifactProof
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, LocalArtifactBinding):
            raise ValueError("binding must be LocalArtifactBinding")
        if not isinstance(self.model_proof, AdmittedArtifactProof) or not isinstance(self.tokenizer_proof, AdmittedArtifactProof):
            raise ValueError("model/tokenizer proofs must be AdmittedArtifactProof values")
        if self.model_proof.statement.subject.artifact_type != "model":
            raise ValueError("model_proof subject must have artifact_type=model")
        if self.tokenizer_proof.statement.subject.artifact_type != "tokenizer":
            raise ValueError("tokenizer_proof subject must have artifact_type=tokenizer")
        if self.model_proof.artifact_sha256 != self.binding.model_tree_sha256:
            raise ValueError("admitted model digest differs from local model binding")
        if self.tokenizer_proof.artifact_sha256 != self.binding.tokenizer_tree_sha256:
            raise ValueError("admitted tokenizer digest differs from local tokenizer binding")
        expected = _digest(self._payload())
        provided = _sha(self.binding_sha256, "binding_sha256")
        if provided != expected:
            raise ValueError("binding_sha256 does not match admitted local artifact binding")
        object.__setattr__(self, "binding_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-admitted-local-artifact-binding/v1",
            "model_tree_sha256": self.binding.model_tree_sha256,
            "tokenizer_tree_sha256": self.binding.tokenizer_tree_sha256,
            "declared_revision": self.binding.declared_revision,
            "model_proof_sha256": self.model_proof.proof_sha256,
            "tokenizer_proof_sha256": self.tokenizer_proof.proof_sha256,
        }

    @classmethod
    def build(
        cls,
        binding: LocalArtifactBinding,
        *,
        model_proof: AdmittedArtifactProof,
        tokenizer_proof: AdmittedArtifactProof,
    ) -> "AdmittedLocalArtifactBinding":
        if not isinstance(binding, LocalArtifactBinding):
            raise ValueError("binding must be LocalArtifactBinding")
        payload = {
            "schema": "rigorousrag-admitted-local-artifact-binding/v1",
            "model_tree_sha256": binding.model_tree_sha256,
            "tokenizer_tree_sha256": binding.tokenizer_tree_sha256,
            "declared_revision": binding.declared_revision,
            "model_proof_sha256": model_proof.proof_sha256,
            "tokenizer_proof_sha256": tokenizer_proof.proof_sha256,
        }
        return cls(binding, model_proof, tokenizer_proof, _digest(payload))

    def verify(self) -> LocalArtifactBinding:
        """Re-hash both local trees and re-bind them to admitted supply-chain subjects."""

        self.binding.verify()
        if artifact_tree_digest(self.binding.model_root) != self.model_proof.artifact_sha256:
            raise RuntimeError("local model tree no longer matches admitted artifact subject")
        if artifact_tree_digest(self.binding.tokenizer_root) != self.tokenizer_proof.artifact_sha256:
            raise RuntimeError("local tokenizer tree no longer matches admitted artifact subject")
        return self.binding


def require_admitted_local_binding(value: AdmittedLocalArtifactBinding) -> LocalArtifactBinding:
    if not isinstance(value, AdmittedLocalArtifactBinding):
        raise ValueError("value must be AdmittedLocalArtifactBinding")
    return value.verify()


__all__ = ["AdmittedArtifactProof", "AdmittedLocalArtifactBinding", "require_admitted_local_binding"]
