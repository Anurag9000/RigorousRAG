from __future__ import annotations

import hashlib

import pytest

from models.admitted_local_adapters import build_admitted_splade_provider
from models.admitted_local_artifacts import AdmittedArtifactProof, AdmittedLocalArtifactBinding
from models.local_hf_adapters import LocalArtifactBinding, artifact_tree_digest
from security.artifact_attestation import (
    ArtifactAdmissionPolicy,
    ArtifactAttestationStatement,
    ArtifactSubject,
    AttestationPredicate,
    VerifiedAttestation,
    decide_artifact_admission,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _proof(artifact_type: str, artifact_sha256: str, artifact_id: str) -> AdmittedArtifactProof:
    subject = ArtifactSubject(artifact_id, artifact_type, artifact_sha256)
    statement = ArtifactAttestationStatement.build(
        subject=subject,
        builder_id="trusted-builder",
        source_revision="a" * 40,
        build_config_sha256=_sha("build-config"),
        dependency_lock_sha256=_sha("lock"),
        sbom_sha256=_sha("sbom"),
        predicates=(
            AttestationPredicate("build_provenance", _sha("build-provenance")),
            AttestationPredicate("sbom", _sha("sbom-predicate")),
            AttestationPredicate("dependency_lock", _sha("lock-predicate")),
        ),
        produced_at=10.0,
    )
    verification = VerifiedAttestation(
        statement.statement_sha256,
        subject.subject_sha256,
        "trusted-key",
        "trusted-verifier",
        _sha("verifier-version"),
        _sha(f"verification:{artifact_id}"),
        11.0,
    )
    policy = ArtifactAdmissionPolicy(
        "local-model-admission",
        ("build_provenance", "sbom", "dependency_lock"),
        ("trusted-builder",),
        ("trusted-key",),
        ("trusted-verifier",),
        maximum_attestation_age_seconds=1000.0,
    )
    decision = decide_artifact_admission(
        statement,
        verification,
        policy=policy,
        now=12.0,
        expected_artifact_sha256=artifact_sha256,
        expected_source_revision="a" * 40,
        expected_dependency_lock_sha256=_sha("lock"),
    )
    assert decision.admitted
    return AdmittedArtifactProof(statement, decision)


def _binding(tmp_path):
    model_root = tmp_path / "model"
    tokenizer_root = tmp_path / "tokenizer"
    model_root.mkdir()
    tokenizer_root.mkdir()
    (model_root / "config.json").write_text('{"model":"example"}', encoding="utf-8")
    (tokenizer_root / "tokenizer.json").write_text('{"tokenizer":"example"}', encoding="utf-8")
    model_sha = artifact_tree_digest(model_root)
    tokenizer_sha = artifact_tree_digest(tokenizer_root)
    binding = LocalArtifactBinding(str(model_root), model_sha, str(tokenizer_root), tokenizer_sha, "revision-1")
    admitted = AdmittedLocalArtifactBinding.build(
        binding,
        model_proof=_proof("model", model_sha, "model-tree"),
        tokenizer_proof=_proof("tokenizer", tokenizer_sha, "tokenizer-tree"),
    )
    return binding, admitted


def test_admitted_binding_rejects_subject_digest_mismatch(tmp_path) -> None:
    binding, admitted = _binding(tmp_path)
    wrong_model = _proof("model", _sha("not-the-tree"), "wrong-model")
    with pytest.raises(ValueError, match="admitted model digest differs"):
        AdmittedLocalArtifactBinding.build(
            binding,
            model_proof=wrong_model,
            tokenizer_proof=admitted.tokenizer_proof,
        )


def test_admitted_binding_detects_local_tree_tampering(tmp_path) -> None:
    _, admitted = _binding(tmp_path)
    (tmp_path / "model" / "config.json").write_text('{"model":"tampered"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        admitted.verify()


def test_admitted_splade_factory_requires_runtime_artifact_digest_to_match_admission(tmp_path) -> None:
    binding, admitted = _binding(tmp_path)
    provider = build_admitted_splade_provider(admitted, artifact_digest=binding.model_tree_sha256)
    assert provider.artifact_digest == binding.model_tree_sha256
    with pytest.raises(ValueError, match="must equal the admitted model tree digest"):
        build_admitted_splade_provider(admitted, artifact_digest=_sha("different"))


def test_admitted_factory_is_lazy_and_does_not_load_model_on_construction(tmp_path) -> None:
    binding, admitted = _binding(tmp_path)
    provider = build_admitted_splade_provider(admitted, artifact_digest=binding.model_tree_sha256)
    assert provider._model is None
    assert provider._tokenizer is None
