from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    CustodyArtifactEvidence,
    RestoreChainOfCustodyManifest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    Rfc3161TimestampVerificationReceipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    bind_rfc3161_timestamp_to_signed_custody,
    sign_governed_restore_chain_of_custody,
    verify_governed_signed_restore_chain_of_custody,
    verify_governed_timestamped_signed_restore_chain_of_custody,
    verify_signed_restore_chain_of_custody,
    verify_timestamped_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    CustodySignerKeyRegistry,
    register_custody_signer_key,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes


def actor(digit: str = "a") -> ReviewActorBinding:
    return ReviewActorBinding(
        actor_id=f"actor-{digit}",
        binding_method="process_environment",
        binding_digest=digit * 64,
        loaded_at=1.0,
    )


def keypair(tmp_path, name: str = "key"):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{name}.pem"
    public_path = tmp_path / f"{name}.pub.pem"
    private_path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_path.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    if os.name != "nt":
        private_path.chmod(0o600)
    return private_path, public_path


def manifest_file(tmp_path):
    value = RestoreChainOfCustodyManifest(
        owner_id="alice",
        chain_digest="1" * 64,
        artifacts=(CustodyArtifactEvidence(artifact_id="a" * 64),),
        generated_at=2.0,
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_bytes(value.public_payload()) + b"\n")
    return path


def registered(tmp_path):
    registry = CustodySignerKeyRegistry(tmp_path / "keys.sqlite3")
    private_path, public_path = keypair(tmp_path)
    record = register_custody_signer_key(
        registry=registry,
        owner_id="alice",
        key_id="key-1",
        public_key_path=public_path,
        actor=actor(),
        valid_from=1.0,
        now=2.0,
    )
    return registry, record, private_path, public_path


def test_signer_registry_lifecycle_collision_and_identity(tmp_path):
    registry, record, _private, public = registered(tmp_path)
    replay = register_custody_signer_key(
        registry=registry,
        owner_id="alice",
        key_id="key-1",
        public_key_path=public,
        actor=actor(),
        valid_from=1.0,
        now=99.0,
    )
    assert replay == record
    _other_private, other_public = keypair(tmp_path, "other")
    with pytest.raises(RuntimeError, match="collision"):
        register_custody_signer_key(
            registry=registry,
            owner_id="alice",
            key_id="key-1",
            public_key_path=other_public,
            actor=actor(),
            valid_from=1.0,
            now=3.0,
        )
    retired = registry.retire(owner_id="alice", key_id="key-1", actor=actor("b"), now=10.0)
    assert retired.state == "retired"
    assert retired.permits(verification_time=9.0, now=20.0)
    assert not retired.permits(verification_time=None, now=20.0)
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(registry.path.read_bytes())
    os.replace(replacement, registry.path)
    with pytest.raises(RuntimeError, match="identity changed"):
        registry.get(owner_id="alice", key_id="key-1")


def test_governed_sign_offline_verify_and_tamper_refusal(tmp_path):
    registry, record, private, public = registered(tmp_path)
    output = tmp_path / "signed.json"
    envelope, used = sign_governed_restore_chain_of_custody(
        registry=registry,
        owner_id="alice",
        key_id="key-1",
        manifest_path=manifest_file(tmp_path),
        private_key_path=private,
        output_path=output,
        now=3.0,
    )
    assert used.record_digest == record.record_digest
    assert verify_signed_restore_chain_of_custody(
        envelope_path=output,
        public_key_path=public,
        expected_key_id="key-1",
        expected_owner_id="alice",
    ).envelope_digest == envelope.envelope_digest
    receipt = verify_governed_signed_restore_chain_of_custody(
        registry=registry,
        owner_id="alice",
        signed_envelope=envelope,
        now=4.0,
    )
    assert receipt.signature_verified and receipt.key_state == "active"
    raw = json.loads(output.read_text())
    raw["manifest"]["chain_digest"] = "f" * 64
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((PermissionError, ValueError)):
        verify_signed_restore_chain_of_custody(envelope_path=output, public_key_path=public)


def test_timestamp_binding_and_retired_historical_verification(tmp_path, monkeypatch):
    registry, _record, private, public = registered(tmp_path)
    signed_path = tmp_path / "signed.json"
    envelope, _ = sign_governed_restore_chain_of_custody(
        registry=registry,
        owner_id="alice",
        key_id="key-1",
        manifest_path=manifest_file(tmp_path),
        private_key_path=private,
        output_path=signed_path,
        now=3.0,
    )
    subject = hashlib.sha256(signed_path.read_bytes()).hexdigest()
    receipt = Rfc3161TimestampVerificationReceipt(
        owner_id="alice",
        subject_sha256=subject,
        generated_at_unix=5.0,
        receipt_digest="9" * 64,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt.public_payload()), encoding="utf-8")
    timestamped_path = tmp_path / "timestamped.json"
    wrapped = bind_rfc3161_timestamp_to_signed_custody(
        signed_envelope_path=signed_path,
        receipt_path=receipt_path,
        public_key_path=public,
        output_path=timestamped_path,
        expected_key_id="key-1",
    )
    assert verify_timestamped_signed_restore_chain_of_custody(
        envelope_path=timestamped_path,
        public_key_path=public,
        expected_owner_id="alice",
    ).binding_digest == wrapped.binding_digest
    registry.retire(owner_id="alice", key_id="key-1", actor=actor("b"), now=6.0)
    with pytest.raises(PermissionError, match="validity"):
        verify_governed_signed_restore_chain_of_custody(
            registry=registry,
            owner_id="alice",
            signed_envelope=envelope,
            now=7.0,
        )
    import tools.evidence_graph_set_signed_retirement_restore_custody_signature as module
    monkeypatch.setattr(
        module,
        "verify_rfc3161_timestamp_response_with_profile",
        lambda **kwargs: (receipt, object()),
    )
    historical = verify_governed_timestamped_signed_restore_chain_of_custody(
        registry=registry,
        tsa_registry=object(),
        owner_id="alice",
        profile_id="tsa-1",
        timestamped_envelope_path=timestamped_path,
        request_bundle_path=tmp_path / "request.json",
        response_path=tmp_path / "response.tsr",
        trust_anchor_bundle_path=tmp_path / "root.pem",
        now=7.0,
    )
    assert historical.historical_retired_key_verified
    assert historical.trusted_timestamp_reverified


def test_private_key_permissions_and_atomic_no_overwrite(tmp_path):
    registry, _record, private, _public = registered(tmp_path)
    output = tmp_path / "signed.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sign_governed_restore_chain_of_custody(
            registry=registry,
            owner_id="alice",
            key_id="key-1",
            manifest_path=manifest_file(tmp_path),
            private_key_path=private,
            output_path=output,
            now=3.0,
        )
    if os.name != "nt":
        output.unlink()
        private.chmod(0o644)
        with pytest.raises(PermissionError, match="permissions"):
            sign_governed_restore_chain_of_custody(
                registry=registry,
                owner_id="alice",
                key_id="key-1",
                manifest_path=manifest_file(tmp_path),
                private_key_path=private,
                output_path=output,
                now=3.0,
            )
