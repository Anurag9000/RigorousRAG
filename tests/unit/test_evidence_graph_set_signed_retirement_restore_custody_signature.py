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
from tools.evidence_graph_set_signed_retirement_restore_custody_export import (
    _canonical_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    Rfc3161TimestampVerificationReceipt,
    canonical_digest as rfc3161_canonical_digest,
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
    return ReviewActorBinding.create(
        actor_id=f"actor-{digit}",
        binding_method="process_environment",
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
    snapshot_digest = "1" * 64
    target_path_digest = "2" * 64
    restore_id = deterministic_signed_retirement_restore_id(
        owner_id="alice",
        snapshot_digest=snapshot_digest,
        target_path_digest=target_path_digest,
    )
    artifact = CustodyArtifactEvidence(
        artifact_id="a" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="4" * 64,
        backup_sha256="5" * 64,
        backup_size_bytes=10,
        receipt_digest="6" * 64,
        actor_id_digest="7" * 64,
        binding_method="process_environment",
        binding_digest="8" * 64,
        completed_at=1.0,
    )
    fields = {
        "owner_id": "alice",
        "restore_id": restore_id,
        "snapshot_digest": snapshot_digest,
        "target_path_digest": target_path_digest,
        "snapshot_record_count": 1,
        "restore_target_verification_digest": "9" * 64,
        "restore_completed_at": 1.0,
        "custody_id": "b" * 64,
        "custody_manifest_digest": "c" * 64,
        "pre_receipt_digest": artifact.receipt_digest,
        "backup_sha256": artifact.backup_sha256,
        "backup_size_bytes": artifact.backup_size_bytes,
        "pre_actor_id_digest": artifact.actor_id_digest,
        "pre_binding_method": artifact.binding_method,
        "pre_binding_digest": artifact.binding_digest,
        "pre_bound_at": 1.0,
        "post_receipt_digest": "d" * 64,
        "post_target_verification_digest": "9" * 64,
        "post_actor_id_digest": "e" * 64,
        "post_binding_method": "process_environment",
        "post_binding_digest": "f" * 64,
        "post_bound_at": 2.0,
        "legal_hold_status": "not_checked",
        "artifacts": (artifact,),
        "generated_at": 2.0,
        "schema_version": 1,
    }
    stable = {
        "scope": "rigorousrag-external-restore-chain-of-custody-v1",
        **{
            key: ([artifact.__dict__] if key == "artifacts" else value)
            for key, value in fields.items()
        },
    }
    value = RestoreChainOfCustodyManifest(
        **fields,
        chain_digest=_canonical_digest(stable),
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
    receipt_fields = {
        "owner_id": "alice",
        "request_bundle_digest": "1" * 64,
        "request_sha256": "2" * 64,
        "subject_sha256": subject,
        "response_sha256": "3" * 64,
        "token_sha256": "4" * 64,
        "status": "granted",
        "policy_oid": "1.2.3.4",
        "message_imprint_sha256": subject,
        "nonce_sha256": "5" * 64,
        "serial_decimal": "1",
        "generated_at_rfc3339": "1970-01-01T00:00:05Z",
        "generated_at_unix": 5.0,
        "accuracy_seconds": None,
        "accuracy_millis": None,
        "accuracy_micros": None,
        "ordering": False,
        "signer_certificate_sha256": "6" * 64,
        "signer_certificate_serial_hex": "01",
        "signer_public_key_algorithm": "ed25519",
        "signature_algorithm": "ed25519",
        "digest_algorithm": "sha256",
        "trust_anchor_bundle_sha256": "7" * 64,
        "untrusted_bundle_sha256": None,
        "crl_bundle_sha256": None,
        "verifier_version_sha256": "8" * 64,
        "schema_version": 1,
    }
    receipt_stable = {
        "scope": "rigorousrag-restore-custody-rfc3161-receipt-v1",
        **receipt_fields,
    }
    receipt = Rfc3161TimestampVerificationReceipt(
        **receipt_fields,
        receipt_digest=rfc3161_canonical_digest(receipt_stable),
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
