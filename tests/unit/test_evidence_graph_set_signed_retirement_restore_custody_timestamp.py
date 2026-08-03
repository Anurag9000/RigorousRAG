from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools import evidence_graph_set_signed_retirement_restore_custody_export as export
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    CustodyArtifactEvidence,
    RestoreChainOfCustodyManifest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    sign_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    issue_custody_timestamp_attestation,
    verify_custody_timestamp_attestation,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority import (
    CustodyTimestampAuthorityRegistry,
    issue_governed_custody_timestamp,
    verify_governed_custody_timestamp,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
)


def manifest() -> RestoreChainOfCustodyManifest:
    owner = "alice"
    snapshot = "1" * 64
    target = "2" * 64
    restore = deterministic_signed_retirement_restore_id(
        owner_id=owner,
        snapshot_digest=snapshot,
        target_path_digest=target,
    )
    artifact = CustodyArtifactEvidence(
        artifact_id="3" * 64,
        backup_path_digest="4" * 64,
        receipt_path_digest="5" * 64,
        backup_sha256="6" * 64,
        backup_size_bytes=100,
        receipt_digest="7" * 64,
        actor_id_digest="8" * 64,
        binding_method="process_environment",
        binding_digest="9" * 64,
        completed_at=5.0,
    )
    values = {
        "owner_id": owner,
        "restore_id": restore,
        "snapshot_digest": snapshot,
        "target_path_digest": target,
        "snapshot_record_count": 2,
        "restore_target_verification_digest": "a" * 64,
        "restore_completed_at": 20.0,
        "custody_id": "b" * 64,
        "custody_manifest_digest": "c" * 64,
        "pre_receipt_digest": artifact.receipt_digest,
        "backup_sha256": artifact.backup_sha256,
        "backup_size_bytes": artifact.backup_size_bytes,
        "pre_actor_id_digest": artifact.actor_id_digest,
        "pre_binding_method": artifact.binding_method,
        "pre_binding_digest": artifact.binding_digest,
        "pre_bound_at": 10.0,
        "post_receipt_digest": "d" * 64,
        "post_target_verification_digest": "a" * 64,
        "post_actor_id_digest": "e" * 64,
        "post_binding_method": "descriptor_file",
        "post_binding_digest": "f" * 64,
        "post_bound_at": 30.0,
        "legal_hold_status": "inactive",
        "artifacts": (artifact,),
        "generated_at": 40.0,
        "schema_version": 1,
    }
    stable = {
        "scope": "rigorousrag-external-restore-chain-of-custody-v1",
        **{**values, "artifacts": [asdict(artifact)]},
    }
    return RestoreChainOfCustodyManifest(
        **values,
        chain_digest=export._canonical_digest(stable),
    )


def write_keys(tmp_path, name: str):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{name}.private.pem"
    public_path = tmp_path / f"{name}.public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    if os.name != "nt":
        private_path.chmod(0o600)
    return private_path, public_path


def write_envelope(tmp_path):
    manifest_path = tmp_path / "chain.json"
    _atomic_create(
        manifest_path,
        _canonical_bytes(manifest().public_payload()) + b"\n",
    )
    custody_private, custody_public = write_keys(tmp_path, "custody")
    envelope_path = tmp_path / "chain.signed.json"
    sign_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=envelope_path,
        key_id="custody-key",
        private_key_path=custody_private,
    )
    return envelope_path, custody_public


def test_timestamp_attestation_round_trip_and_no_overwrite(tmp_path):
    envelope_path, custody_public = write_envelope(tmp_path)
    authority_private, authority_public = write_keys(tmp_path, "authority")
    output = tmp_path / "chain.timestamp.json"

    issued = issue_custody_timestamp_attestation(
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        output_path=output,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=authority_private,
        now=50.0,
        nonce=b"n" * 32,
    )
    verified = verify_custody_timestamp_attestation(
        attestation_path=output,
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        authority_public_key_path=authority_public,
        expected_owner_id="alice",
        expected_authority_id="tsa-1",
        expected_key_id="timestamp-key-1",
        now=50.0,
    )

    assert verified == issued
    assert issued.asserted_at == 50.0
    assert len(issued.serial) == 64
    assert issued.rfc3161_token is False
    assert issued.hardware_clock_proven is False
    assert issued.contains_private_key_material is False
    with pytest.raises(FileExistsError):
        issue_custody_timestamp_attestation(
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            output_path=output,
            owner_id="alice",
            authority_id="tsa-1",
            key_id="timestamp-key-1",
            authority_private_key_path=authority_private,
            now=50.0,
            nonce=b"n" * 32,
        )


def test_timestamp_tamper_wrong_key_future_and_chronology_refuse(tmp_path):
    envelope_path, custody_public = write_envelope(tmp_path)
    authority_private, authority_public = write_keys(tmp_path, "authority")
    _wrong_private, wrong_public = write_keys(tmp_path, "wrong")
    output = tmp_path / "chain.timestamp.json"
    issue_custody_timestamp_attestation(
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        output_path=output,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=authority_private,
        now=50.0,
        nonce=b"x" * 32,
    )

    with pytest.raises(PermissionError, match="fingerprint"):
        verify_custody_timestamp_attestation(
            attestation_path=output,
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            authority_public_key_path=wrong_public,
            now=50.0,
        )
    with pytest.raises(PermissionError, match="future"):
        verify_custody_timestamp_attestation(
            attestation_path=output,
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            authority_public_key_path=authority_public,
            now=0.0,
            maximum_future_seconds=1.0,
        )

    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["asserted_at"] = 51.0
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((PermissionError, ValueError)):
        verify_custody_timestamp_attestation(
            attestation_path=output,
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            authority_public_key_path=authority_public,
            now=51.0,
        )

    with pytest.raises(ValueError, match="predates"):
        issue_custody_timestamp_attestation(
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            output_path=tmp_path / "too-early.json",
            owner_id="alice",
            authority_id="tsa-1",
            key_id="timestamp-key-1",
            authority_private_key_path=authority_private,
            now=39.0,
            nonce=b"z" * 32,
        )


def test_governed_authority_lifecycle_and_historical_window(tmp_path):
    envelope_path, custody_public = write_envelope(tmp_path)
    authority_private, authority_public = write_keys(tmp_path, "authority")
    registry = CustodyTimestampAuthorityRegistry(tmp_path / "authorities.sqlite3")
    actor = ReviewActorBinding.create(
        actor_id="security-officer",
        binding_method="process_environment",
        loaded_at=40.0,
    )
    record = registry.register(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        public_key_path=authority_public,
        actor=actor,
        now=45.0,
    )
    replay = registry.register(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        public_key_path=authority_public,
        actor=actor,
        now=999.0,
    )
    assert replay == record

    output = tmp_path / "governed.timestamp.json"
    issued = issue_governed_custody_timestamp(
        registry=registry,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=authority_private,
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        output_path=output,
        now=50.0,
        nonce=b"g" * 32,
    )
    retired = registry.retire(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        confirm_key_id="timestamp-key-1",
        actor=actor,
        now=60.0,
    )
    assert retired.state == "retired"
    assert verify_governed_custody_timestamp(
        registry=registry,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        attestation_path=output,
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        authority_public_key_path=authority_public,
        now=70.0,
    ) == issued
    with pytest.raises(PermissionError, match="not active"):
        issue_governed_custody_timestamp(
            registry=registry,
            owner_id="alice",
            authority_id="tsa-1",
            key_id="timestamp-key-1",
            authority_private_key_path=authority_private,
            signed_envelope_path=envelope_path,
            custody_signer_public_key_path=custody_public,
            output_path=tmp_path / "after-retirement.json",
            now=70.0,
        )


def test_authority_registry_collision_tamper_and_identity_fail_closed(tmp_path):
    _private_one, public_one = write_keys(tmp_path, "one")
    _private_two, public_two = write_keys(tmp_path, "two")
    actor = ReviewActorBinding.create(
        actor_id="security-officer",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    path = tmp_path / "authorities.sqlite3"
    registry = CustodyTimestampAuthorityRegistry(path)
    record = registry.register(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="key-1",
        public_key_path=public_one,
        actor=actor,
        now=1.0,
    )
    with pytest.raises(RuntimeError, match="collision"):
        registry.register(
            owner_id="alice",
            authority_id="tsa-1",
            key_id="key-1",
            public_key_path=public_two,
            actor=actor,
            now=2.0,
        )

    with registry._lock, registry._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_timestamp_authorities "
            "SET state='retired' WHERE owner_id='alice' "
            "AND authority_id='tsa-1' AND key_id='key-1'"
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        registry.get(owner_id="alice", authority_id="tsa-1", key_id="key-1")

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        registry.list(owner_id="alice")
    assert len(record.record_digest) == 64
