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
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority import (
    CustodyTimestampAuthorityRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
    CustodyTimestampIssuanceRecoveryError,
    execute_custody_timestamp_issuance,
    seed_custody_timestamp_issuance,
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
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    if os.name != "nt":
        private_path.chmod(0o600)
    return private_path, public_path


def setup(tmp_path):
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
    authority_private, authority_public = write_keys(tmp_path, "authority")
    registry = CustodyTimestampAuthorityRegistry(tmp_path / "authorities.sqlite3")
    actor = ReviewActorBinding.create(
        actor_id="security-officer",
        binding_method="process_environment",
        loaded_at=40.0,
    )
    registry.register(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        public_key_path=authority_public,
        actor=actor,
        now=45.0,
    )
    journal = CustodyTimestampIssuanceJournal(tmp_path / "issuances.sqlite3")
    output = tmp_path / "chain.timestamp.json"
    attempt, attestation = seed_custody_timestamp_issuance(
        journal=journal,
        registry=registry,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=authority_private,
        signed_envelope_path=envelope_path,
        custody_signer_public_key_path=custody_public,
        output_path=output,
        confirm_output_path_digest=timestamp_output_path_digest(output),
        now=50.0,
        nonce=b"n" * 32,
    )
    return journal, registry, output, attempt, attestation, {
        "authority_private": authority_private,
        "envelope_path": envelope_path,
        "custody_public": custody_public,
    }


def test_timestamp_issuance_normal_replay_and_unique_serial(tmp_path):
    journal, registry, output, attempt, attestation, source = setup(tmp_path)
    result = execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="worker",
        lease_seconds=60,
        output_path=output,
        journal=journal,
        registry=registry,
        now=51.0,
    )
    assert result.state == "completed"
    assert result.output_created is True
    assert json.loads(output.read_text(encoding="utf-8"))["serial"] == attestation.serial
    replay = execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="other",
        lease_seconds=60,
        output_path=output,
        journal=journal,
        registry=registry,
        now=60.0,
    )
    assert replay.existing_exact_output_reused is True

    second_output = tmp_path / "second.timestamp.json"
    with pytest.raises(RuntimeError, match="serial"):
        seed_custody_timestamp_issuance(
            journal=journal,
            registry=registry,
            owner_id="alice",
            authority_id="tsa-1",
            key_id="timestamp-key-1",
            authority_private_key_path=source["authority_private"],
            signed_envelope_path=source["envelope_path"],
            custody_signer_public_key_path=source["custody_public"],
            output_path=second_output,
            confirm_output_path_digest=timestamp_output_path_digest(second_output),
            now=50.0,
            nonce=b"n" * 32,
        )


def test_timestamp_issuance_recovers_both_output_crash_windows(tmp_path):
    journal, registry, output, attempt, _attestation, _source = setup(tmp_path)
    with pytest.raises(CustodyTimestampIssuanceRecoveryError):
        execute_custody_timestamp_issuance(
            attempt.issuance_id,
            worker_id="worker",
            lease_seconds=60,
            output_path=output,
            journal=journal,
            registry=registry,
            now=51.0,
            _phase_hook=lambda phase: (_ for _ in ()).throw(RuntimeError("crash"))
            if phase == "after_output_publish"
            else None,
        )
    assert journal.get(attempt.issuance_id).phase == "planned"
    journal.retry(
        attempt.issuance_id,
        owner_id="alice",
        confirm_issuance_id=attempt.issuance_id,
        now=52.0,
    )
    assert execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="worker-2",
        lease_seconds=60,
        output_path=output,
        journal=journal,
        registry=registry,
        now=53.0,
    ).existing_exact_output_reused is True

    second = tmp_path / "second"
    second.mkdir()
    journal2, registry2, output2, attempt2, _attestation2, _source2 = setup(second)
    with pytest.raises(CustodyTimestampIssuanceRecoveryError):
        execute_custody_timestamp_issuance(
            attempt2.issuance_id,
            worker_id="worker",
            lease_seconds=60,
            output_path=output2,
            journal=journal2,
            registry=registry2,
            now=51.0,
            _phase_hook=lambda phase: (_ for _ in ()).throw(RuntimeError("crash"))
            if phase == "after_output_phase"
            else None,
        )
    assert journal2.get(attempt2.issuance_id).phase == "output_published"
    journal2.retry(
        attempt2.issuance_id,
        owner_id="alice",
        confirm_issuance_id=attempt2.issuance_id,
        now=52.0,
    )
    assert execute_custody_timestamp_issuance(
        attempt2.issuance_id,
        worker_id="worker-2",
        lease_seconds=60,
        output_path=output2,
        journal=journal2,
        registry=registry2,
        now=53.0,
    ).state == "completed"


def test_timestamp_issuance_refuses_collision_and_missing_published_output(tmp_path):
    journal, registry, output, attempt, _attestation, _source = setup(tmp_path)
    output.write_text("different", encoding="utf-8")
    with pytest.raises(CustodyTimestampIssuanceRecoveryError):
        execute_custody_timestamp_issuance(
            attempt.issuance_id,
            worker_id="worker",
            lease_seconds=60,
            output_path=output,
            journal=journal,
            registry=registry,
            now=51.0,
        )

    second = tmp_path / "second"
    second.mkdir()
    journal2, registry2, output2, attempt2, _attestation2, _source2 = setup(second)
    claimed = journal2.claim(
        attempt2.issuance_id,
        worker_id="worker",
        lease_seconds=60,
        now=51.0,
    )
    journal2.record_output_published(
        claimed.issuance_id,
        worker_id="worker",
        now=51.0,
    )
    journal2.fail(
        claimed.issuance_id,
        worker_id="worker",
        failure_type="SimulatedCrash",
        now=51.0,
    )
    journal2.retry(
        claimed.issuance_id,
        owner_id="alice",
        confirm_issuance_id=claimed.issuance_id,
        now=52.0,
    )
    with pytest.raises(CustodyTimestampIssuanceRecoveryError):
        execute_custody_timestamp_issuance(
            claimed.issuance_id,
            worker_id="worker-2",
            lease_seconds=60,
            output_path=output2,
            journal=journal2,
            registry=registry2,
            now=53.0,
        )


def test_timestamp_issuance_retired_window_and_database_tamper(tmp_path):
    journal, registry, output, attempt, _attestation, _source = setup(tmp_path)
    actor = ReviewActorBinding.create(
        actor_id="security-officer",
        binding_method="process_environment",
        loaded_at=55.0,
    )
    registry.retire(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        confirm_key_id="timestamp-key-1",
        actor=actor,
        now=60.0,
    )
    assert execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="worker",
        lease_seconds=60,
        output_path=output,
        journal=journal,
        registry=registry,
        now=70.0,
    ).state == "completed"

    with journal._lock, journal._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_timestamp_issuances "
            "SET serial=? WHERE issuance_id=?",
            ("f" * 64, attempt.issuance_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.get(attempt.issuance_id)
