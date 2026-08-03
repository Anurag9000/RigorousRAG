from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export as export,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_cli as cli,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_runtime as runtime,
)
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
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority_readonly import (
    ReadOnlyCustodyTimestampAuthorityRegistry,
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
    fingerprint = hashlib.sha256(
        private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).hexdigest()
    return private_path, public_path, fingerprint


def setup_files(tmp_path):
    manifest_path = tmp_path / "chain.json"
    _atomic_create(
        manifest_path,
        _canonical_bytes(manifest().public_payload()) + b"\n",
    )
    custody_private, custody_public, _fingerprint = write_keys(tmp_path, "custody")
    envelope_path = tmp_path / "chain.signed.json"
    sign_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=envelope_path,
        key_id="custody-key",
        private_key_path=custody_private,
    )
    authority_private, authority_public, authority_fingerprint = write_keys(
        tmp_path,
        "authority",
    )
    return (
        envelope_path,
        custody_public,
        authority_private,
        authority_public,
        authority_fingerprint,
    )


def test_read_only_authority_registry_requires_initialized_file_and_refuses_writes(
    tmp_path,
):
    with pytest.raises((FileNotFoundError, ValueError)):
        ReadOnlyCustodyTimestampAuthorityRegistry(tmp_path / "missing.sqlite3")

    registry = CustodyTimestampAuthorityRegistry(tmp_path / "authorities.sqlite3")
    read_only = ReadOnlyCustodyTimestampAuthorityRegistry(registry.path)
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_timestamp_authorities"
            )


def test_timestamp_authority_runtime_rejects_aliases_and_caches(
    tmp_path,
    monkeypatch,
):
    runtime.clear_custody_timestamp_authority_registry_cache()
    protected = tmp_path / "signers.sqlite3"
    protected.write_bytes(b"not a timestamp registry")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_custody_timestamp_authority_registry(protected)

    monkeypatch.delenv("EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH")
    selected = tmp_path / "authorities.sqlite3"
    first = runtime.get_custody_timestamp_authority_registry(selected)
    second = runtime.get_custody_timestamp_authority_registry(str(selected))
    assert first is second


def test_bad_registration_confirmation_precedes_actor_and_store(
    tmp_path,
    monkeypatch,
    capsys,
):
    _envelope, _custody_public, _private, public, _fingerprint = setup_files(
        tmp_path
    )
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor"),
    )
    monkeypatch.setattr(
        cli,
        "get_custody_timestamp_authority_registry",
        lambda path: calls.append("store"),
    )

    assert cli.main(
        [
            "register",
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--public-key-path",
            str(public),
            "--confirm-public-key-sha256",
            "f" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert calls == []
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}


def test_timestamp_cli_governed_issue_retire_and_historical_verify(
    tmp_path,
    monkeypatch,
    capsys,
):
    (
        envelope_path,
        custody_public,
        authority_private,
        authority_public,
        authority_fingerprint,
    ) = setup_files(tmp_path)
    registry_path = tmp_path / "authorities.sqlite3"
    output = tmp_path / "chain.timestamp.json"
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "security-officer")

    assert cli.main(
        [
            "register",
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--public-key-path",
            str(authority_public),
            "--confirm-public-key-sha256",
            authority_fingerprint,
            "--registry-db-path",
            str(registry_path),
        ]
    ) == 0
    registered = json.loads(capsys.readouterr().out)
    assert registered["state"] == "active"
    assert registered["contains_actor_id"] is False
    assert str(authority_public) not in json.dumps(registered)

    assert cli.main(
        [
            "issue-governed",
            str(envelope_path),
            "--custody-signer-public-key-path",
            str(custody_public),
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--authority-private-key-path",
            str(authority_private),
            "--output",
            str(output),
            "--registry-db-path",
            str(registry_path),
        ]
    ) == 0
    issued = json.loads(capsys.readouterr().out)
    assert issued["attestation_created"] is True
    assert issued["rfc3161_token"] is False
    assert issued["hardware_clock_proven"] is False
    assert str(authority_private) not in json.dumps(issued)

    assert cli.main(
        [
            "retire",
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--confirm-key-id",
            "timestamp-key-1",
            "--registry-db-path",
            str(registry_path),
        ]
    ) == 0
    retired = json.loads(capsys.readouterr().out)
    assert retired["state"] == "retired"

    assert cli.main(
        [
            "verify-governed",
            str(output),
            "--signed-envelope-path",
            str(envelope_path),
            "--custody-signer-public-key-path",
            str(custody_public),
            "--authority-public-key-path",
            str(authority_public),
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--registry-db-path",
            str(registry_path),
        ]
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["historical_governance_window_valid"] is True
    assert verified["contains_raw_paths"] is False
    assert str(authority_public) not in json.dumps(verified)
