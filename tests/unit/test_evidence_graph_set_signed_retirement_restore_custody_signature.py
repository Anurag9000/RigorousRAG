from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export_cli_boundary as cli,
)
from tools import evidence_graph_set_signed_retirement_restore_custody_export as export
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    CustodyArtifactEvidence,
    RestoreChainOfCustodyManifest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    sign_restore_chain_of_custody,
    verify_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
)


def manifest():
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


def write_manifest(tmp_path):
    path = tmp_path / "chain.json"
    _atomic_create(path, _canonical_bytes(manifest().public_payload()) + b"\n")
    return path


def write_keys(tmp_path, name="key"):
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


def test_ed25519_envelope_round_trip_and_no_overwrite(tmp_path):
    manifest_path = write_manifest(tmp_path)
    private_path, public_path = write_keys(tmp_path)
    output = tmp_path / "chain.signed.json"

    signed = sign_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=output,
        key_id="custody-ed25519-1",
        private_key_path=private_path,
    )
    verified = verify_signed_restore_chain_of_custody(
        envelope_path=output,
        public_key_path=public_path,
        expected_key_id="custody-ed25519-1",
        expected_public_key_sha256=signed.public_key_sha256,
    )

    assert verified == signed
    assert signed.algorithm == "ed25519"
    assert signed.contains_private_key_material is False
    assert len(signed.signature) == 88
    with pytest.raises(FileExistsError):
        sign_restore_chain_of_custody(
            manifest_path=manifest_path,
            output_path=output,
            key_id="custody-ed25519-1",
            private_key_path=private_path,
        )


def test_ed25519_wrong_key_fingerprint_key_id_and_tamper_refuse(tmp_path):
    manifest_path = write_manifest(tmp_path)
    private_path, public_path = write_keys(tmp_path, "first")
    _wrong_private, wrong_public = write_keys(tmp_path, "wrong")
    output = tmp_path / "chain.signed.json"
    signed = sign_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=output,
        key_id="key-1",
        private_key_path=private_path,
    )

    with pytest.raises(PermissionError, match="fingerprint"):
        verify_signed_restore_chain_of_custody(
            envelope_path=output,
            public_key_path=wrong_public,
        )
    with pytest.raises(PermissionError, match="key ID"):
        verify_signed_restore_chain_of_custody(
            envelope_path=output,
            public_key_path=public_path,
            expected_key_id="key-2",
        )
    with pytest.raises(PermissionError, match="expected public-key"):
        verify_signed_restore_chain_of_custody(
            envelope_path=output,
            public_key_path=public_path,
            expected_public_key_sha256="0" * 64,
        )

    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["signature"] = "A" * 88
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((PermissionError, ValueError)):
        verify_signed_restore_chain_of_custody(
            envelope_path=output,
            public_key_path=public_path,
        )
    assert signed.public_key_sha256 != "0" * 64


def test_ed25519_private_key_permissions_and_type_fail_closed(tmp_path):
    manifest_path = write_manifest(tmp_path)
    private_path, _public_path = write_keys(tmp_path)
    if os.name != "nt":
        private_path.chmod(0o644)
        with pytest.raises(PermissionError, match="permissions"):
            sign_restore_chain_of_custody(
                manifest_path=manifest_path,
                output_path=tmp_path / "broad.json",
                key_id="broad",
                private_key_path=private_path,
            )

    invalid = tmp_path / "invalid.pem"
    invalid.write_text("not a private key", encoding="utf-8")
    if os.name != "nt":
        invalid.chmod(0o600)
    with pytest.raises(ValueError, match="invalid"):
        sign_restore_chain_of_custody(
            manifest_path=manifest_path,
            output_path=tmp_path / "invalid.json",
            key_id="invalid",
            private_key_path=invalid,
        )


def test_signature_cli_is_offline_and_secret_free(tmp_path, capsys):
    manifest_path = write_manifest(tmp_path)
    private_path, public_path = write_keys(tmp_path)
    output = tmp_path / "chain.signed.json"

    assert cli.main(
        [
            "sign",
            str(manifest_path),
            "--output",
            str(output),
            "--key-id",
            "key-1",
            "--private-key-path",
            str(private_path),
        ]
    ) == 0
    signed_summary = json.loads(capsys.readouterr().out)
    assert signed_summary["algorithm"] == "ed25519"
    assert signed_summary["publicly_verifiable"] is True
    assert signed_summary["contains_private_key_material"] is False
    assert str(private_path) not in json.dumps(signed_summary)

    assert cli.main(
        [
            "verify-signature",
            str(output),
            "--public-key-path",
            str(public_path),
            "--expected-key-id",
            "key-1",
            "--expected-public-key-sha256",
            signed_summary["public_key_sha256"],
        ]
    ) == 0
    verified_summary = json.loads(capsys.readouterr().out)
    assert verified_summary["signature_type"] == "public_key"
    assert verified_summary["import_performed"] is False
    assert verified_summary["mutation_performed"] is False
    assert str(public_path) not in json.dumps(verified_summary)
