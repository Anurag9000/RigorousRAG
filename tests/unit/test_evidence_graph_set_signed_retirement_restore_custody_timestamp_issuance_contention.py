from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path

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
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority_readonly import (
    ReadOnlyCustodyTimestampAuthorityRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds import (
    CustodyTimestampIssuanceHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_readonly import (
    ReadOnlyCustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
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


def setup_source(tmp_path):
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
    registry_path = tmp_path / "authorities.sqlite3"
    registry = CustodyTimestampAuthorityRegistry(registry_path)
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
    return {
        "registry_path": str(registry_path),
        "authority_private": str(authority_private),
        "envelope": str(envelope_path),
        "custody_public": str(custody_public),
    }


def seed_worker(queue, journal_path, setup, output, nonce):
    try:
        attempt, _attestation = seed_custody_timestamp_issuance(
            journal=CustodyTimestampIssuanceJournal(journal_path),
            registry=ReadOnlyCustodyTimestampAuthorityRegistry(
                setup["registry_path"]
            ),
            owner_id="alice",
            authority_id="tsa-1",
            key_id="timestamp-key-1",
            authority_private_key_path=setup["authority_private"],
            signed_envelope_path=setup["envelope"],
            custody_signer_public_key_path=setup["custody_public"],
            output_path=output,
            confirm_output_path_digest=timestamp_output_path_digest(output),
            now=50.0,
            nonce=nonce,
        )
        queue.put(("ok", attempt.issuance_id, attempt.serial))
    except Exception as exc:
        queue.put(("error", type(exc).__name__, str(exc)))


def execute_worker(
    queue,
    journal_path,
    registry_path,
    issuance_id,
    output,
    worker,
):
    try:
        result = execute_custody_timestamp_issuance(
            issuance_id,
            worker_id=worker,
            lease_seconds=30,
            output_path=output,
            journal=CustodyTimestampIssuanceJournal(journal_path),
            registry=ReadOnlyCustodyTimestampAuthorityRegistry(registry_path),
            now=60.0,
        )
        queue.put(("ok", result.issuance_id, result.state))
    except Exception as exc:
        queue.put(("error", issuance_id, type(exc).__name__, str(exc)))


def hold_worker(queue, hold_path, issuance_path, issuance_id, reason):
    try:
        actor = ReviewActorBinding.create(
            actor_id="legal-officer",
            binding_method="process_environment",
            loaded_at=70.0,
        )
        hold = CustodyTimestampIssuanceHoldStore(hold_path).place(
            owner_id="alice",
            issuance_id=issuance_id,
            hold_key="matter-1",
            reason_code=reason,
            actor=actor,
            issuance_journal=ReadOnlyCustodyTimestampIssuanceJournal(
                issuance_path
            ),
            now=70.0,
        )
        queue.put(("ok", hold.hold_id, hold.reason_code))
    except Exception as exc:
        queue.put(("error", type(exc).__name__, str(exc)))


def process_context():
    if os.name == "nt":
        pytest.skip("independent-process contention currently runs on POSIX")
    try:
        return mp.get_context("fork")
    except ValueError:
        pytest.skip("fork multiprocessing is unavailable")


def run_two(context, target, first_args, second_args):
    queue = context.Queue()
    first = context.Process(target=target, args=(queue, *first_args))
    second = context.Process(target=target, args=(queue, *second_args))
    first.start()
    second.start()
    first.join(20)
    second.join(20)
    assert first.exitcode == 0
    assert second.exitcode == 0
    return [queue.get(timeout=2), queue.get(timeout=2)]


def test_independent_process_serial_reservation_is_idempotent_or_exclusive(
    tmp_path,
):
    context = process_context()
    setup = setup_source(tmp_path)
    journal_path = str(tmp_path / "issuances.sqlite3")
    output = str(tmp_path / "same.json")
    same = run_two(
        context,
        seed_worker,
        (journal_path, setup, output, b"n" * 32),
        (journal_path, setup, output, b"n" * 32),
    )
    assert all(item[0] == "ok" for item in same)
    assert len({item[1] for item in same}) == 1

    other_journal = str(tmp_path / "other-issuances.sqlite3")
    different = run_two(
        context,
        seed_worker,
        (other_journal, setup, str(tmp_path / "one.json"), b"x" * 32),
        (other_journal, setup, str(tmp_path / "two.json"), b"x" * 32),
    )
    assert sorted(item[0] for item in different) == ["error", "ok"]
    assert any(
        "serial" in item[-1].lower()
        for item in different
        if item[0] == "error"
    )


def test_independent_process_output_path_contention_has_one_winner(tmp_path):
    context = process_context()
    setup = setup_source(tmp_path)
    journal_path = str(tmp_path / "issuances.sqlite3")
    journal = CustodyTimestampIssuanceJournal(journal_path)
    output = str(tmp_path / "shared.json")
    registry = ReadOnlyCustodyTimestampAuthorityRegistry(setup["registry_path"])
    first, _ = seed_custody_timestamp_issuance(
        journal=journal,
        registry=registry,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=setup["authority_private"],
        signed_envelope_path=setup["envelope"],
        custody_signer_public_key_path=setup["custody_public"],
        output_path=output,
        confirm_output_path_digest=timestamp_output_path_digest(output),
        now=50.0,
        nonce=b"a" * 32,
    )
    second, _ = seed_custody_timestamp_issuance(
        journal=journal,
        registry=registry,
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=setup["authority_private"],
        signed_envelope_path=setup["envelope"],
        custody_signer_public_key_path=setup["custody_public"],
        output_path=output,
        confirm_output_path_digest=timestamp_output_path_digest(output),
        now=51.0,
        nonce=b"b" * 32,
    )
    results = run_two(
        context,
        execute_worker,
        (
            journal_path,
            setup["registry_path"],
            first.issuance_id,
            output,
            "worker-one",
        ),
        (
            journal_path,
            setup["registry_path"],
            second.issuance_id,
            output,
            "worker-two",
        ),
    )
    assert sorted(item[0] for item in results) == ["error", "ok"]
    assert {
        journal.get(first.issuance_id).state,
        journal.get(second.issuance_id).state,
    } == {"completed", "failed"}
    assert Path(output).is_file()


def test_independent_process_hold_placement_is_idempotent_and_conflict_refuses(
    tmp_path,
):
    context = process_context()
    setup = setup_source(tmp_path)
    issuance_path = str(tmp_path / "issuances.sqlite3")
    output = str(tmp_path / "out.json")
    attempt, _ = seed_custody_timestamp_issuance(
        journal=CustodyTimestampIssuanceJournal(issuance_path),
        registry=ReadOnlyCustodyTimestampAuthorityRegistry(setup["registry_path"]),
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        authority_private_key_path=setup["authority_private"],
        signed_envelope_path=setup["envelope"],
        custody_signer_public_key_path=setup["custody_public"],
        output_path=output,
        confirm_output_path_digest=timestamp_output_path_digest(output),
        now=50.0,
        nonce=b"c" * 32,
    )
    hold_path = str(tmp_path / "holds.sqlite3")
    same = run_two(
        context,
        hold_worker,
        (hold_path, issuance_path, attempt.issuance_id, "legal_matter"),
        (hold_path, issuance_path, attempt.issuance_id, "legal_matter"),
    )
    assert all(item[0] == "ok" for item in same)
    assert len({item[1] for item in same}) == 1

    conflict_path = str(tmp_path / "conflicting-holds.sqlite3")
    conflict = run_two(
        context,
        hold_worker,
        (conflict_path, issuance_path, attempt.issuance_id, "legal_matter"),
        (
            conflict_path,
            issuance_path,
            attempt.issuance_id,
            "regulatory_matter",
        ),
    )
    assert sorted(item[0] for item in conflict) == ["error", "ok"]
    assert any(
        "collision" in item[-1].lower()
        for item in conflict
        if item[0] == "error"
    )
