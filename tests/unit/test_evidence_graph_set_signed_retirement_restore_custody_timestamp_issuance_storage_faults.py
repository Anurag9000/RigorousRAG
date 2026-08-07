from __future__ import annotations

import base64
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    CustodyTimestampAttestation,
    _canonical_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    CustodyTimestampIssuanceAttempt,
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds import (
    CustodyTimestampIssuanceHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
    seed_custody_timestamp_issuance,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes


class NoWaitIssuanceJournal(CustodyTimestampIssuanceJournal):
    def _connect(self):
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=0.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        return connection


class NoWaitHoldStore(CustodyTimestampIssuanceHoldStore):
    def _connect(self):
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=0.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        return connection


def attestation():
    stable = {
        "scope": "rigorousrag-restore-custody-timestamp-attestation-v1",
        "owner_id": "alice",
        "authority_id": "tsa-1",
        "key_id": "key-1",
        "algorithm": "ed25519",
        "public_key_sha256": "1" * 64,
        "custody_envelope_sha256": "2" * 64,
        "custody_manifest_digest": "3" * 64,
        "custody_chain_digest": "4" * 64,
        "asserted_at": 10.0,
        "nonce_sha256": "5" * 64,
        "schema_version": 1,
    }
    return CustodyTimestampAttestation(
        **{key: value for key, value in stable.items() if key != "scope"},
        serial=_canonical_digest(stable),
        signature=base64.b64encode(b"s" * 64).decode("ascii"),
    )


def attempt_for(output):
    value = attestation()
    digest = hashlib.sha256(
        _canonical_bytes(value.public_payload())
    ).hexdigest()
    return (
        CustodyTimestampIssuanceAttempt.create(
            owner_id="alice",
            authority_id="tsa-1",
            key_id="key-1",
            serial=value.serial,
            attestation_digest=digest,
            output_path_digest=timestamp_output_path_digest(output),
            now=1.0,
        ),
        value,
    )


def test_sqlite_lock_refuses_issuance_insert_without_partial_row(tmp_path):
    path = tmp_path / "issuances.sqlite3"
    journal = NoWaitIssuanceJournal(path)
    attempt, value = attempt_for(tmp_path / "output.json")
    locker = sqlite3.connect(path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            journal.seed(attempt, attestation=value)
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    with pytest.raises(KeyError):
        journal.get(attempt.issuance_id)


def test_sqlite_lock_refuses_hold_insert_without_partial_row(tmp_path):
    path = tmp_path / "holds.sqlite3"
    store = NoWaitHoldStore(path)
    issuance_id = "1" * 64
    locker = sqlite3.connect(path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    actor = ReviewActorBinding.create(
        actor_id="officer",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.place(
                owner_id="alice",
                issuance_id=issuance_id,
                hold_key="matter",
                reason_code="legal_matter",
                actor=actor,
                issuance_journal=SimpleNamespace(
                    get=lambda value: SimpleNamespace(owner_id="alice")
                ),
                now=2.0,
            )
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert store.list(owner_id="alice") == ()


def test_missing_private_key_creates_no_issuance_or_output(tmp_path):
    journal = CustodyTimestampIssuanceJournal(tmp_path / "issuances.sqlite3")
    output = tmp_path / "timestamp.json"
    registry = SimpleNamespace(
        get=lambda **kwargs: SimpleNamespace(
            state="active",
            public_key_sha256="1" * 64,
            registered_at=0.0,
            retired_at=None,
            owner_id="alice",
            authority_id="tsa-1",
            key_id="key-1",
        )
    )
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        seed_custody_timestamp_issuance(
            journal=journal,
            registry=registry,
            owner_id="alice",
            authority_id="tsa-1",
            key_id="key-1",
            authority_private_key_path=tmp_path / "missing-private.pem",
            signed_envelope_path=tmp_path / "missing-envelope.json",
            custody_signer_public_key_path=tmp_path / "missing-public.pem",
            output_path=output,
            confirm_output_path_digest=timestamp_output_path_digest(output),
            now=10.0,
            nonce=b"n" * 32,
        )
    assert journal.list(owner_id="alice") == ()
    assert not output.exists()
