from __future__ import annotations

import base64
import hashlib
import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile as reconcile,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    CustodyTimestampAttestation,
    _canonical_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    CustodyTimestampIssuanceAttempt,
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
    execute_custody_timestamp_issuance,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes


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


class Registry:
    def get(self, **kwargs):
        return SimpleNamespace(
            public_key_sha256="1" * 64,
            registered_at=0.0,
            retired_at=None,
        )


def seeded(tmp_path):
    value = attestation()
    output = tmp_path / "timestamp.json"
    journal = CustodyTimestampIssuanceJournal(tmp_path / "issuances.sqlite3")
    digest = hashlib.sha256(
        _canonical_bytes(value.public_payload())
    ).hexdigest()
    attempt = CustodyTimestampIssuanceAttempt.create(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="key-1",
        serial=value.serial,
        attestation_digest=digest,
        output_path_digest=timestamp_output_path_digest(output),
        max_attempts=3,
        now=10.0,
    )
    journal.seed(attempt, attestation=value)
    return journal, output, attempt


def kill_worker(journal_path, issuance_id, output, phase):
    execute_custody_timestamp_issuance(
        issuance_id,
        worker_id="killed-worker",
        lease_seconds=5,
        output_path=output,
        journal=CustodyTimestampIssuanceJournal(journal_path),
        registry=Registry(),
        now=20.0,
        _phase_hook=lambda current: os._exit(19) if current == phase else None,
    )


def process_context():
    if os.name == "nt":
        pytest.skip("independent process-kill recovery currently runs on POSIX")
    try:
        return mp.get_context("fork")
    except ValueError:
        pytest.skip("fork multiprocessing is unavailable")


@pytest.mark.parametrize(
    "phase,expected_phase",
    [
        ("after_output_publish", "planned"),
        ("after_output_phase", "output_published"),
    ],
)
def test_process_kill_recovers_after_lease_expiry(
    tmp_path,
    phase,
    expected_phase,
):
    context = process_context()
    journal, output, attempt = seeded(tmp_path)
    process = context.Process(
        target=kill_worker,
        args=(str(journal.path), attempt.issuance_id, str(output), phase),
    )
    process.start()
    process.join(20)

    assert process.exitcode == 19
    interrupted = journal.get(attempt.issuance_id)
    assert interrupted.state == "running"
    assert interrupted.phase == expected_phase
    assert output.is_file()

    recovered = execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="recovery-worker",
        lease_seconds=30,
        output_path=output,
        journal=journal,
        registry=Registry(),
        now=30.0,
    )
    assert recovered.state == "completed"
    assert recovered.existing_exact_output_reused is True
    assert journal.get(attempt.issuance_id).attempt_count == 2


def test_output_publication_failure_is_retryable_without_partial_file(
    tmp_path,
    monkeypatch,
):
    journal, output, attempt = seeded(tmp_path)
    original = reconcile._atomic_create
    monkeypatch.setattr(
        reconcile,
        "_atomic_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("simulated filesystem failure")
        ),
    )
    with pytest.raises(reconcile.CustodyTimestampIssuanceRecoveryError):
        execute_custody_timestamp_issuance(
            attempt.issuance_id,
            worker_id="worker",
            lease_seconds=30,
            output_path=output,
            journal=journal,
            registry=Registry(),
            now=20.0,
        )
    failed = journal.get(attempt.issuance_id)
    assert failed.state == "failed"
    assert failed.phase == "planned"
    assert not output.exists()

    journal.retry(
        attempt.issuance_id,
        owner_id="alice",
        confirm_issuance_id=attempt.issuance_id,
        now=21.0,
    )
    monkeypatch.setattr(reconcile, "_atomic_create", original)
    assert execute_custody_timestamp_issuance(
        attempt.issuance_id,
        worker_id="recovery",
        lease_seconds=30,
        output_path=output,
        journal=journal,
        registry=Registry(),
        now=22.0,
    ).state == "completed"
