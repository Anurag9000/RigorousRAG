from __future__ import annotations

import pytest

from tools import evidence_graph_set_signed_retirement_boundary as boundary
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)


def seeded(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    value = journal.seed(
        SignedPublicationRetirementAttempt.create(
            owner_id="alice",
            publication_operation_id="1" * 64,
            graph_set_key="review",
            signed_candidate_set_id="2" * 64,
            signed_candidate_set_digest="3" * 64,
            authorization_candidate_set_id="4" * 64,
            signed_authority_digest="5" * 64,
            now=1.0,
        )
    )
    return journal, value


def dependencies(journal):
    return {
        "retirement_journal": journal,
        "authorization_journal": object(),
        "signed_journal": object(),
        "set_store": object(),
        "generations": object(),
        "graphs": object(),
    }


def test_boundary_records_raw_post_claim_failure(tmp_path, monkeypatch):
    journal, value = seeded(tmp_path)

    def fail(retirement_id, **kwargs):
        journal.claim(
            retirement_id,
            worker_id=kwargs["worker_id"],
            lease_seconds=kwargs["lease_seconds"],
            now=kwargs["now"],
        )
        raise RuntimeError("candidate corrupt")

    monkeypatch.setattr(boundary, "_execute", fail)
    with pytest.raises(
        boundary.SignedPublicationRetirementRecoveryError
    ) as captured:
        boundary.execute_signed_publication_retirement(
            value.retirement_id,
            worker_id="worker",
            lease_seconds=30,
            now=2.0,
            **dependencies(journal),
        )
    stored = journal.get(value.retirement_id)
    assert stored.state == "failed"
    assert stored.failure_type == "RuntimeError"
    assert captured.value.state == "failed"
    assert captured.value.phase == "planned"


def test_boundary_preserves_existing_recovery_error(tmp_path, monkeypatch):
    journal, value = seeded(tmp_path)
    marker = boundary.SignedPublicationRetirementRecoveryError(
        "failed",
        retirement_id=value.retirement_id,
        state="failed",
        phase="planned",
    )

    def fail(*args, **kwargs):
        raise marker

    monkeypatch.setattr(boundary, "_execute", fail)
    with pytest.raises(boundary.SignedPublicationRetirementRecoveryError) as captured:
        boundary.execute_signed_publication_retirement(
            value.retirement_id,
            worker_id="worker",
            lease_seconds=30,
            now=2.0,
            **dependencies(journal),
        )
    assert captured.value is marker
    assert journal.get(value.retirement_id).state == "planned"


def test_boundary_reconcile_captures_one_timestamp(tmp_path, monkeypatch):
    journal, value = seeded(tmp_path)
    observed = {}

    def execute(retirement_id, **kwargs):
        observed["retirement_id"] = retirement_id
        observed.update(kwargs)
        return "done"

    monkeypatch.setattr(boundary, "execute_signed_publication_retirement", execute)
    result = boundary.execute_next_signed_publication_retirement(
        owner_id="alice",
        worker_id="worker",
        lease_seconds=30,
        now=9.0,
        **dependencies(journal),
    )
    assert result == "done"
    assert observed["retirement_id"] == value.retirement_id
    assert observed["now"] == 9.0
