from __future__ import annotations

import pytest

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.evidence_graph_set_signed_retirement_mutation import (
    claim_or_renew_authorization_publication_retirement_lease,
    retire_claimed_authorization_publication_attempt,
    signed_retirement_lease_owner,
)


def running_attempt(tmp_path):
    journal = EvidenceGraphSetPublicationJournal(tmp_path / "publications.sqlite3")
    value = EvidenceGraphSetPublicationAttempt.create(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=("1" * 64,),
        expected_current_set_id=None,
        max_attempts=3,
        now=1.0,
    )
    journal.seed(value)
    journal.claim(value.operation_id, worker_id="publisher", lease_seconds=2, now=2.0)
    journal.record_candidate(
        value.operation_id,
        worker_id="publisher",
        previous_graph_set_id=None,
        previous_graph_set_digest=None,
        candidate_graph_set_id="2" * 64,
        candidate_graph_set_digest="3" * 64,
        member_count=2,
        edge_count=1,
        now=3.0,
    )
    return journal, value.operation_id


def test_retirement_takes_over_only_expired_lease_without_retry_increment(tmp_path):
    journal, operation_id = running_attempt(tmp_path)
    retirement_id = "4" * 64
    claimed = claim_or_renew_authorization_publication_retirement_lease(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        lease_seconds=30,
        now=5.0,
    )
    assert claimed.state == "running"
    assert claimed.lease_owner == signed_retirement_lease_owner(retirement_id)
    assert claimed.lease_expires_at == 35.0
    assert claimed.attempt_count == 1

    renewed = claim_or_renew_authorization_publication_retirement_lease(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        lease_seconds=30,
        now=10.0,
    )
    assert renewed.lease_expires_at == 40.0
    assert renewed.attempt_count == 1


def test_live_other_worker_lease_and_scope_mismatch_fail_closed(tmp_path):
    journal, operation_id = running_attempt(tmp_path)
    with pytest.raises(RuntimeError, match="another worker"):
        claim_or_renew_authorization_publication_retirement_lease(
            journal,
            operation_id=operation_id,
            retirement_id="4" * 64,
            owner_id="alice",
            graph_set_key="review",
            expected_candidate_set_id="2" * 64,
            lease_seconds=30,
            now=3.5,
        )
    with pytest.raises(RuntimeError, match="scope"):
        claim_or_renew_authorization_publication_retirement_lease(
            journal,
            operation_id=operation_id,
            retirement_id="4" * 64,
            owner_id="alice",
            graph_set_key="other",
            expected_candidate_set_id="2" * 64,
            lease_seconds=30,
            now=5.0,
        )


def test_only_exact_retirement_lease_can_cancel_weaker_attempt(tmp_path):
    journal, operation_id = running_attempt(tmp_path)
    retirement_id = "4" * 64
    claim_or_renew_authorization_publication_retirement_lease(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        lease_seconds=30,
        now=5.0,
    )
    with pytest.raises(RuntimeError, match="not leased"):
        retire_claimed_authorization_publication_attempt(
            journal,
            operation_id=operation_id,
            retirement_id="5" * 64,
            owner_id="alice",
            graph_set_key="review",
            expected_candidate_set_id="2" * 64,
            now=6.0,
        )

    retired, mutated = retire_claimed_authorization_publication_attempt(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        now=6.0,
    )
    assert mutated is True
    assert retired.state == "cancelled"
    assert retired.phase == "candidate_stored"
    assert retired.lease_owner is None

    replayed, mutated = retire_claimed_authorization_publication_attempt(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        now=7.0,
    )
    assert replayed.state == "cancelled"
    assert mutated is False


def test_expired_retirement_lease_must_be_renewed_before_cancel(tmp_path):
    journal, operation_id = running_attempt(tmp_path)
    retirement_id = "4" * 64
    claim_or_renew_authorization_publication_retirement_lease(
        journal,
        operation_id=operation_id,
        retirement_id=retirement_id,
        owner_id="alice",
        graph_set_key="review",
        expected_candidate_set_id="2" * 64,
        lease_seconds=2,
        now=5.0,
    )
    with pytest.raises(RuntimeError, match="expired"):
        retire_claimed_authorization_publication_attempt(
            journal,
            operation_id=operation_id,
            retirement_id=retirement_id,
            owner_id="alice",
            graph_set_key="review",
            expected_candidate_set_id="2" * 64,
            now=7.0,
        )
