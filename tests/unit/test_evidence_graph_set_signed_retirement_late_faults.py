from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_reconcile as reconcile
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)


SIGNED_ID = "2" * 64
SIGNED_DIGEST = "3" * 64
AUTH_ID = "4" * 64
AUTHORITY_DIGEST = "5" * 64
OPERATION_ID = "1" * 64


class JournalView:
    def __init__(self, value):
        self.value = value

    def get(self, operation_id):
        assert operation_id == OPERATION_ID
        return self.value


class SetStore:
    def __init__(self):
        self.current_id = SIGNED_ID
        self.candidate = SimpleNamespace(
            owner_id="alice",
            graph_set_key="review",
            graph_set_id=SIGNED_ID,
            graph_set_digest=SIGNED_DIGEST,
            members=(
                SimpleNamespace(doc_id="doc-a"),
                SimpleNamespace(doc_id="doc-b"),
            ),
        )

    def get(self, *, owner_id, graph_set_id):
        assert owner_id == "alice"
        assert graph_set_id == SIGNED_ID
        return self.candidate

    def current(self, *, owner_id, graph_set_key):
        assert owner_id == "alice"
        assert graph_set_key == "review"
        return SimpleNamespace(graph_set_id=self.current_id)

    def commit(self, *args, **kwargs):
        raise AssertionError("pointer commit is not expected in these tests")


def setup(tmp_path):
    common = SimpleNamespace(
        owner_id="alice",
        operation_id=OPERATION_ID,
        graph_set_key="review",
        proposal_ids=("9" * 64,),
        expected_current_set_id=None,
        candidate_graph_set_id=AUTH_ID,
        state="running",
        phase="candidate_stored",
        lease_owner="old-worker",
        lease_expires_at=4.0,
    )
    signed = SimpleNamespace(
        owner_id="alice",
        operation_id=OPERATION_ID,
        graph_set_key="review",
        proposal_ids=("9" * 64,),
        expected_current_set_id=None,
        candidate_graph_set_id=SIGNED_ID,
        candidate_graph_set_digest=SIGNED_DIGEST,
        state="completed",
        phase="verified",
    )
    retirement_journal = SignedPublicationRetirementJournal(
        tmp_path / "retirements.sqlite3"
    )
    retirement = retirement_journal.seed(
        SignedPublicationRetirementAttempt.create(
            owner_id="alice",
            publication_operation_id=OPERATION_ID,
            graph_set_key="review",
            signed_candidate_set_id=SIGNED_ID,
            signed_candidate_set_digest=SIGNED_DIGEST,
            authorization_candidate_set_id=AUTH_ID,
            signed_authority_digest=AUTHORITY_DIGEST,
            max_attempts=3,
            now=1.0,
        )
    )
    return (
        common,
        JournalView(common),
        JournalView(signed),
        retirement_journal,
        retirement,
        SetStore(),
    )


def execute(retirement, *, common_journal, signed_journal, retirement_journal, store):
    return reconcile.execute_signed_publication_retirement(
        retirement.retirement_id,
        worker_id="retirer",
        lease_seconds=30,
        retirement_journal=retirement_journal,
        authorization_journal=common_journal,
        signed_journal=signed_journal,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
    )


def test_authority_drift_after_weaker_retirement_never_revives_weaker_state(
    tmp_path, monkeypatch
):
    common, common_journal, signed_journal, journal, retirement, store = setup(
        tmp_path
    )
    monkeypatch.setattr(
        reconcile,
        "_document_lock",
        lambda owner_id, doc_id: nullcontext(),
    )
    authority_calls = 0

    def authority(*args, **kwargs):
        nonlocal authority_calls
        authority_calls += 1
        return SimpleNamespace(
            authoritative_current=authority_calls <= 3,
            authority_digest=AUTHORITY_DIGEST,
        )

    monkeypatch.setattr(reconcile, "assess_graph_set_authority", authority)

    def claim(publication_journal, **kwargs):
        common.lease_owner = f"signed-retirement:{retirement.retirement_id}"
        common.lease_expires_at = 40.0
        return common

    def retire(publication_journal, **kwargs):
        common.state = "cancelled"
        common.lease_owner = None
        common.lease_expires_at = None
        return common, True

    monkeypatch.setattr(
        reconcile,
        "claim_or_renew_authorization_publication_retirement_lease",
        claim,
    )
    monkeypatch.setattr(
        reconcile,
        "retire_claimed_authorization_publication_attempt",
        retire,
    )

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common_journal=common_journal,
            signed_journal=signed_journal,
            retirement_journal=journal,
            store=store,
        )

    failed = journal.get(retirement.retirement_id)
    assert failed.state == "failed"
    assert failed.phase == "authorization_retired"
    assert common.state == "cancelled"
    assert store.current_id == SIGNED_ID


def test_expired_saga_lease_is_renewed_before_weaker_cancellation(
    tmp_path, monkeypatch
):
    common, common_journal, signed_journal, journal, retirement, store = setup(
        tmp_path
    )
    monkeypatch.setattr(
        reconcile,
        "_document_lock",
        lambda owner_id, doc_id: nullcontext(),
    )
    monkeypatch.setattr(
        reconcile,
        "assess_graph_set_authority",
        lambda *args, **kwargs: SimpleNamespace(
            authoritative_current=True,
            authority_digest=AUTHORITY_DIGEST,
        ),
    )
    claim_calls = 0

    def claim(publication_journal, **kwargs):
        nonlocal claim_calls
        claim_calls += 1
        common.lease_owner = f"signed-retirement:{retirement.retirement_id}"
        common.lease_expires_at = 9.0 if claim_calls == 1 else 40.0
        return common

    def retire(publication_journal, **kwargs):
        assert claim_calls >= 2
        assert common.lease_expires_at == 40.0
        common.state = "cancelled"
        common.lease_owner = None
        common.lease_expires_at = None
        return common, True

    monkeypatch.setattr(
        reconcile,
        "claim_or_renew_authorization_publication_retirement_lease",
        claim,
    )
    monkeypatch.setattr(
        reconcile,
        "retire_claimed_authorization_publication_attempt",
        retire,
    )

    result = execute(
        retirement,
        common_journal=common_journal,
        signed_journal=signed_journal,
        retirement_journal=journal,
        store=store,
    )

    assert result.state == "completed"
    assert claim_calls >= 2
    assert common.state == "cancelled"
