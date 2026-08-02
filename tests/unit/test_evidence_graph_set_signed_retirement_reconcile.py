from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools import evidence_graph_set_signed_retirement_reconcile as reconcile


SIGNED_ID = "2" * 64
SIGNED_DIGEST = "3" * 64
AUTH_ID = "4" * 64
AUTH_DIGEST = "6" * 64
AUTHORITY_DIGEST = "5" * 64
PROPOSAL_ID = "1" * 64


class SetStore:
    def __init__(self, *, current_id):
        self.current_id = current_id
        self.values = {
            SIGNED_ID: SimpleNamespace(
                owner_id="alice",
                graph_set_key="review",
                graph_set_id=SIGNED_ID,
                graph_set_digest=SIGNED_DIGEST,
                members=(SimpleNamespace(doc_id="doc-a"), SimpleNamespace(doc_id="doc-b")),
            ),
            AUTH_ID: SimpleNamespace(
                owner_id="alice",
                graph_set_key="review",
                graph_set_id=AUTH_ID,
                graph_set_digest=AUTH_DIGEST,
                members=(SimpleNamespace(doc_id="doc-a"), SimpleNamespace(doc_id="doc-b")),
            ),
        }
        self.commits = []

    def get(self, *, owner_id, graph_set_id):
        value = self.values[graph_set_id]
        assert value.owner_id == owner_id
        return value

    def current(self, *, owner_id, graph_set_key):
        assert owner_id == "alice"
        assert graph_set_key == "review"
        return None if self.current_id is None else self.values.get(
            self.current_id,
            SimpleNamespace(
                owner_id="alice",
                graph_set_key="review",
                graph_set_id=self.current_id,
                graph_set_digest="e" * 64,
            ),
        )

    def commit(self, value, *, make_current, expected_current_set_id, now):
        assert make_current is True
        if self.current_id != expected_current_set_id:
            raise RuntimeError("graph set current pointer changed concurrently.")
        self.current_id = value.graph_set_id
        self.commits.append((value.graph_set_id, expected_current_set_id, now))
        return value


def publication_journals(tmp_path):
    common = EvidenceGraphSetPublicationJournal(tmp_path / "common.sqlite3")
    signed = EvidenceGraphSetPublicationJournal(tmp_path / "signed.sqlite3")
    template = EvidenceGraphSetPublicationAttempt.create(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(PROPOSAL_ID,),
        expected_current_set_id=None,
        max_attempts=5,
        now=1.0,
    )
    common.seed(template)
    signed.seed(template)

    common.claim(template.operation_id, worker_id="common", lease_seconds=2, now=2.0)
    common.record_candidate(
        template.operation_id,
        worker_id="common",
        previous_graph_set_id=None,
        previous_graph_set_digest=None,
        candidate_graph_set_id=AUTH_ID,
        candidate_graph_set_digest=AUTH_DIGEST,
        member_count=2,
        edge_count=1,
        now=3.0,
    )

    signed.claim(template.operation_id, worker_id="signed", lease_seconds=30, now=2.0)
    signed.record_candidate(
        template.operation_id,
        worker_id="signed",
        previous_graph_set_id=None,
        previous_graph_set_digest=None,
        candidate_graph_set_id=SIGNED_ID,
        candidate_graph_set_digest=SIGNED_DIGEST,
        member_count=2,
        edge_count=1,
        now=3.0,
    )
    signed.record_pointer_activated(
        template.operation_id, worker_id="signed", now=4.0
    )
    signed.complete(
        template.operation_id,
        worker_id="signed",
        verification_digest=AUTHORITY_DIGEST,
        now=5.0,
    )
    return common, signed, template.operation_id


def retirement_journal(tmp_path, operation_id):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    value = SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id=operation_id,
        graph_set_key="review",
        signed_candidate_set_id=SIGNED_ID,
        signed_candidate_set_digest=SIGNED_DIGEST,
        authorization_candidate_set_id=AUTH_ID,
        signed_authority_digest=AUTHORITY_DIGEST,
        max_attempts=5,
        now=6.0,
    )
    return journal, journal.seed(value)


def install(monkeypatch, *, authoritative=True):
    monkeypatch.setattr(
        reconcile,
        "_document_lock",
        lambda owner_id, doc_id: nullcontext(),
    )
    monkeypatch.setattr(
        reconcile,
        "assess_graph_set_authority",
        lambda candidate, **kwargs: SimpleNamespace(
            authoritative_current=authoritative,
            authority_digest=AUTHORITY_DIGEST,
        ),
    )


def execute(
    retirement,
    *,
    common,
    signed,
    store,
    journal,
    hook=None,
):
    return reconcile.execute_signed_publication_retirement(
        retirement.retirement_id,
        worker_id="retirer",
        lease_seconds=30,
        retirement_journal=journal,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
        _phase_hook=hook,
    )


def test_restores_signed_pointer_and_retires_weaker_attempt(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=AUTH_ID)

    result = execute(
        retirement,
        common=common,
        signed=signed,
        store=store,
        journal=journal,
    )

    assert result.state == "completed"
    assert result.phase == "verified"
    assert result.pointer_mutation_performed is True
    assert result.authorization_journal_mutation_performed is True
    assert store.current_id == SIGNED_ID
    assert len(store.commits) == 1
    assert common.get(operation_id).state == "cancelled"
    assert journal.get(retirement.retirement_id).verification_digest is not None


def test_already_signed_pointer_retires_without_pointer_mutation(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    result = execute(
        retirement,
        common=common,
        signed=signed,
        store=store,
        journal=journal,
    )

    assert result.state == "completed"
    assert result.pointer_mutation_performed is False
    assert result.authorization_journal_mutation_performed is True
    assert store.commits == []


def test_recovers_crash_after_pointer_commit_before_phase_record(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=AUTH_ID)

    def crash(name, attempt):
        if name == "signed_pointer_committed":
            raise RuntimeError("crash")

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
            hook=crash,
        )
    failed = journal.get(retirement.retirement_id)
    assert failed.state == "failed"
    assert failed.phase == "pointer_restore_intent"
    assert store.current_id == SIGNED_ID
    assert common.get(operation_id).state == "running"

    journal.retry(
        retirement.retirement_id,
        owner_id="alice",
        confirm_retirement_id=retirement.retirement_id,
        now=11.0,
    )
    result = reconcile.execute_signed_publication_retirement(
        retirement.retirement_id,
        worker_id="retirer-2",
        lease_seconds=30,
        retirement_journal=journal,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=12.0,
    )
    assert result.state == "completed"
    assert store.current_id == SIGNED_ID
    assert len(store.commits) == 1


def test_recovers_crash_after_weaker_cancel_before_phase_record(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    def crash(name, attempt):
        if name == "authorization_attempt_retired":
            raise RuntimeError("crash")

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
            hook=crash,
        )
    assert journal.get(retirement.retirement_id).phase == "pointer_safe"
    assert common.get(operation_id).state == "cancelled"

    journal.retry(
        retirement.retirement_id,
        owner_id="alice",
        confirm_retirement_id=retirement.retirement_id,
        now=11.0,
    )
    result = reconcile.execute_signed_publication_retirement(
        retirement.retirement_id,
        worker_id="retirer-2",
        lease_seconds=30,
        retirement_journal=journal,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=12.0,
    )
    assert result.state == "completed"
    assert result.authorization_journal_mutation_performed is False


def test_external_pointer_after_intent_is_preserved(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)
    external = "e" * 64

    def move(name, attempt):
        if name == "pointer_restore_intent":
            store.current_id = external

    result = execute(
        retirement,
        common=common,
        signed=signed,
        store=store,
        journal=journal,
        hook=move,
    )
    assert result.state == "completed"
    assert result.final_pointer_set_id == external
    assert store.current_id == external
    assert store.commits == []
    assert common.get(operation_id).state == "cancelled"


def test_external_pointer_before_intent_fails_without_mutation(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id="e" * 64)

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
        )
    failed = journal.get(retirement.retirement_id)
    assert failed.phase == "planned"
    assert common.get(operation_id).state == "running"
    assert store.commits == []


def test_stale_signed_candidate_fails_before_durable_intent(tmp_path, monkeypatch):
    install(monkeypatch, authoritative=False)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
        )
    assert journal.get(retirement.retirement_id).phase == "planned"
    assert common.get(operation_id).state == "running"


def test_live_other_worker_lease_blocks_retirement_after_intent(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    # Reclaim the expired common lease before retirement starts.
    common.claim(operation_id, worker_id="other", lease_seconds=100, now=5.0)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
        )
    assert journal.get(retirement.retirement_id).phase == "pointer_restore_intent"
    assert common.get(operation_id).lease_owner == "other"
    assert store.commits == []


def test_weaker_pointer_reactivation_after_pointer_safe_refuses_cancel(
    tmp_path, monkeypatch
):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    def reactivate(name, attempt):
        if name == "pointer_safe_recorded":
            store.current_id = AUTH_ID

    with pytest.raises(reconcile.SignedPublicationRetirementRecoveryError):
        execute(
            retirement,
            common=common,
            signed=signed,
            store=store,
            journal=journal,
            hook=reactivate,
        )
    assert common.get(operation_id).state == "running"
    assert journal.get(retirement.retirement_id).phase == "pointer_safe"


def test_execute_next_returns_idle_or_runs_oldest_attempt(tmp_path, monkeypatch):
    install(monkeypatch)
    common, signed, operation_id = publication_journals(tmp_path)
    journal, retirement = retirement_journal(tmp_path, operation_id)
    store = SetStore(current_id=SIGNED_ID)

    result = reconcile.execute_next_signed_publication_retirement(
        owner_id="alice",
        worker_id="retirer",
        lease_seconds=30,
        retirement_journal=journal,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
    )
    assert result is not None and result.state == "completed"
    assert reconcile.execute_next_signed_publication_retirement(
        owner_id="alice",
        worker_id="retirer",
        lease_seconds=30,
        retirement_journal=journal,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=11.0,
    ) is None
