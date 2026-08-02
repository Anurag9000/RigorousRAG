from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_set_signed_publication_cli as cli
from tools.evidence_graph_set_publish_reconcile import (
    EvidenceGraphSetPublicationExecution,
)


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def attempt():
    return SimpleNamespace(
        operation_id="1" * 64,
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=("2" * 64,),
        expected_current_set_id=None,
        state="planned",
        phase="planned",
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        previous_graph_set_id=None,
        candidate_graph_set_id=None,
        candidate_graph_set_digest=None,
        member_count=0,
        edge_count=0,
        verification_digest=None,
        failure_type=None,
        compensation_errors=(),
        created_at=1.0,
        updated_at=1.0,
        completed_at=None,
    )


class Journal:
    def __init__(self):
        self.value = attempt()

    def seed(self, value):
        self.seeded = value
        return self.value

    def get(self, operation_id):
        return self.value

    def list(self, **kwargs):
        return (self.value,)

    def retry(self, operation_id, **kwargs):
        return self.value

    def cancel(self, operation_id, **kwargs):
        return SimpleNamespace(**{**vars(self.value), "state": "cancelled"})


def install(monkeypatch):
    journal = Journal()
    monkeypatch.setattr(
        cli, "get_evidence_graph_set_publication_journal", lambda: journal
    )
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: "ledger")
    monkeypatch.setattr(
        cli, "get_relation_review_authorization_store", lambda: "authorizations"
    )
    monkeypatch.setattr(cli, "get_signed_actor_use_store", lambda: "actor-uses")
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: "sets")
    monkeypatch.setattr(cli, "get_generation_store", lambda: "generations")
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: "graphs")
    return journal


def test_seed_validates_signed_provenance_before_journaling(
    monkeypatch, capsys
):
    journal = install(monkeypatch)
    observed = {}

    def validate(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "signed_actor_publication_ledger", validate)
    assert cli.main([
        "seed",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--proposal-id", "2" * 64,
        "--expect-no-current",
    ]) == 0
    output, error = read(capsys)

    assert error is None
    assert observed["ledger"] == "ledger"
    assert observed["authorization_store"] == "authorizations"
    assert observed["actor_use_store"] == "actor-uses"
    assert journal.seeded.owner_id == "alice"
    assert output["committed_review_authorizations_required"] is True
    assert output["signed_actor_use_provenance_required_when_present"] is True


def test_execute_uses_signed_governance_dependencies(monkeypatch, capsys):
    install(monkeypatch)
    observed = {}
    result = EvidenceGraphSetPublicationExecution(
        operation_id="1" * 64,
        state="completed",
        phase="completed",
        graph_set_key="review",
        candidate_graph_set_id="3" * 64,
        candidate_graph_set_digest="4" * 64,
        previous_graph_set_id=None,
        member_count=2,
        edge_count=1,
        verification_digest="6" * 64,
        attempt_count=1,
        pointer_current_set_id="3" * 64,
        graph_set_mutation_performed=True,
        authoritative_mutation_performed=False,
    )

    def execute(operation_id, **kwargs):
        observed["operation_id"] = operation_id
        observed.update(kwargs)
        return result

    monkeypatch.setattr(cli, "execute_signed_actor_publication_attempt", execute)
    assert cli.main([
        "execute", "1" * 64,
        "--worker-id", "worker",
        "--lease-seconds", "30",
    ]) == 0
    output, error = read(capsys)

    assert error is None
    assert observed["authorization_store"] == "authorizations"
    assert observed["actor_use_store"] == "actor-uses"
    assert observed["worker_id"] == "worker"
    assert output["state"] == "completed"
    assert output["graph_set_key"] == "review"
    assert output["signed_actor_use_provenance_validated"] is True
    assert output["source_text_returned"] is False


def test_reconcile_idle_and_read_only_status(monkeypatch, capsys):
    install(monkeypatch)
    monkeypatch.setattr(
        cli, "execute_next_signed_actor_publication_attempt", lambda **kwargs: None
    )
    assert cli.main([
        "reconcile-one",
        "--owner-id", "alice",
        "--worker-id", "worker",
    ]) == 0
    idle, error = read(capsys)
    assert error is None
    assert idle == {
        "mutation_performed": False,
        "source_text_returned": False,
        "status": "idle",
    }

    assert cli.main(["status", "1" * 64]) == 0
    status, error = read(capsys)
    assert error is None
    assert status["authoritative_mutation_performed"] is False
    assert status["source_text_returned"] is False
