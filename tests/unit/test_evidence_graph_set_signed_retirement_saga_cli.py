from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_set_signed_retirement_saga_cli as cli
from tools.evidence_graph_set_signed_retirement_reconcile import (
    SignedPublicationRetirementExecution,
    SignedPublicationRetirementRecoveryError,
)


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def attempt(*, state="planned", phase="planned"):
    return SimpleNamespace(
        retirement_id="a" * 64,
        owner_id="alice",
        publication_operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        state=state,
        phase=phase,
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        final_pointer_set_id=None,
        verification_digest=None,
        failure_type=None,
        created_at=1.0,
        updated_at=1.0,
        completed_at=None,
    )


class Journal:
    def __init__(self):
        self.value = attempt()

    def get(self, retirement_id):
        return self.value

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return (self.value,)

    def retry(self, retirement_id, **kwargs):
        self.retry_kwargs = kwargs
        return self.value

    def cancel(self, retirement_id, **kwargs):
        self.cancel_kwargs = kwargs
        return self.value


def install(monkeypatch):
    journal = Journal()
    dependencies = {
        "retirement_journal": journal,
        "authorization_journal": "authorization",
        "signed_journal": "signed",
        "set_store": "sets",
        "generations": "generations",
        "graphs": "graphs",
    }
    monkeypatch.setattr(cli, "_dependencies", lambda: dependencies)
    return journal, dependencies


def test_seed_requires_exact_confirmation_and_reports_preflight(
    monkeypatch, capsys
):
    journal, dependencies = install(monkeypatch)
    observed = {}

    def seed(**kwargs):
        observed.update(kwargs)
        return journal.value, SimpleNamespace(
            report_digest="6" * 64,
            disposition="retire_expired_journal_only",
            eligible=True,
        )

    monkeypatch.setattr(cli, "seed_signed_publication_retirement", seed)
    assert cli.main([
        "seed",
        "1" * 64,
        "--owner-id", "alice",
        "--confirm-operation-id", "1" * 64,
    ]) == 0
    output, error = read(capsys)
    assert error is None
    assert observed["authorization_journal"] == "authorization"
    assert output["preflight_eligible"] is True
    assert output["retirement_journal_mutation_performed"] is True
    assert output["publication_mutation_performed"] is False
    assert output["source_text_returned"] is False

    assert cli.main([
        "seed",
        "1" * 64,
        "--owner-id", "alice",
        "--confirm-operation-id", "2" * 64,
    ]) == 2
    _output, error = read(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_execute_and_reconcile_emit_text_free_recovery_results(monkeypatch, capsys):
    _journal, _dependencies = install(monkeypatch)
    result = SignedPublicationRetirementExecution(
        retirement_id="a" * 64,
        publication_operation_id="1" * 64,
        state="completed",
        phase="verified",
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        authorization_candidate_set_id="4" * 64,
        final_pointer_set_id="2" * 64,
        verification_digest="6" * 64,
        attempt_count=1,
        pointer_mutation_performed=True,
        authorization_journal_mutation_performed=True,
    )
    monkeypatch.setattr(
        cli, "execute_signed_publication_retirement", lambda *args, **kwargs: result
    )
    assert cli.main([
        "execute", "a" * 64,
        "--worker-id", "worker",
    ]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["state"] == "completed"
    assert output["weaker_pointer_restoration_performed"] is False
    assert output["source_text_returned"] is False

    monkeypatch.setattr(
        cli, "execute_next_signed_publication_retirement", lambda **kwargs: None
    )
    assert cli.main([
        "reconcile-one",
        "--owner-id", "alice",
        "--worker-id", "worker",
    ]) == 0
    output, error = read(capsys)
    assert error is None
    assert output == {
        "mutation_performed": False,
        "source_text_returned": False,
        "status": "idle",
    }


def test_recovery_error_exposes_only_generic_state(monkeypatch, capsys):
    _journal, _dependencies = install(monkeypatch)

    def fail(*args, **kwargs):
        raise SignedPublicationRetirementRecoveryError(
            "private failure details",
            retirement_id="a" * 64,
            state="failed",
            phase="pointer_safe",
        )

    monkeypatch.setattr(cli, "execute_signed_publication_retirement", fail)
    assert cli.main([
        "execute", "a" * 64,
        "--worker-id", "worker",
    ]) == 1
    output, error = read(capsys)
    assert output is None
    assert error["error"] == "retirement_failed"
    assert error["phase"] == "pointer_safe"
    rendered = json.dumps(error).lower()
    assert "private failure details" not in rendered
    assert "source" in rendered


def test_status_list_retry_and_cancel_use_read_only_or_exact_boundaries(
    monkeypatch, capsys
):
    journal, _dependencies = install(monkeypatch)
    assert cli.main(["status", "a" * 64]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["retirement_mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["mutation_performed"] is False
    assert journal.list_kwargs["owner_id"] == "alice"

    assert cli.main([
        "retry", "a" * 64,
        "--owner-id", "alice",
        "--confirm-retirement-id", "a" * 64,
    ]) == 0
    read(capsys)
    assert journal.retry_kwargs["confirm_retirement_id"] == "a" * 64

    assert cli.main([
        "cancel", "a" * 64,
        "--owner-id", "alice",
        "--confirm-retirement-id", "a" * 64,
    ]) == 0
    read(capsys)
    assert journal.cancel_kwargs["confirm_retirement_id"] == "a" * 64
