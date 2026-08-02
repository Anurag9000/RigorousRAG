from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement as retirement
from tools import evidence_graph_set_signed_retirement_cli as cli

OPERATION = "1" * 64
COMMON_CANDIDATE = "2" * 64
SIGNED_CANDIDATE = "3" * 64
SIGNED_DIGEST = "4" * 64
AUTHORITY_DIGEST = "5" * 64


def attempt(
    *,
    signed: bool,
    state: str,
    phase: str,
    lease_expires_at=None,
    candidate_id=None,
    candidate_digest=None,
):
    return SimpleNamespace(
        operation_id=OPERATION,
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=("6" * 64,),
        expected_current_set_id=None,
        state=state,
        phase=phase,
        lease_expires_at=lease_expires_at,
        candidate_graph_set_id=candidate_id,
        candidate_graph_set_digest=candidate_digest,
    )


class Journal:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get(self, operation_id):
        self.calls.append(operation_id)
        assert operation_id == OPERATION
        return self.value


class SetStore:
    def __init__(self, *, current_id):
        self.current_id = current_id
        self.candidate = SimpleNamespace(
            graph_set_id=SIGNED_CANDIDATE,
            graph_set_digest=SIGNED_DIGEST,
            owner_id="alice",
            graph_set_key="review",
        )

    def get(self, *, owner_id, graph_set_id):
        assert owner_id == "alice"
        assert graph_set_id == SIGNED_CANDIDATE
        return self.candidate

    def current(self, *, owner_id, graph_set_key):
        assert owner_id == "alice"
        assert graph_set_key == "review"
        if self.current_id is None:
            return None
        return SimpleNamespace(graph_set_id=self.current_id)


def values(*, current_id, lease=9.0, signed_state="completed", signed_phase="verified"):
    common = attempt(
        signed=False,
        state="running",
        phase="candidate_stored",
        lease_expires_at=lease,
        candidate_id=COMMON_CANDIDATE,
        candidate_digest="7" * 64,
    )
    signed = attempt(
        signed=True,
        state=signed_state,
        phase=signed_phase,
        candidate_id=(SIGNED_CANDIDATE if signed_state == "completed" else None),
        candidate_digest=(SIGNED_DIGEST if signed_state == "completed" else None),
    )
    return Journal(common), Journal(signed), SetStore(current_id=current_id)


def install_authority(monkeypatch, *, current=True):
    monkeypatch.setattr(
        retirement,
        "assess_graph_set_authority",
        lambda value, **kwargs: SimpleNamespace(
            authoritative_current=current,
            authority_digest=AUTHORITY_DIGEST,
        ),
    )


def preflight(monkeypatch, *, current_id, lease=9.0, authority=True):
    install_authority(monkeypatch, current=authority)
    common, signed, store = values(current_id=current_id, lease=lease)
    return retirement.preflight_expired_signed_publication_duplicate_retirement(
        owner_id="alice",
        operation_id=OPERATION,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
    )


def test_preflight_allows_journal_only_retirement(monkeypatch):
    report = preflight(monkeypatch, current_id=SIGNED_CANDIDATE)

    assert report.eligible is True
    assert report.disposition == "retire_expired_journal_only"
    assert report.current_pointer_set_id == SIGNED_CANDIDATE
    assert report.signed_candidate_authoritative is True
    assert report.mutation_performed is False
    assert report.source_text_returned is False


def test_preflight_requires_signed_pointer_restoration(monkeypatch):
    report = preflight(monkeypatch, current_id=COMMON_CANDIDATE)

    assert report.eligible is True
    assert report.disposition == "restore_signed_pointer_then_retire"
    assert report.authorization_candidate_set_id == COMMON_CANDIDATE


def test_preflight_refuses_external_pointer_and_stale_signed_candidate(monkeypatch):
    external = preflight(monkeypatch, current_id="8" * 64)
    assert external.eligible is False
    assert external.disposition == "external_pointer_change_refusal"

    stale = preflight(
        monkeypatch,
        current_id=SIGNED_CANDIDATE,
        authority=False,
    )
    assert stale.eligible is False
    assert stale.disposition == "signed_candidate_not_authoritative"


def test_preflight_waits_for_active_lease_without_authority_check(monkeypatch):
    monkeypatch.setattr(
        retirement,
        "assess_graph_set_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("authority should not be checked")
        ),
    )
    common, signed, store = values(current_id=SIGNED_CANDIDATE, lease=20.0)

    report = retirement.preflight_expired_signed_publication_duplicate_retirement(
        owner_id="alice",
        operation_id=OPERATION,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
    )

    assert report.eligible is False
    assert report.lease_expired is False
    assert report.disposition == "wait_for_authorization_only_lease"


def test_preflight_requires_completed_signed_attempt(monkeypatch):
    install_authority(monkeypatch)
    common, signed, store = values(
        current_id=COMMON_CANDIDATE,
        signed_state="planned",
        signed_phase="planned",
    )

    report = retirement.preflight_expired_signed_publication_duplicate_retirement(
        owner_id="alice",
        operation_id=OPERATION,
        authorization_journal=common,
        signed_journal=signed,
        set_store=store,
        generations=object(),
        graphs=object(),
        now=10.0,
    )

    assert report.eligible is False
    assert report.disposition == "signed_attempt_not_completed"


def test_preflight_report_revalidates_digest(monkeypatch):
    report = preflight(monkeypatch, current_id=SIGNED_CANDIDATE)
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="f" * 64)


def test_retirement_cli_is_read_only_and_text_free(monkeypatch, capsys):
    report = preflight(monkeypatch, current_id=SIGNED_CANDIDATE)
    monkeypatch.setattr(
        cli,
        "preflight_expired_signed_publication_duplicate_retirement",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(
        cli, "get_evidence_graph_set_publication_journal", lambda: "common"
    )
    monkeypatch.setattr(
        cli, "get_evidence_graph_set_signed_publication_journal", lambda: "signed"
    )
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: "sets")
    monkeypatch.setattr(cli, "get_generation_store", lambda: "generations")
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: "graphs")

    assert cli.main(["preflight", OPERATION, "--owner-id", "alice"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["eligible"] is True
    assert payload["automatic_retirement_performed"] is False
    assert payload["pointer_mutation_performed"] is False
    assert payload["journal_mutation_performed"] is False
    assert payload["source_text_returned"] is False
    assert "private text" not in json.dumps(payload).lower()
