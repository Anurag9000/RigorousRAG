from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools import evidence_graph_set_signed_retirement_reconcile as reconcile


def preflight(*, eligible=True, disposition="retire_expired_journal_only"):
    return SimpleNamespace(
        eligible=eligible,
        disposition=disposition,
        owner_id="alice",
        operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        report_digest="6" * 64,
    )


def test_seed_binds_fresh_eligible_preflight_to_immutable_attempt(
    tmp_path, monkeypatch
):
    observed = {}

    def inspect(**kwargs):
        observed.update(kwargs)
        return preflight()

    monkeypatch.setattr(
        reconcile,
        "preflight_expired_signed_publication_duplicate_retirement",
        inspect,
    )
    signed = SimpleNamespace(
        get=lambda operation_id: SimpleNamespace(
            candidate_graph_set_digest="3" * 64
        )
    )
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")

    attempt, report = reconcile.seed_signed_publication_retirement(
        owner_id="alice",
        publication_operation_id="1" * 64,
        authorization_journal="authorization-journal",
        signed_journal=signed,
        retirement_journal=journal,
        set_store="sets",
        generations="generations",
        graphs="graphs",
        max_attempts=4,
        now=10.0,
    )

    assert report.eligible is True
    assert attempt.owner_id == "alice"
    assert attempt.publication_operation_id == "1" * 64
    assert attempt.signed_candidate_set_digest == "3" * 64
    assert attempt.authorization_candidate_set_id == "4" * 64
    assert attempt.max_attempts == 4
    assert journal.get(attempt.retirement_id) == attempt
    assert observed["now"] == 10.0


def test_seed_refuses_ineligible_or_incomplete_preflight(tmp_path, monkeypatch):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    monkeypatch.setattr(
        reconcile,
        "preflight_expired_signed_publication_duplicate_retirement",
        lambda **kwargs: preflight(
            eligible=False, disposition="external_pointer_change_refusal"
        ),
    )
    with pytest.raises(RuntimeError, match="not eligible"):
        reconcile.seed_signed_publication_retirement(
            owner_id="alice",
            publication_operation_id="1" * 64,
            authorization_journal=object(),
            signed_journal=object(),
            retirement_journal=journal,
            set_store=object(),
            generations=object(),
            graphs=object(),
            now=10.0,
        )

    monkeypatch.setattr(
        reconcile,
        "preflight_expired_signed_publication_duplicate_retirement",
        lambda **kwargs: SimpleNamespace(
            **{**vars(preflight()), "signed_authority_digest": None}
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        reconcile.seed_signed_publication_retirement(
            owner_id="alice",
            publication_operation_id="1" * 64,
            authorization_journal=object(),
            signed_journal=SimpleNamespace(
                get=lambda operation_id: SimpleNamespace(
                    candidate_graph_set_digest="3" * 64
                )
            ),
            retirement_journal=journal,
            set_store=object(),
            generations=object(),
            graphs=object(),
            now=10.0,
        )
