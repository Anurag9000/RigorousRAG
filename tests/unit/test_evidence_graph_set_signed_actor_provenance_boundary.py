from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_actor_provenance_boundary as boundary


def test_reconcile_one_captures_one_finite_timestamp(monkeypatch):
    observed = {}
    journal = SimpleNamespace(
        next_claimable_id=lambda **kwargs: observed.setdefault("claim", kwargs) or "9" * 64
    )
    marker = object()

    def execute(operation_id, **kwargs):
        observed["operation_id"] = operation_id
        observed["execute"] = kwargs
        return marker

    monkeypatch.setattr(boundary, "execute_signed_actor_publication_attempt", execute)
    result = boundary.execute_next_signed_actor_publication_attempt(
        owner_id="alice",
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        ledger=object(),
        authorization_store=object(),
        actor_use_store=object(),
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=12.0,
    )

    assert result is marker
    assert observed["claim"] == {"owner_id": "alice", "now": 12.0}
    assert observed["operation_id"] == "9" * 64
    assert observed["execute"]["now"] == 12.0


def test_reconcile_one_returns_idle_without_execution(monkeypatch):
    journal = SimpleNamespace(next_claimable_id=lambda **kwargs: None)
    monkeypatch.setattr(
        boundary,
        "execute_signed_actor_publication_attempt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("execution should not run")
        ),
    )

    assert boundary.execute_next_signed_actor_publication_attempt(
        owner_id="alice",
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        ledger=object(),
        authorization_store=object(),
        actor_use_store=object(),
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=1.0,
    ) is None


def test_reconcile_one_rejects_nonfinite_time():
    journal = SimpleNamespace(next_claimable_id=lambda **kwargs: None)
    with pytest.raises(ValueError, match="finite"):
        boundary.execute_next_signed_actor_publication_attempt(
            owner_id="alice",
            worker_id="worker",
            lease_seconds=30,
            journal=journal,
            ledger=object(),
            authorization_store=object(),
            actor_use_store=object(),
            set_store=object(),
            generations=object(),
            graphs=object(),
            now=float("nan"),
        )
