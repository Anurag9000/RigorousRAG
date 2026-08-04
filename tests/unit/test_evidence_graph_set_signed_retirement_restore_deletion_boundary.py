from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_deletion_boundary as boundary
from tools.evidence_graph_relation_actor import ReviewActorBinding


def actor():
    return ReviewActorBinding.create(
        actor_id="operator-1",
        binding_method="process_environment",
        loaded_at=1.0,
    )


class Journal:
    def __init__(self, value):
        self.value = value

    def get(self, restore_id):
        return self.value

    def list(self, **kwargs):
        return (self.value,)


class Holds:
    def active_restore_ids(self, **kwargs):
        return frozenset()


def restore():
    return SimpleNamespace(
        restore_id="1" * 64,
        owner_id="alice",
        snapshot_digest="2" * 64,
        target_path_digest="3" * 64,
        state="cancelled",
        phase="planned",
        completed_at=1.0,
    )


def plan(value, *, candidate):
    item = SimpleNamespace(
        restore_id=value.restore_id,
        snapshot_digest=value.snapshot_digest,
        target_path_digest=value.target_path_digest,
        retention_candidate=candidate,
    )
    return SimpleNamespace(plan_digest="4" * 64, items=(item,))


def test_authorization_requires_current_candidate(tmp_path, monkeypatch):
    value = restore()
    store = (
        boundary.GovernedSignedRetirementRestoreDeletionAuthorizationStore(
            tmp_path / "authorizations.sqlite3"
        )
    )
    monkeypatch.setattr(
        boundary,
        "plan_signed_retirement_restore_retention",
        lambda **kwargs: plan(value, candidate=False),
    )

    with pytest.raises(RuntimeError, match="no longer a current"):
        store.authorize(
            owner_id="alice",
            restore_id=value.restore_id,
            plan_digest="4" * 64,
            plan_generated_at=10.0,
            authorization_key="ticket",
            actor=actor(),
            restore_journal=Journal(value),
            hold_store=Holds(),
            minimum_age_seconds=1.0,
            retain_latest_per_target=1,
            include_completed=True,
            now=20.0,
        )


def test_authorization_rejects_future_plan_timestamp(tmp_path, monkeypatch):
    value = restore()
    store = (
        boundary.GovernedSignedRetirementRestoreDeletionAuthorizationStore(
            tmp_path / "authorizations.sqlite3"
        )
    )
    monkeypatch.setattr(
        boundary,
        "plan_signed_retirement_restore_retention",
        lambda **kwargs: plan(value, candidate=True),
    )

    with pytest.raises(ValueError, match="future"):
        store.authorize(
            owner_id="alice",
            restore_id=value.restore_id,
            plan_digest="4" * 64,
            plan_generated_at=21.0,
            authorization_key="ticket",
            actor=actor(),
            restore_journal=Journal(value),
            hold_store=Holds(),
            minimum_age_seconds=1.0,
            retain_latest_per_target=1,
            include_completed=True,
            now=20.0,
        )
