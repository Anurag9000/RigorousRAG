from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_authorizations as mod,
)
from tools import evidence_graph_set_signed_retirement_restore_deletion_cli as cli
from tools import evidence_graph_set_signed_retirement_restore_deletion_runtime as runtime
from tools.evidence_graph_relation_actor import ReviewActorBinding


def actor(actor_id: str = "operator-1", *, loaded_at: float = 1.0):
    return ReviewActorBinding.create(
        actor_id=actor_id,
        binding_method="process_environment",
        loaded_at=loaded_at,
    )


class Journal:
    def __init__(self, values):
        self.values = {value.restore_id: value for value in values}

    def get(self, restore_id):
        if restore_id not in self.values:
            raise KeyError(restore_id)
        return self.values[restore_id]

    def list(self, **kwargs):
        return tuple(self.values.values())[: kwargs["limit"]]


class Holds:
    def __init__(self, restore_ids=()):
        self.restore_ids = frozenset(restore_ids)

    def active_restore_ids(self, **kwargs):
        return self.restore_ids


def restore(digit="1", *, target="2", snapshot="3"):
    return SimpleNamespace(
        restore_id=digit * 64,
        owner_id="alice",
        snapshot_digest=snapshot * 64,
        target_path_digest=target * 64,
        state="cancelled",
        phase="planned",
        completed_at=1.0,
    )


def plan(value, *, digest="4" * 64, candidate=True):
    item = SimpleNamespace(
        restore_id=value.restore_id,
        snapshot_digest=value.snapshot_digest,
        target_path_digest=value.target_path_digest,
        retention_candidate=candidate,
    )
    return SimpleNamespace(plan_digest=digest, items=(item,))


def install_plan(monkeypatch, value, *, digest="4" * 64, candidate=True):
    monkeypatch.setattr(
        mod,
        "plan_signed_retirement_restore_retention",
        lambda **kwargs: plan(value, digest=digest, candidate=candidate),
    )


def authorize(store, monkeypatch, *, value=None, actor_value=None, now=20.0):
    value = restore() if value is None else value
    install_plan(monkeypatch, value)
    return store.authorize(
        owner_id="alice",
        restore_id=value.restore_id,
        plan_digest="4" * 64,
        plan_generated_at=10.0,
        authorization_key="ticket-1",
        actor=actor() if actor_value is None else actor_value,
        restore_journal=Journal((value,)),
        hold_store=Holds(),
        minimum_age_seconds=1.0,
        retain_latest_per_target=1,
        include_completed=True,
        expires_in_seconds=100.0,
        now=now,
    )


def test_authorization_identity_idempotence_and_actor_collision(
    tmp_path, monkeypatch
):
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    first = authorize(store, monkeypatch)

    assert first.status == "authorized"
    assert first.authorization_id == (
        mod.deterministic_restore_deletion_authorization_id(
            owner_id="alice",
            restore_id="1" * 64,
            snapshot_digest="3" * 64,
            target_path_digest="2" * 64,
            plan_digest="4" * 64,
            policy_digest=first.policy_digest,
            authorization_key="ticket-1",
        )
    )
    assert store.get(first.authorization_id) == first
    assert authorize(store, monkeypatch, now=99.0) == first

    with pytest.raises(RuntimeError, match="collision"):
        authorize(
            store,
            monkeypatch,
            actor_value=actor("operator-2"),
        )


def test_active_hold_plan_digest_and_candidate_checks_fail_closed(
    tmp_path, monkeypatch
):
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    value = restore()
    install_plan(monkeypatch, value)
    with pytest.raises(RuntimeError, match="legal hold"):
        store.authorize(
            owner_id="alice",
            restore_id=value.restore_id,
            plan_digest="4" * 64,
            plan_generated_at=10.0,
            authorization_key="ticket",
            actor=actor(),
            restore_journal=Journal((value,)),
            hold_store=Holds((value.restore_id,)),
            minimum_age_seconds=1.0,
            retain_latest_per_target=1,
            include_completed=True,
            now=20.0,
        )

    install_plan(monkeypatch, value, digest="5" * 64)
    with pytest.raises(RuntimeError, match="plan digest"):
        store.authorize(
            owner_id="alice",
            restore_id=value.restore_id,
            plan_digest="4" * 64,
            plan_generated_at=10.0,
            authorization_key="ticket",
            actor=actor(),
            restore_journal=Journal((value,)),
            hold_store=Holds(),
            minimum_age_seconds=1.0,
            retain_latest_per_target=1,
            include_completed=True,
            now=20.0,
        )

    install_plan(monkeypatch, value, candidate=False)
    with pytest.raises(RuntimeError, match="not an authorized retention candidate"):
        store.authorize(
            owner_id="alice",
            restore_id=value.restore_id,
            plan_digest="4" * 64,
            plan_generated_at=10.0,
            authorization_key="ticket",
            actor=actor(),
            restore_journal=Journal((value,)),
            hold_store=Holds(),
            minimum_age_seconds=1.0,
            retain_latest_per_target=1,
            include_completed=True,
            now=20.0,
        )


def test_revocation_is_exact_monotonic_and_integrity_backed(
    tmp_path, monkeypatch
):
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    value = authorize(store, monkeypatch)
    with pytest.raises(ValueError, match="confirmation"):
        store.revoke(
            value.authorization_id,
            owner_id="alice",
            confirm_authorization_id="f" * 64,
            actor=actor("operator-2"),
            now=30.0,
        )

    revoked = store.revoke(
        value.authorization_id,
        owner_id="alice",
        confirm_authorization_id=value.authorization_id,
        actor=actor("operator-2"),
        now=30.0,
    )
    assert revoked.status == "revoked"
    assert revoked.revoked_actor_id == "operator-2"
    assert store.revoke(
        value.authorization_id,
        owner_id="alice",
        confirm_authorization_id=value.authorization_id,
        actor=actor("operator-1"),
        now=40.0,
    ) == revoked

    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE signed_retirement_restore_deletion_authorizations "
            "SET expires_at=? WHERE authorization_id=?",
            (999.0, value.authorization_id),
        )
    with pytest.raises(RuntimeError, match="integrity differs"):
        store.get(value.authorization_id)


def test_preflight_rechecks_current_candidate_hold_expiry_and_scope(
    tmp_path, monkeypatch
):
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    value = restore()
    authorization = authorize(store, monkeypatch, value=value)
    install_plan(monkeypatch, value, digest="9" * 64)

    current = mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal((value,)),
        hold_store=Holds(),
        now=30.0,
    )
    assert current.disposition == "authorized_candidate_current"
    assert current.eligible_for_future_deletion_executor is True

    held = mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal((value,)),
        hold_store=Holds((value.restore_id,)),
        now=30.0,
    )
    assert held.disposition == "durable_legal_hold_active"

    expired = mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal((value,)),
        hold_store=Holds(),
        now=120.0,
    )
    assert expired.disposition == "authorization_expired"

    assert mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal(()),
        hold_store=Holds(),
        now=30.0,
    ).disposition == "restore_missing"

    changed = restore(snapshot="4")
    assert mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal((changed,)),
        hold_store=Holds(),
        now=30.0,
    ).disposition == "restore_scope_changed"


def test_preflight_candidate_drift_and_report_tampering_fail_closed(
    tmp_path, monkeypatch
):
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    value = restore()
    authorization = authorize(store, monkeypatch, value=value)
    install_plan(monkeypatch, value, digest="9" * 64, candidate=False)
    report = mod.preflight_signed_retirement_restore_deletion(
        authorization=authorization,
        restore_journal=Journal((value,)),
        hold_store=Holds(),
        now=30.0,
    )
    assert report.disposition == "no_longer_retention_candidate"
    assert report.deletion_performed is False

    with pytest.raises(ValueError, match="report_digest"):
        mod.SignedRetirementRestoreDeletionPreflight(
            **{
                **report.__dict__,
                "report_digest": "f" * 64,
            }
        )


def test_database_identity_and_runtime_aliases_fail_closed(
    tmp_path, monkeypatch
):
    path = tmp_path / "authorizations.sqlite3"
    store = mod.SignedRetirementRestoreDeletionAuthorizationStore(path)
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.list(owner_id="alice")

    runtime.clear_signed_retirement_restore_deletion_authorization_store_cache()
    protected = tmp_path / "restore.sqlite3"
    protected.write_bytes(b"database")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_signed_retirement_restore_deletion_authorization_store(
            protected
        )


def test_cli_confirmation_read_boundaries_and_no_delete_command(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor") or actor(),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_deletion_authorization_store",
        lambda: calls.append("store") or object(),
    )
    assert cli.main(
        [
            "authorize",
            "1" * 64,
            "--owner-id",
            "alice",
            "--confirm-restore-id",
            "2" * 64,
            "--plan-digest",
            "3" * 64,
            "--plan-generated-at",
            "1",
            "--authorization-key",
            "ticket",
        ]
    ) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["delete", "1" * 64])

    value = SimpleNamespace(
        authorization_id="1" * 64,
        owner_id="alice",
        restore_id="2" * 64,
        snapshot_digest="3" * 64,
        target_path_digest="4" * 64,
        plan_digest="5" * 64,
        policy_digest="6" * 64,
        authorization_key="ticket",
        minimum_age_seconds=1.0,
        retain_latest_per_target=1,
        include_completed=False,
        status="authorized",
        authorized_actor_id="operator",
        authorized_binding_method="process_environment",
        authorized_binding_digest="7" * 64,
        authorized_at=1.0,
        expires_at=2.0,
        revoked_actor_id=None,
        revoked_binding_method=None,
        revoked_binding_digest=None,
        revoked_at=None,
        authorization_digest="8" * 64,
    )

    class Store:
        def get(self, authorization_id):
            return value

        def list(self, **kwargs):
            return (value,)

    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_deletion_authorization_store",
        lambda: Store(),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("read command loaded restore journal")
        ),
    )
    assert cli.main(["status", "1" * 64]) == 0
    assert json.loads(capsys.readouterr().out)["mutation_performed"] is False
    assert cli.main(["list", "--owner-id", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["deletion_performed"] is False
