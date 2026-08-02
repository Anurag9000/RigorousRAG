from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_manifest import (
    SignedRetirementRestoreCustodyStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_operations import (
    audit_restore_custody_operations,
    plan_restore_custody_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_readonly import (
    ReadOnlySignedRetirementRestoreCustodyStore,
)


def manifest(
    digit: str,
    *,
    state: str = "pre_bound",
    target_digit: str = "a",
    pre_bound_at: float = 10.0,
    post_bound_at: float | None = None,
):
    return SimpleNamespace(
        custody_id=digit * 64,
        owner_id="alice",
        restore_id=digit * 64,
        snapshot_digest="d" * 64,
        target_path_digest=target_digit * 64,
        pre_receipt_digest="e" * 64,
        backup_sha256="f" * 64,
        backup_size_bytes=123,
        state=state,
        pre_bound_actor_id="actor",
        pre_bound_method="process_environment",
        pre_bound_binding_digest="1" * 64,
        pre_bound_at=pre_bound_at,
        post_receipt_digest=None if state == "pre_bound" else "2" * 64,
        target_verification_digest=None if state == "pre_bound" else "3" * 64,
        post_bound_actor_id=None if state == "pre_bound" else "actor",
        post_bound_method=None if state == "pre_bound" else "process_environment",
        post_bound_binding_digest=None if state == "pre_bound" else "4" * 64,
        post_bound_at=post_bound_at if state == "post_bound" else None,
        manifest_digest="5" * 64,
    )


class Store:
    def __init__(self, values):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        values = self.values
        if kwargs.get("state") is not None:
            values = tuple(value for value in values if value.state == kwargs["state"])
        return values[: kwargs["limit"]]


def test_custody_audit_classifies_filters_and_revalidates_digest():
    pending = manifest("6", pre_bound_at=10.0)
    completed = manifest(
        "7",
        state="post_bound",
        pre_bound_at=10.0,
        post_bound_at=20.0,
    )
    store = Store((completed, pending))

    report = audit_restore_custody_operations(
        owner_id="alice",
        store=store,
        now=30.0,
        limit=10,
    )

    assert report.item_count == 2
    assert report.classification_counts == {
        "post_bound_complete": 1,
        "pre_bound_pending_post": 1,
    }
    assert {item.age_seconds for item in report.items} == {10.0, 20.0}
    assert all(not hasattr(item, "pre_bound_actor_id") for item in report.items)
    assert report.raw_path_returned is False
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="0" * 64)

    filtered = audit_restore_custody_operations(
        owner_id="alice",
        store=store,
        target_path_digest="a" * 64,
        state="post_bound",
        now=30.0,
        limit=10,
    )
    assert filtered.item_count == 1
    assert store.calls[-1]["state"] == "post_bound"


def test_custody_audit_refuses_truncated_and_duplicate_results():
    value = manifest("6")
    with pytest.raises(RuntimeError, match="bounded"):
        audit_restore_custody_operations(
            owner_id="alice",
            store=Store((value,)),
            now=30.0,
            limit=1,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_restore_custody_operations(
            owner_id="alice",
            store=Store((value, value)),
            now=30.0,
            limit=10,
        )


def test_custody_retention_protects_incomplete_durable_holds_and_latest():
    older = manifest(
        "6",
        state="post_bound",
        target_digit="a",
        pre_bound_at=1.0,
        post_bound_at=2.0,
    )
    newer = manifest(
        "7",
        state="post_bound",
        target_digit="a",
        pre_bound_at=2.0,
        post_bound_at=3.0,
    )
    held = manifest(
        "8",
        state="post_bound",
        target_digit="b",
        pre_bound_at=1.0,
        post_bound_at=2.0,
    )
    pending = manifest("9", target_digit="c", pre_bound_at=1.0)
    store = Store((older, newer, held, pending))

    default = plan_restore_custody_retention(
        owner_id="alice",
        store=store,
        now=1000.0,
        minimum_age_seconds=10.0,
        held_restore_ids=(held.restore_id,),
        limit=10,
    )
    reasons = {item.custody_id: item.reason for item in default.items}
    assert default.candidate_count == 0
    assert reasons[pending.custody_id] == "pre_bound_incomplete_never_candidate"
    assert reasons[held.custody_id] == "legal_hold"
    assert reasons[newer.custody_id] == "latest_post_bound_for_target"
    assert reasons[older.custody_id] == "post_bound_retained_by_default"

    enabled = plan_restore_custody_retention(
        owner_id="alice",
        store=store,
        now=1000.0,
        minimum_age_seconds=10.0,
        include_post_bound=True,
        held_restore_ids=(held.restore_id,),
        limit=10,
    )
    assert [
        item.custody_id for item in enabled.items if item.retention_candidate
    ] == [older.custody_id]
    assert enabled.deletion_performed is False
    assert enabled.mutation_performed is False
    with pytest.raises(ValueError, match="plan_digest"):
        replace(enabled, plan_digest="0" * 64)


def test_read_only_custody_store_requires_schema_and_rejects_writes(tmp_path):
    uninitialized = tmp_path / "uninitialized.sqlite3"
    uninitialized.write_bytes(b"")
    with pytest.raises(RuntimeError, match="not initialized"):
        ReadOnlySignedRetirementRestoreCustodyStore(uninitialized)

    writable = SignedRetirementRestoreCustodyStore(tmp_path / "custody.sqlite3")
    read_only = ReadOnlySignedRetirementRestoreCustodyStore(writable.path)
    assert read_only.list(owner_id="alice", limit=10) == ()
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_set_signed_restore_custody"
            )


def test_custody_operations_cli_uses_durable_holds_and_has_no_mutation(
    monkeypatch,
    capsys,
):
    older = manifest(
        "6",
        state="post_bound",
        target_digit="a",
        pre_bound_at=1.0,
        post_bound_at=2.0,
    )
    newer = manifest(
        "7",
        state="post_bound",
        target_digit="a",
        pre_bound_at=2.0,
        post_bound_at=3.0,
    )
    store = Store((older, newer))

    monkeypatch.setattr(
        cli,
        "ReadOnlySignedRetirementRestoreCustodyStore",
        lambda _path: store,
    )

    class Holds:
        def __init__(self, path):
            assert path == "holds.sqlite3"

        def active_restore_ids(self, **kwargs):
            assert kwargs["owner_id"] == "alice"
            return frozenset({older.restore_id})

    monkeypatch.setattr(cli, "ReadOnlySignedRetirementRestoreHoldStore", Holds)

    assert cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--custody-db-path",
            "custody.sqlite3",
            "--durable-hold-db-path",
            "holds.sqlite3",
            "--minimum-age-seconds",
            "1",
            "--include-post-bound",
            "--limit",
            "10",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["durable_restore_hold_count"] == 1
    assert payload["candidate_count"] == 0
    assert payload["custody_store_mutation_performed"] is False
    assert payload["hold_store_mutation_performed"] is False
    assert payload["deletion_performed"] is False
    assert payload["raw_path_returned"] is False

    with pytest.raises(SystemExit):
        cli.main(["delete"])
