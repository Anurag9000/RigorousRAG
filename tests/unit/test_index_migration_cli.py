import json
from types import SimpleNamespace

import tools.index_migration_cli as cli
from tools.migration_types import MigrationCandidate, MigrationTask


def candidate(*, eligible=True, reason="ready"):
    return MigrationCandidate(
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=1,
        source_profile_fingerprint="a" * 64,
        target_profile_name="e5-base-v2",
        target_profile_fingerprint="b" * 64,
        retained_source=eligible,
        eligible=eligible,
        reason=reason,
    )


def task(*, owner="alice", state="planned"):
    return MigrationTask(
        task_id="c" * 64,
        owner_id=owner,
        doc_id="doc-1",
        source_sequence=1,
        source_profile_fingerprint="a" * 64,
        target_profile_name="e5-base-v2",
        target_profile_fingerprint="b" * 64,
        state=state,
        attempt=0,
        created_at=1.0,
        updated_at=1.0,
    )


def test_inventory_reports_eligibility_without_paths(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_candidates",
        lambda owner, target, limit: (
            candidate(),
            candidate(eligible=False, reason="retained_source_unavailable"),
        ),
    )
    assert cli.main(
        [
            "inventory",
            "--owner-id",
            "alice",
            "--target-profile",
            "e5-base-v2",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible"] == 1
    assert len(payload["candidates"]) == 2
    assert "source_path" not in json.dumps(payload)


def test_seed_persists_only_planner_eligible_tasks(monkeypatch, capsys):
    values = (candidate(), candidate(eligible=False, reason="deleted"))
    journal = SimpleNamespace(seed=lambda received: (task(),))
    monkeypatch.setattr(cli, "_candidates", lambda *_args: values)
    monkeypatch.setattr(cli, "get_migration_journal", lambda: journal)
    assert cli.main(
        [
            "seed",
            "--owner-id",
            "alice",
            "--target-profile",
            "e5-base-v2",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["seeded"]) == 1
    assert payload["ineligible"][0]["reason"] == "deleted"


def test_status_forwards_owner_state_and_limit(monkeypatch, capsys):
    calls = []
    journal = SimpleNamespace(
        list_tasks=lambda **kwargs: calls.append(kwargs) or (task(),)
    )
    monkeypatch.setattr(cli, "get_migration_journal", lambda: journal)
    assert cli.main(
        [
            "status",
            "--owner-id",
            "alice",
            "--state",
            "planned",
            "--limit",
            "5",
        ]
    ) == 0
    assert calls == [{"owner_id": "alice", "state": "planned", "limit": 5}]
    assert json.loads(capsys.readouterr().out)["tasks"][0]["state"] == "planned"


def test_cross_owner_cancel_is_refused_before_mutation(monkeypatch, capsys):
    calls = []
    journal = SimpleNamespace(
        get=lambda _task_id: task(owner="bob"),
        cancel=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(cli, "get_migration_journal", lambda: journal)
    assert cli.main(
        [
            "cancel",
            "--owner-id",
            "alice",
            "--task-id",
            "c" * 64,
        ]
    ) == 1
    assert calls == []
    assert "safely" in capsys.readouterr().err


def test_owned_cancel_mutates_exact_task(monkeypatch, capsys):
    current = task()
    calls = []
    journal = SimpleNamespace(
        get=lambda _task_id: current,
        cancel=lambda **kwargs: calls.append(kwargs) or current,
    )
    monkeypatch.setattr(cli, "get_migration_journal", lambda: journal)
    assert cli.main(
        [
            "cancel",
            "--owner-id",
            "alice",
            "--task-id",
            "c" * 64,
        ]
    ) == 0
    assert calls == [{"task_id": "c" * 64}]
    assert json.loads(capsys.readouterr().out)["task"]["owner_id"] == "alice"


def test_validation_precedes_backend_initialization(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_migration_journal",
        lambda: (_ for _ in ()).throw(AssertionError("journal initialized")),
    )
    assert cli.main(["status", "--owner-id", "../alice"]) == 1
    assert cli.main(
        ["status", "--owner-id", "alice", "--limit", "0"]
    ) == 1
    assert "safely" in capsys.readouterr().err


def test_backend_failure_is_generic_and_private_data_free(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_migration_journal",
        lambda: (_ for _ in ()).throw(
            RuntimeError("failed at /private/index_migrations.sqlite3")
        ),
    )
    assert cli.main(["status", "--owner-id", "alice"]) == 1
    error = capsys.readouterr().err
    assert "unavailable" in error
    assert "/private" not in error
