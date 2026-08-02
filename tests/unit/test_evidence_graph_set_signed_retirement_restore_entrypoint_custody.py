from __future__ import annotations

import pytest

from scripts import evidence_graph_set_signed_retirement_restore_execute as script


def test_public_restore_entrypoint_guards_mutating_target_before_cli(monkeypatch):
    observed = []

    def custody(**kwargs):
        observed.append(("custody", kwargs))
        return object()

    def main(argv):
        observed.append(("main", tuple(argv)))
        return 0

    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        custody,
    )
    monkeypatch.setattr(script, "_main", main)

    argv = [
        "execute",
        "a" * 64,
        "--snapshot",
        "snapshot.json",
        "--target-db-path",
        "target.sqlite3",
        "--pre-receipt",
        "pre.json",
        "--backup",
        "backup.sqlite3",
        "--worker-id",
        "worker",
    ]
    assert script.main(argv) == 0
    assert observed[0] == (
        "custody",
        {"target_db_path": "target.sqlite3"},
    )
    assert observed[1] == ("main", tuple(argv))


def test_public_restore_entrypoint_does_not_initialize_custody_for_reads(
    monkeypatch,
):
    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("read-only commands must not initialize custody")
        ),
    )
    monkeypatch.setattr(script, "_main", lambda argv: 0)
    assert script.main(["status", "a" * 64]) == 0


def test_public_restore_entrypoint_alias_refusal_stops_cli(monkeypatch):
    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("alias")),
    )
    monkeypatch.setattr(
        script,
        "_main",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("CLI must not run after alias refusal")
        ),
    )
    with pytest.raises(RuntimeError, match="alias"):
        script.main(
            [
                "seed",
                "--snapshot",
                "snapshot.json",
                "--target-db-path",
                "target.sqlite3",
                "--confirm-snapshot-digest",
                "1" * 64,
                "--pre-receipt",
                "pre.json",
                "--backup",
                "backup.sqlite3",
            ]
        )
