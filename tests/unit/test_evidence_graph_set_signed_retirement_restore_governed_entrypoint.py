from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import evidence_graph_set_signed_retirement_restore_governed as script


def argv():
    return [
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


def test_governed_entrypoint_verifies_custody_before_cli(monkeypatch):
    observed = []
    receipt = SimpleNamespace(target_path_digest="1" * 64)
    monkeypatch.setattr(
        script,
        "verify_pre_restore_backup_receipt",
        lambda **kwargs: observed.append(("verify", kwargs)) or receipt,
    )
    monkeypatch.setattr(
        script,
        "target_path_digest",
        lambda value: observed.append(("target", value)) or "1" * 64,
    )
    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        lambda **kwargs: observed.append(("store", kwargs)) or object(),
    )
    monkeypatch.setattr(
        script,
        "_main",
        lambda values: observed.append(("main", tuple(values))) or 0,
    )

    values = argv()
    assert script.main(values) == 0
    assert observed == [
        (
            "verify",
            {
                "receipt_path": "pre.json",
                "backup_path": "backup.sqlite3",
            },
        ),
        ("target", "target.sqlite3"),
        ("store", {"target_db_path": "target.sqlite3"}),
        ("main", tuple(values)),
    ]


def test_governed_entrypoint_refuses_target_mismatch_before_store_or_cli(
    monkeypatch,
):
    monkeypatch.setattr(
        script,
        "verify_pre_restore_backup_receipt",
        lambda **kwargs: SimpleNamespace(target_path_digest="1" * 64),
    )
    monkeypatch.setattr(script, "target_path_digest", lambda value: "2" * 64)
    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("store must not open after mismatch")
        ),
    )
    monkeypatch.setattr(
        script,
        "_main",
        lambda values: (_ for _ in ()).throw(
            AssertionError("CLI must not run after mismatch")
        ),
    )
    with pytest.raises(RuntimeError, match="differs from target path"):
        script.main(argv())


def test_governed_entrypoint_requires_all_custody_inputs(monkeypatch):
    monkeypatch.setattr(
        script,
        "_main",
        lambda values: (_ for _ in ()).throw(
            AssertionError("CLI must not run with missing custody input")
        ),
    )
    values = argv()
    del values[values.index("--backup") : values.index("--backup") + 2]
    with pytest.raises(ValueError, match="receipt, and backup are required"):
        script.main(values)


def test_governed_entrypoint_keeps_read_commands_dependency_light(monkeypatch):
    monkeypatch.setattr(
        script,
        "verify_pre_restore_backup_receipt",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("status must not verify custody files")
        ),
    )
    monkeypatch.setattr(
        script,
        "get_signed_retirement_restore_custody_store",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("status must not initialize custody")
        ),
    )
    monkeypatch.setattr(script, "_main", lambda values: 0)
    assert script.main(["status", "a" * 64]) == 0
