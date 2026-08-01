import json
from types import SimpleNamespace

import tools.index_reconciliation_cli as cli
from tools.three_store_coordinator import ReconciliationReport


def report(**overrides):
    values = {
        "owner_id": "alice",
        "healthy": ("healthy",),
        "vector_only": (),
        "sparse_only": (),
        "store_pair_without_manifest": (),
        "manifest_without_store_pair": (),
        "deleted_but_present": (),
        "metadata_mismatch": (),
        "inspection_failed": (),
    }
    values.update(overrides)
    return ReconciliationReport(**values)


def install(monkeypatch, reports):
    coordinator = SimpleNamespace()
    coordinator.reconcile_owner = lambda **_kwargs: reports.pop(0)
    monkeypatch.setattr(cli, "get_rag_layer", lambda: object())
    monkeypatch.setattr(
        cli,
        "get_authoritative_index_coordinator",
        lambda **_kwargs: coordinator,
    )
    return coordinator


def test_scan_emits_bounded_public_categories(monkeypatch, capsys):
    install(
        monkeypatch,
        [report(vector_only=("doc-1",), metadata_mismatch=("doc-2",))],
    )
    assert cli.main(["scan", "--owner-id", "alice"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "scan"
    assert payload["report"]["vector_only"] == ["doc-1"]
    assert payload["report"]["metadata_mismatch"] == ["doc-2"]
    assert "path" not in json.dumps(payload).lower()


def test_plan_is_deterministic_and_reports_truncation(monkeypatch, capsys):
    install(
        monkeypatch,
        [
            report(
                vector_only=("doc-2",),
                sparse_only=("doc-1",),
                deleted_but_present=("doc-3",),
            )
        ],
    )
    assert cli.main(
        ["plan", "--owner-id", "alice", "--maximum", "2"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["truncated"] is True
    assert len(payload["actions"]) == 2
    assert payload["actions"] == sorted(
        payload["actions"],
        key=lambda item: (item["doc_id"], item["action"]),
    )


def test_repair_requires_exact_confirmation_before_mutation(monkeypatch, capsys):
    coordinator = install(
        monkeypatch,
        [report(deleted_but_present=("doc-1",))],
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "apply_deleted_residue_repairs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ("doc-1",),
    )
    assert cli.main(
        [
            "repair-deleted-residue",
            "--owner-id",
            "alice",
            "--confirmation",
            "wrong",
        ]
    ) == 0
    assert calls[0][0][0] is coordinator
    assert calls[0][1]["confirmation"] == "wrong"


def test_real_confirmation_failure_is_generic_and_does_not_leak(monkeypatch, capsys):
    install(monkeypatch, [report(deleted_but_present=("doc-1",))])
    assert cli.main(
        [
            "repair-deleted-residue",
            "--owner-id",
            "alice",
            "--confirmation",
            "wrong",
        ]
    ) == 1
    captured = capsys.readouterr()
    assert "could not be completed safely" in captured.err
    assert "DELETE_DELETED_GENERATION_RESIDUE" not in captured.err
    assert captured.out == ""


def test_successful_repair_emits_after_state(monkeypatch, capsys):
    coordinator = install(
        monkeypatch,
        [
            report(deleted_but_present=("doc-1",)),
            report(deleted_but_present=()),
        ],
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "apply_deleted_residue_repairs",
        lambda selected, current, **kwargs: (
            calls.append((selected, current, kwargs)) or ("doc-1",)
        ),
    )
    assert cli.main(
        [
            "repair-deleted-residue",
            "--owner-id",
            "alice",
            "--confirmation",
            "DELETE_DELETED_GENERATION_RESIDUE",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repaired"] == ["doc-1"]
    assert payload["remaining"]["clean"] is False
    assert calls[0][0] is coordinator


def test_owner_and_limit_validation_precede_runtime_initialization(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_rag_layer",
        lambda: (_ for _ in ()).throw(AssertionError("backend initialized")),
    )
    assert cli.main(["scan", "--owner-id", "../alice"]) == 1
    assert cli.main(
        ["scan", "--owner-id", "alice", "--maximum", "0"]
    ) == 1
    assert "safely" in capsys.readouterr().err


def test_runtime_failure_is_generic(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_rag_layer",
        lambda: (_ for _ in ()).throw(
            RuntimeError("failed at /private/source.sqlite3")
        ),
    )
    assert cli.main(["scan", "--owner-id", "alice"]) == 1
    error = capsys.readouterr().err
    assert "unavailable" in error
    assert "/private" not in error
