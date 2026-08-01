from __future__ import annotations

import json

from tools import lifecycle_cli
from tools.lifecycle_outbox import LifecycleOutbox, LifecycleReconcileResult

HASH = "a" * 64


def planned(outbox, operation_id="replace-1", owner_id="alice"):
    return outbox.plan_replace(
        operation_id=operation_id,
        owner_id=owner_id,
        doc_id="doc-1",
        content_sha256=HASH,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path="/private/alice/paper.pdf",
        retain_source=True,
        max_attempts=1,
        now=1.0,
    )


def parse_output(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_pending_output_is_owner_scoped_and_excludes_private_paths(
    tmp_path, monkeypatch, capsys
):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    planned(outbox)
    planned(outbox, "replace-2", "bob")
    monkeypatch.setattr(lifecycle_cli, "get_lifecycle_outbox", lambda: outbox)
    assert lifecycle_cli.main(["pending", "--owner-id", "alice"]) == 0
    output, error = parse_output(capsys)
    assert error is None
    assert output["count"] == 1
    assert output["operations"][0]["owner_id"] == "alice"
    assert "source_path" not in output["operations"][0]
    assert "/private" not in json.dumps(output)


def test_status_returns_public_summary_or_bounded_not_found(
    tmp_path, monkeypatch, capsys
):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    planned(outbox)
    monkeypatch.setattr(lifecycle_cli, "get_lifecycle_outbox", lambda: outbox)
    assert lifecycle_cli.main(["status", "replace-1"]) == 0
    output, _error = parse_output(capsys)
    assert output["operation_id"] == "replace-1"
    assert "source_path" not in output

    assert lifecycle_cli.main(["status", "missing"]) == 1
    _output, error = parse_output(capsys)
    assert error == {"error": "not_found", "operation_id": "missing"}


def test_reconcile_returns_nonzero_for_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        lifecycle_cli,
        "reconcile_lifecycle_pending",
        lambda **kwargs: (
            LifecycleReconcileResult(
                operation_id="one",
                outcome="completed",
                state="completed",
                source_cleanup_required=None,
            ),
            LifecycleReconcileResult(
                operation_id="two",
                outcome="error",
                state="index_committed",
                source_cleanup_required=None,
            ),
        ),
    )
    assert lifecycle_cli.main(["reconcile", "--limit", "2"]) == 1
    output, error = parse_output(capsys)
    assert error is None
    assert output["count"] == 2
    assert output["results"][1]["outcome"] == "error"


def test_retry_requires_exact_confirmation_and_failed_state(
    tmp_path, monkeypatch, capsys
):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    planned(outbox)
    claimed = outbox.claim(worker_id="worker", now=2.0)
    outbox.record_failure(
        claimed[0].operation_id,
        worker_id="worker",
        error_type="RuntimeError",
        now=3.0,
    )
    monkeypatch.setattr(lifecycle_cli, "get_lifecycle_outbox", lambda: outbox)

    assert lifecycle_cli.main(
        [
            "retry-failed",
            "replace-1",
            "--confirm-operation-id",
            "wrong",
        ]
    ) == 2
    _output, error = parse_output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert outbox.get("replace-1").state == "failed"

    assert lifecycle_cli.main(
        [
            "retry-failed",
            "replace-1",
            "--confirm-operation-id",
            "replace-1",
        ]
    ) == 0
    output, error = parse_output(capsys)
    assert error is None
    assert output["state"] == "planned"
    assert output["attempts"] == 0


def test_invalid_limits_and_nonfailed_retry_are_generic(
    tmp_path, monkeypatch, capsys
):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    planned(outbox)
    monkeypatch.setattr(lifecycle_cli, "get_lifecycle_outbox", lambda: outbox)
    assert lifecycle_cli.main(["pending", "--limit", "0"]) == 2
    _output, error = parse_output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert lifecycle_cli.main(
        [
            "retry-failed",
            "replace-1",
            "--confirm-operation-id",
            "replace-1",
        ]
    ) == 2
    _output, error = parse_output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
