from __future__ import annotations

import json
from pathlib import Path

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_rfc3161_cli as cli,
)


def output(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_create_emit_inspect_are_offline_and_path_free(tmp_path: Path, capsys):
    subject = tmp_path / "private-custody-envelope.json"
    subject.write_text("custody")
    bundle = tmp_path / "request.json"
    request = tmp_path / "request.tsq"

    assert cli.main(
        [
            "create-request",
            "--owner-id",
            "alice",
            "--custody-envelope",
            str(subject),
            "--output-bundle",
            str(bundle),
            "--policy-oid",
            "1.2.3.4.1",
        ]
    ) == 0
    created, error = output(capsys)
    assert error is None
    assert created["network_request_performed"] is False
    assert created["trusted_time_obtained"] is False
    assert created["contains_raw_subject_content"] is False
    assert str(subject) not in json.dumps(created)

    assert cli.main(
        [
            "emit-request",
            "--request-bundle",
            str(bundle),
            "--output-der",
            str(request),
        ]
    ) == 0
    emitted, error = output(capsys)
    assert error is None
    assert emitted["rfc3161_request"] is True
    assert request.read_bytes()

    assert cli.main(
        ["inspect-request", "--request-bundle", str(bundle)]
    ) == 0
    inspected, error = output(capsys)
    assert error is None
    assert inspected["request_bundle_digest"] == created["request_bundle_digest"]


def test_cli_failures_are_generic(tmp_path: Path, capsys):
    missing = tmp_path / "missing"
    assert cli.main(
        ["inspect-request", "--request-bundle", str(missing)]
    ) == 2
    rendered, error = output(capsys)
    assert rendered is None
    assert error == {"error": "invalid_or_unavailable"}
