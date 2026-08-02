from __future__ import annotations

import json
import os
import stat

import pytest

from tools import evidence_graph_set_signed_retirement_snapshot_cli as cli
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    build_signed_retirement_snapshot,
    export_signed_retirement_snapshot,
    verify_signed_retirement_snapshot,
)


def attempt(digit: str, *, operation_digit: str | None = None):
    return SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id=(operation_digit or digit) * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        now=float(int(digit, 16) + 1),
    )


class Journal:
    def __init__(self, values=()):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def test_snapshot_build_is_deterministic_and_owner_scoped():
    journal = Journal((attempt("b"), attempt("a")))
    first = build_signed_retirement_snapshot(
        owner_id="alice",
        journal=journal,
        now=100.0,
        limit=100,
    )
    second = build_signed_retirement_snapshot(
        owner_id="alice",
        journal=journal,
        now=100.0,
        limit=100,
    )

    assert first == second
    assert tuple(value.retirement_id for value in first.records) == tuple(
        sorted(value.retirement_id for value in first.records)
    )
    assert first.record_count == 2
    assert len(first.snapshot_digest) == 64
    assert first.public_payload()["contains_source_text"] is False


def test_export_is_atomic_no_overwrite_and_round_trips(tmp_path):
    output = tmp_path / "snapshots" / "retirements.json"
    snapshot = export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Journal((attempt("a"),)),
        output_path=output,
        now=100.0,
        limit=100,
    )
    verified = verify_signed_retirement_snapshot(output)

    assert verified == snapshot
    assert output.read_bytes().endswith(b"\n")
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        export_signed_retirement_snapshot(
            owner_id="alice",
            journal=Journal((attempt("a"),)),
            output_path=output,
            now=100.0,
            limit=100,
        )


def test_snapshot_tampering_and_duplicate_keys_fail_closed(tmp_path):
    output = tmp_path / "retirements.json"
    export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Journal((attempt("a"),)),
        output_path=output,
        now=100.0,
        limit=100,
    )
    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["records"][0]["graph_set_key"] = "other"
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_digest"):
        verify_signed_retirement_snapshot(output)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        verify_signed_retirement_snapshot(duplicate)


def test_snapshot_rejects_redirects_and_bounded_result_truncation(tmp_path):
    with pytest.raises(RuntimeError, match="bounded result limit"):
        build_signed_retirement_snapshot(
            owner_id="alice",
            journal=Journal((attempt("a"),)),
            now=1.0,
            limit=1,
        )

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="redirect"):
        verify_signed_retirement_snapshot(link)


def test_verify_cli_does_not_load_live_journal(tmp_path, monkeypatch, capsys):
    output = tmp_path / "retirements.json"
    export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Journal((attempt("a"),)),
        output_path=output,
        now=100.0,
        limit=100,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_publication_retirement_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("verify must not load live journal")
        ),
    )

    assert cli.main(["verify", str(output)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["record_count"] == 1
    assert payload["restore_performed"] is False
    assert payload["journal_mutation_performed"] is False
    assert payload["contains_assertion_secrets"] is False
