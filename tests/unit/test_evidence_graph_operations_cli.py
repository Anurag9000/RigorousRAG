from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_operations_cli as cli
from tools.evidence_graph_jobs import EvidenceGraphJob, deterministic_graph_job_id

A = "a" * 64
B = "b" * 64


def completed_job():
    return EvidenceGraphJob(
        job_id=deterministic_graph_job_id(
            owner_id="alice",
            doc_id="doc-1",
            source_sequence=1,
            source_state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=2,
        ),
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=1,
        source_state="active",
        content_sha256=A,
        profile_fingerprint=B,
        sparse_generation=2,
        state="completed",
        attempt_count=1,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        graph_digest="c" * 64,
        failure_type=None,
        created_at=1.0,
        updated_at=2.0,
    )


class Journal:
    def list(self, **kwargs):
        return (completed_job(),)


class Generations:
    def current(self, **kwargs):
        return SimpleNamespace(
            owner_id="alice",
            doc_id="doc-1",
            sequence=2,
            state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=3,
        )


class Graphs:
    def get(self, **kwargs):
        return SimpleNamespace(generation=1, graph_digest="c" * 64)

    def current(self, **kwargs):
        return SimpleNamespace(generation=2, graph_digest="d" * 64)


def install(monkeypatch):
    monkeypatch.setattr(cli, "get_evidence_graph_job_journal", lambda: Journal())
    monkeypatch.setattr(cli, "get_generation_store", lambda: Generations())
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: Graphs())


def read(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out) if captured.out else json.loads(captured.err)


def test_audit_output_is_privacy_safe(monkeypatch, capsys):
    install(monkeypatch)
    assert cli.main(["audit", "--owner-id", "alice"]) == 0
    output = read(capsys)
    rendered = json.dumps(output).lower()
    assert output["mutation_performed"] is False
    assert output["contains_graph_text"] is False
    assert "path" not in rendered and "document text" not in rendered


def test_retention_plan_never_authorizes_deletion(monkeypatch, capsys):
    install(monkeypatch)
    assert (
        cli.main(
            [
                "retention-plan",
                "--owner-id",
                "alice",
                "--min-age-seconds",
                "1",
            ]
        )
        == 0
    )
    output = read(capsys)
    assert output["mutation_performed"] is False
    assert output["deletion_authorized"] is False
    assert len(output["candidates"]) == 1


def test_invalid_age_fails_closed(monkeypatch, capsys):
    install(monkeypatch)
    assert (
        cli.main(
            [
                "retention-plan",
                "--owner-id",
                "alice",
                "--min-age-seconds",
                "nan",
            ]
        )
        == 2
    )
    assert read(capsys) == {"error": "invalid_or_unavailable"}
