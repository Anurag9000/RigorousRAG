from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_jobs_cli as cli
from tools.evidence_graph_jobs import EvidenceGraphJobJournal

A = "a" * 64
B = "b" * 64


class Generations:
    def __init__(self):
        self.record = SimpleNamespace(
            owner_id="alice",
            doc_id="doc-1",
            sequence=2,
            state="deleted",
            content_sha256=A,
            profile_fingerprint=B,
            vector_rows=0,
            sparse_generation=0,
        )

    def current(self, **kwargs):
        return self.record


class Graphs:
    def __init__(self):
        self.value = None

    def current(self, **kwargs):
        return self.value

    def commit(self, batch, **kwargs):
        self.value = batch
        return batch


class Sparse:
    def snapshot_document(self, **kwargs):
        return None


def install(monkeypatch, tmp_path):
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    generations = Generations()
    graphs = Graphs()
    sparse = Sparse()
    monkeypatch.setattr(cli, "get_evidence_graph_job_journal", lambda: journal)
    monkeypatch.setattr(cli, "get_generation_store", lambda: generations)
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: graphs)
    monkeypatch.setattr(cli, "get_sparse_index", lambda: sparse)
    return journal


def output(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_seed_status_list_and_reconcile_are_path_free(
    tmp_path, monkeypatch, capsys
):
    install(monkeypatch, tmp_path)
    assert cli.main(["seed", "--owner-id", "alice", "--doc-id", "doc-1"]) == 0
    seeded, error = output(capsys)
    assert error is None
    assert seeded["state"] == "planned"
    assert seeded["authoritative_mutation_performed"] is False
    assert "path" not in json.dumps(seeded).lower()
    assert "text" not in json.dumps(seeded).lower()
    assert cli.main(["status", seeded["job_id"]]) == 0
    status, _error = output(capsys)
    assert status["job_id"] == seeded["job_id"]
    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing, _error = output(capsys)
    assert listing["count"] == 1
    assert (
        cli.main(
            [
                "reconcile-one",
                "--owner-id",
                "alice",
                "--worker-id",
                "worker",
            ]
        )
        == 0
    )
    completed, _error = output(capsys)
    assert completed["state"] == "completed"


def test_missing_status_and_idle_are_bounded(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    assert cli.main(["status", "f" * 64]) == 1
    _out, error = output(capsys)
    assert error == {"error": "not_found", "job_id": "f" * 64}
    assert (
        cli.main(
            [
                "reconcile-one",
                "--owner-id",
                "alice",
                "--worker-id",
                "worker",
            ]
        )
        == 0
    )
    idle, error = output(capsys)
    assert error is None and idle["status"] == "idle"


def test_cancel_requires_exact_confirmation(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    cli.main(["seed", "--owner-id", "alice", "--doc-id", "doc-1"])
    seeded, _error = output(capsys)
    assert (
        cli.main(
            [
                "cancel",
                seeded["job_id"],
                "--owner-id",
                "alice",
                "--confirm-job-id",
                "e" * 64,
            ]
        )
        == 2
    )
    _out, error = output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert (
        cli.main(
            [
                "cancel",
                seeded["job_id"],
                "--owner-id",
                "alice",
                "--confirm-job-id",
                seeded["job_id"],
            ]
        )
        == 0
    )
    cancelled, _error = output(capsys)
    assert cancelled["state"] == "cancelled"


def test_cli_rejects_non_positive_attempt_policy(tmp_path, monkeypatch, capsys):
    install(monkeypatch, tmp_path)
    assert (
        cli.main(
            [
                "seed",
                "--owner-id",
                "alice",
                "--doc-id",
                "doc-1",
                "--max-attempts",
                "0",
            ]
        )
        == 2
    )
    _out, error = output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
