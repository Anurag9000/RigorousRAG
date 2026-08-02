from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_rag_run_cli as cli

PLAN = "a" * 64


class Store:
    def __init__(self):
        self.values = [
            SimpleNamespace(
                plan_fingerprint=PLAN,
                benchmark_fingerprint="b" * 64,
                benchmark_id="bench",
                run_id="run-1",
                seed=1,
                case_count=10,
                run_contract_digest="c" * 64,
                run_report=SimpleNamespace(report_digest="d" * 64),
                stored_run_digest="e" * 64,
                completed_at=1.0,
            )
        ]
        self.removed = None

    def list_plan(self, value):
        return tuple(self.values) if value == PLAN else ()

    def remove_plan(self, value, *, confirm_plan_fingerprint):
        if value != confirm_plan_fingerprint:
            raise ValueError("confirmation")
        self.removed = value
        return bool(self.values)


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_status_is_text_free_and_read_only(monkeypatch, capsys):
    store = Store()
    monkeypatch.setattr(cli, "get_graph_rag_run_store", lambda: store)
    assert cli.main(["status", PLAN]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["mutation_performed"] is False
    assert output["runs"][0]["contains_raw_query"] is False
    assert "private text" not in json.dumps(output).lower()


def test_missing_plan_is_bounded(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_graph_rag_run_store", lambda: Store())
    assert cli.main(["status", "f" * 64]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found"}


def test_remove_requires_exact_confirmation(monkeypatch, capsys):
    store = Store()
    monkeypatch.setattr(cli, "get_graph_rag_run_store", lambda: store)
    assert (
        cli.main(
            [
                "remove-plan",
                PLAN,
                "--confirm-plan-fingerprint",
                PLAN,
            ]
        )
        == 0
    )
    output, error = parse(capsys)
    assert error is None and output["removed"] is True
    assert store.removed == PLAN
    assert output["query_or_evidence_text_removed"] is False
