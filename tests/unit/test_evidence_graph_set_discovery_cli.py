from __future__ import annotations

import json

from tools import evidence_graph_set_discovery_cli as cli


def test_cli_is_read_only_and_text_free(monkeypatch, capsys):
    captured = {}

    def listing(**kwargs):
        captured.update(kwargs)
        return [{"graph_set_key": "review", "member_count": 2}]

    monkeypatch.setattr(cli, "list_evidence_graph_sets", listing)
    assert cli.main(
        [
            "--owner-id",
            "alice",
            "--limit",
            "5",
            "--include-unavailable",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "count": 1,
        "graph_sets": [
            {"graph_set_key": "review", "member_count": 2}
        ],
        "mutation_performed": False,
        "semantic_inference_performed": False,
        "source_text_returned": False,
    }
    assert captured == {
        "owner_id": "alice",
        "limit": 5,
        "include_unavailable": True,
    }


def test_cli_contains_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "list_evidence_graph_sets",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("private detail")
        ),
    )
    assert cli.main(["--owner-id", "alice"]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }
