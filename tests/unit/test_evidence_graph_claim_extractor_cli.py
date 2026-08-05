from __future__ import annotations

import json

from tools import evidence_graph_claim_extractor_cli as cli
from tools.evidence_graph_claim_extractor_runtime import (
    clear_scientific_claim_extractor_runtime_cache,
    get_scientific_claim_extractor_registry,
)


def configure(tmp_path, monkeypatch):
    path = tmp_path / "extractors.sqlite3"
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_REGISTRY_DB_PATH",
        str(path),
    )
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "extractor-admin")
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", raising=False)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH", raising=False)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_JSON",
        json.dumps(
            {
                "schema_version": 1,
                "administrators": [
                    {
                        "administrator_id": "extractor-admin",
                        "owners": ["alice"],
                        "extractor_names": ["claims"],
                        "actions": ["register", "retire"],
                    }
                ],
            }
        ),
    )
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH",
        raising=False,
    )
    clear_scientific_claim_extractor_runtime_cache()
    return path


def register_args():
    return [
        "register",
        "--owner-id", "alice",
        "--extractor-name", "claims",
        "--extractor-version", "1",
        "--extractor-kind", "model",
        "--implementation-sha256", "a" * 64,
        "--configuration-sha256", "b" * 64,
        "--claim-type", "finding",
        "--modality", "asserted",
        "--language", "en",
    ]


def read(capsys):
    captured = capsys.readouterr()
    return (
        None if not captured.out else json.loads(captured.out),
        None if not captured.err else json.loads(captured.err),
        captured.out + captured.err,
    )


def test_register_status_and_list_are_digest_only(tmp_path, monkeypatch, capsys):
    configure(tmp_path, monkeypatch)
    assert cli.main(register_args()) == 0
    registered, error, rendered = read(capsys)
    assert error is None
    assert registered["state"] == "active"
    assert registered["registration_performed"] is True
    assert registered["contains_credentials"] is False
    assert registered["contains_prompt_text"] is False
    assert registered["contains_model_response"] is False
    assert "api_key" not in rendered.casefold()

    assert cli.main([
        "status",
        "--owner-id", "alice",
        "--extractor-name", "claims",
        "--extractor-version", "1",
    ]) == 0
    status, error, _rendered = read(capsys)
    assert error is None
    assert status["record_digest"] == registered["record_digest"]
    assert status["mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice", "--state", "active"]) == 0
    listing, error, _rendered = read(capsys)
    assert error is None
    assert listing["item_count"] == 1
    assert listing["mutation_performed"] is False


def test_retire_requires_exact_current_record_confirmation(tmp_path, monkeypatch, capsys):
    configure(tmp_path, monkeypatch)
    assert cli.main(register_args()) == 0
    registered, _error, _rendered = read(capsys)

    base = [
        "retire",
        "--owner-id", "alice",
        "--extractor-name", "claims",
        "--extractor-version", "1",
    ]
    assert cli.main([*base, "--confirm-record-digest", "f" * 64]) == 2
    output, error, _rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}

    assert cli.main([
        *base,
        "--confirm-record-digest",
        registered["record_digest"],
    ]) == 0
    retired, error, _rendered = read(capsys)
    assert error is None
    assert retired["state"] == "retired"
    assert retired["retirement_performed"] is True


def test_wrong_actor_and_missing_policy_fail_generically(tmp_path, monkeypatch, capsys):
    configure(tmp_path, monkeypatch)
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "other-admin")
    assert cli.main(register_args()) == 2
    output, error, rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert "not authorized" not in rendered.casefold()

    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "extractor-admin")
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_JSON",
        raising=False,
    )
    assert cli.main(register_args()) == 2
    output, error, rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert "policy" not in rendered.casefold()


def test_runtime_cache_is_canonical_path_scoped(tmp_path, monkeypatch):
    path = configure(tmp_path, monkeypatch)
    first = get_scientific_claim_extractor_registry()
    second = get_scientific_claim_extractor_registry(path)
    assert first is second
    assert first.path == path
