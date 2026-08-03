from __future__ import annotations

import json
from types import SimpleNamespace

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_cli as cli,
)


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def profile(state="active"):
    return SimpleNamespace(
        owner_id="alice",
        profile_id="tsa",
        policy_oid="1.2.3.4.1",
        trust_anchor_bundle_sha256="1" * 64,
        untrusted_bundle_sha256=None,
        crl_bundle_sha256=None,
        allowed_signer_certificate_sha256=("2" * 64,),
        valid_from=1.0,
        valid_until=None,
        state=state,
        registered_at=2.0,
        retired_at=None if state == "active" else 3.0,
        record_digest="3" * 64,
    )


def test_bad_confirmation_precedes_registry_and_actor(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_rfc3161_trust_registry",
        lambda: (_ for _ in ()).throw(AssertionError("registry loaded")),
    )
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(AssertionError("actor loaded")),
    )
    assert cli.main(
        [
            "register",
            "--owner-id",
            "alice",
            "--profile-id",
            "tsa",
            "--confirm-profile-id",
            "other",
            "--policy-oid",
            "1.2.3.4.1",
            "--trust-anchor-bundle",
            "roots.pem",
            "--valid-from",
            "1",
        ]
    ) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}


def test_status_and_list_are_read_only_path_free(monkeypatch, capsys):
    value = profile()

    class Registry:
        def get(self, **kwargs):
            assert kwargs == {"owner_id": "alice", "profile_id": "tsa"}
            return value

        def list(self, **kwargs):
            assert kwargs == {
                "owner_id": "alice",
                "state": None,
                "limit": 100,
            }
            return (value,)

    monkeypatch.setattr(
        cli,
        "get_rfc3161_trust_registry",
        lambda: Registry(),
    )
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(AssertionError("actor loaded")),
    )

    assert cli.main(
        ["status", "--owner-id", "alice", "--profile-id", "tsa"]
    ) == 0
    status, error = read(capsys)
    assert error is None
    assert status["registry_mutation_performed"] is False
    assert status["contains_raw_paths"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing, error = read(capsys)
    assert error is None
    assert listing["record_count"] == 1
    assert listing["registry_mutation_performed"] is False
    assert listing["contains_raw_paths"] is False
    assert listing["records"][0]["contains_raw_paths"] is False
