from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_custody_signature_cli as sig_cli
from tools import evidence_graph_set_signed_retirement_restore_custody_signature_keys_cli as key_cli
from tools import evidence_graph_set_signed_retirement_restore_custody_signature_keys_runtime as runtime


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def record(state="active"):
    return SimpleNamespace(
        owner_id="alice",
        key_id="key-1",
        algorithm="ed25519",
        public_key_sha256="1" * 64,
        valid_from=1.0,
        valid_until=None,
        state=state,
        registered_actor_id_digest="2" * 64,
        registered_binding_method="process_environment",
        registered_binding_digest="3" * 64,
        registered_at=1.0,
        retired_actor_id_digest=None if state == "active" else "4" * 64,
        retired_binding_method=None if state == "active" else "process_environment",
        retired_binding_digest=None if state == "active" else "5" * 64,
        retired_at=None if state == "active" else 2.0,
        record_digest="6" * 64,
    )


def test_key_confirmation_precedes_actor_and_registry(monkeypatch, capsys):
    monkeypatch.setattr(
        key_cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(AssertionError("actor loaded")),
    )
    monkeypatch.setattr(
        key_cli,
        "get_custody_signer_key_registry",
        lambda *args: (_ for _ in ()).throw(AssertionError("registry loaded")),
    )
    assert key_cli.main(
        [
            "register",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--public-key-path",
            "public.pem",
            "--confirm-key-id",
            "other",
        ]
    ) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}


def test_key_status_list_and_retire_are_path_free(monkeypatch, capsys):
    class Registry:
        def get(self, **kwargs):
            return record()

        def list(self, **kwargs):
            return (record(),)

        def retire(self, **kwargs):
            return record("retired")

    monkeypatch.setattr(key_cli, "get_custody_signer_key_registry", lambda *args: Registry())
    monkeypatch.setattr(key_cli, "load_relation_review_actor", lambda: object())

    assert key_cli.main(["status", "--owner-id", "alice", "--key-id", "key-1"]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["public_key_sha256"] == "1" * 64
    assert "public_key_path" not in output
    assert "private_key_path" not in output
    assert "registry_path" not in output

    assert key_cli.main(["list", "--owner-id", "alice"]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["count"] == 1
    assert output["mutation_performed"] is False

    assert key_cli.main(
        [
            "retire",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--confirm-key-id",
            "key-1",
        ]
    ) == 0
    output, error = read(capsys)
    assert error is None
    assert output["state"] == "retired"
    assert output["mutation_performed"] is True


def test_signature_cli_sign_and_generic_failure(monkeypatch, capsys):
    envelope = SimpleNamespace(
        owner_id="alice",
        key_id="key-1",
        algorithm="ed25519",
        public_key_sha256="1" * 64,
        manifest=SimpleNamespace(chain_digest="2" * 64),
        envelope_digest="3" * 64,
        created_at=1.0,
    )
    monkeypatch.setattr(sig_cli, "get_custody_signer_key_registry", lambda path=None: object())
    monkeypatch.setattr(
        sig_cli,
        "sign_governed_restore_chain_of_custody",
        lambda **kwargs: (envelope, record()),
    )
    assert sig_cli.main(
        [
            "sign",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--manifest",
            "/private/manifest.json",
            "--private-key-path",
            "/private/key.pem",
            "--output",
            "/private/signed.json",
        ]
    ) == 0
    output, error = read(capsys)
    assert error is None
    rendered = json.dumps(output).lower()
    assert "private/key" not in rendered
    assert "private/manifest" not in rendered
    assert output["contains_private_key_material"] is False
    assert output["output_created"] is True

    monkeypatch.setattr(
        sig_cli,
        "verify_signed_restore_chain_of_custody",
        lambda **kwargs: (_ for _ in ()).throw(PermissionError("private details")),
    )
    assert sig_cli.main(
        ["verify", "signed.json", "--public-key-path", "public.pem"]
    ) == 1
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "signature_verification_failed"}


def test_runtime_cache_is_canonical(tmp_path, monkeypatch):
    runtime.clear_custody_signer_key_registry_cache()
    path = tmp_path / "keys.sqlite3"
    monkeypatch.setenv("EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_KEY_DB_PATH", str(path))
    first = runtime.get_custody_signer_key_registry()
    second = runtime.get_custody_signer_key_registry(path)
    assert first is second
    assert first.path == path
