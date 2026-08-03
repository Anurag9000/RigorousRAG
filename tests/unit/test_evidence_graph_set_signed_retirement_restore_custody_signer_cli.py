from __future__ import annotations

import json
from types import SimpleNamespace

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_cli as cli,
)


def record(*, state="active"):
    return SimpleNamespace(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        algorithm="ed25519",
        public_key_sha256="1" * 64,
        state=state,
        registered_actor_id="registrar",
        registered_binding_method="process_environment",
        registered_binding_digest="2" * 64,
        registered_at=10.0,
        retired_actor_id="retirer" if state == "retired" else None,
        retired_binding_method=(
            "process_environment" if state == "retired" else None
        ),
        retired_binding_digest="3" * 64 if state == "retired" else None,
        retired_at=20.0 if state == "retired" else None,
        record_digest="4" * 64,
    )


def test_register_confirmation_fails_before_actor_and_registry(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_public", lambda _path: object())
    monkeypatch.setattr(cli, "_public_fingerprint", lambda _key: "1" * 64)
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(
            AssertionError("actor must not be loaded")
        ),
    )
    monkeypatch.setattr(
        cli,
        "get_custody_signer_key_registry",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("registry must not be opened")
        ),
    )

    assert cli.main(
        [
            "register",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--issuer",
            "lab-security",
            "--public-key-path",
            "/private/public.pem",
            "--confirm-public-key-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}


def test_sign_governed_requires_active_matching_owner_and_private_key(
    monkeypatch,
    capsys,
):
    current = record()

    class ReadOnly:
        def __init__(self, _path):
            pass

        def get(self, **kwargs):
            return current

    monkeypatch.setattr(cli, "ReadOnlyCustodySignerKeyRegistry", ReadOnly)
    monkeypatch.setattr(
        cli,
        "verify_restore_chain_of_custody",
        lambda _path: SimpleNamespace(
            owner_id="alice",
            restore_id="5" * 64,
            chain_digest="6" * 64,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_private",
        lambda _path: SimpleNamespace(public_key=lambda: object()),
    )
    monkeypatch.setattr(
        cli,
        "_public_fingerprint",
        lambda _key: current.public_key_sha256,
    )
    envelope = SimpleNamespace(
        manifest=SimpleNamespace(
            owner_id="alice",
            restore_id="5" * 64,
            chain_digest="6" * 64,
        )
    )
    monkeypatch.setattr(
        cli,
        "sign_restore_chain_of_custody",
        lambda **kwargs: envelope,
    )

    assert cli.main(
        [
            "sign-governed",
            "chain.json",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--private-key-path",
            "/private/private.pem",
            "--output",
            "/private/signed.json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    rendered = json.dumps(payload)
    assert payload["signature_created"] is True
    assert payload["eligible_for_new_signatures"] is True
    assert payload["contains_private_key_material"] is False
    assert "/private" not in rendered
    assert "registrar" not in rendered

    current.state = "retired"
    assert cli.main(
        [
            "sign-governed",
            "chain.json",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--private-key-path",
            "private.pem",
            "--output",
            "signed.json",
        ]
    ) == 1
    assert json.loads(capsys.readouterr().err) == {
        "error": "not_authorized_or_untrusted"
    }


def test_retired_registered_key_still_verifies_historical_envelope(
    monkeypatch,
    capsys,
):
    current = record(state="retired")

    class ReadOnly:
        def __init__(self, _path):
            pass

        def get(self, **kwargs):
            return current

    monkeypatch.setattr(cli, "ReadOnlyCustodySignerKeyRegistry", ReadOnly)
    monkeypatch.setattr(
        cli,
        "verify_signed_restore_chain_of_custody",
        lambda **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(
                owner_id="alice",
                restore_id="5" * 64,
                chain_digest="6" * 64,
            )
        ),
    )

    assert cli.main(
        [
            "verify-registered",
            "signed.json",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--public-key-path",
            "public.pem",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signature_valid"] is True
    assert payload["historical_verification_allowed"] is True
    assert payload["eligible_for_new_signatures"] is False
    assert payload["registry_mutation_performed"] is False


def test_status_and_list_hide_actor_ids_and_paths(monkeypatch, capsys):
    current = record()

    class ReadOnly:
        def __init__(self, _path):
            pass

        def get(self, **kwargs):
            return current

        def list(self, **kwargs):
            return (current,)

    monkeypatch.setattr(cli, "ReadOnlyCustodySignerKeyRegistry", ReadOnly)
    assert cli.main(
        ["status", "--owner-id", "alice", "--key-id", "key-1"]
    ) == 0
    status = capsys.readouterr().out
    assert "registrar" not in status
    assert "path" not in status.lower()

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing = capsys.readouterr().out
    assert "registrar" not in listing
    assert "key_material_mutation_performed" in listing
