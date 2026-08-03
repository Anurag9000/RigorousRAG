from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_admin_cli_boundary as cli_boundary,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_boundary as credential_boundary,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use import (
    CustodySignerAdminUse,
    CustodySignerAdminUseStore,
    deterministic_signer_admin_use_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_readonly import (
    ReadOnlyCustodySignerAdminUseStore,
)


class Binding:
    def __init__(self, method: str, *, issuer="issuer", expires=100.0):
        self.actor_id = "operator"
        self.binding_method = method
        self.binding_digest = "1" * 64
        self.assertion_digest = "2" * 64
        self.assertion_issuer = issuer
        self.assertion_expires_at = expires


def reserved():
    return CustodySignerAdminUse(
        use_id=deterministic_signer_admin_use_id(
            binding_digest="1" * 64,
            owner_id="alice",
            action="register",
            key_id="key-1",
            action_digest="3" * 64,
        ),
        binding_digest="1" * 64,
        assertion_digest="2" * 64,
        assertion_issuer="issuer",
        assertion_expires_at=100.0,
        actor_id="operator",
        binding_method="signed_assertion",
        owner_id="alice",
        action="register",
        key_id="key-1",
        action_digest="3" * 64,
        state="reserved",
        reserved_at=10.0,
        committed_at=None,
    )


def test_credential_boundary_accepts_expiring_signed_methods_and_refuses_direct(
    monkeypatch,
):
    monkeypatch.setattr(credential_boundary._base, "ReviewActorBinding", Binding)
    value = Binding("oidc_signed_assertion")
    assertion, issuer, expires = credential_boundary._assertion_fields(value)
    assert assertion == "2" * 64
    assert issuer == "issuer"
    assert expires == 100.0
    assert "oidc_signed_assertion" in credential_boundary._base._SIGNED_METHODS

    for method in ("process_environment", "descriptor_file", "command_line"):
        with pytest.raises(PermissionError, match="signed expiring assertion"):
            credential_boundary._assertion_fields(Binding(method))
    with pytest.raises(PermissionError, match="issuer and expiry"):
        credential_boundary._assertion_fields(
            Binding("future_signed_method", issuer=None, expires=None)
        )


def test_read_only_admin_use_view_requires_schema_and_rejects_writes(tmp_path):
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="not initialized"):
        ReadOnlyCustodySignerAdminUseStore(empty)

    writable = CustodySignerAdminUseStore(tmp_path / "uses.sqlite3")
    value = writable.reserve(reserved())
    read_only = ReadOnlyCustodySignerAdminUseStore(writable.path)
    assert read_only.get(value.use_id) == value
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_signer_admin_uses"
            )


def test_canonical_status_uses_only_query_only_view(monkeypatch, capsys):
    value = reserved()

    class ReadOnly:
        def __init__(self, path):
            assert path == "uses.sqlite3"

        def get(self, use_id):
            assert use_id == value.use_id
            return value

    monkeypatch.setattr(
        cli_boundary,
        "ReadOnlyCustodySignerAdminUseStore",
        ReadOnly,
    )
    monkeypatch.setattr(
        cli_boundary._base,
        "get_custody_signer_admin_use_store",
        lambda path: (_ for _ in ()).throw(
            AssertionError("writable store must not be loaded")
        ),
    )
    assert cli_boundary.main(
        ["status", value.use_id, "--admin-use-db-path", "uses.sqlite3"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["use_id"] == value.use_id
    assert payload["admin_use_mutation_performed"] is False
    assert payload["registry_mutation_performed"] is False
    assert payload["contains_assertion_body"] is False
