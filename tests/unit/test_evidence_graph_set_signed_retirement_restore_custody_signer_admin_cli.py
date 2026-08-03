from __future__ import annotations

import json
from types import SimpleNamespace

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_admin_cli as cli,
)


def binding():
    return SimpleNamespace(
        actor_id="reviewer",
        binding_method="signed_assertion",
        binding_digest="1" * 64,
        assertion_digest="2" * 64,
        assertion_issuer="review-issuer",
        assertion_expires_at=9999999999.0,
    )


def use(state="reserved"):
    return SimpleNamespace(
        use_id="3" * 64,
        assertion_issuer="review-issuer",
        assertion_expires_at=9999999999.0,
        binding_method="signed_assertion",
        owner_id="alice",
        action="register",
        key_id="key-1",
        action_digest="4" * 64,
        state=state,
        reserved_at=10.0,
        committed_at=20.0 if state == "committed" else None,
    )


def record(state="active"):
    return SimpleNamespace(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        algorithm="ed25519",
        public_key_sha256="5" * 64,
        state=state,
        registered_binding_method="signed_assertion",
        registered_binding_digest="1" * 64,
        registered_at=10.0,
        retired_binding_method="signed_assertion" if state == "retired" else None,
        retired_binding_digest="1" * 64 if state == "retired" else None,
        retired_at=20.0 if state == "retired" else None,
        record_digest="6" * 64,
    )


class Store:
    def __init__(self, prior=None):
        self.prior = prior
        self.reserved = []
        self.committed = []

    def get(self, use_id):
        if self.prior is None:
            raise KeyError(use_id)
        return self.prior

    def reserve(self, value):
        self.reserved.append(value)
        return use()

    def commit(self, use_id, **kwargs):
        self.committed.append((use_id, kwargs))
        return use("committed")


class Registry:
    def __init__(self, existing=None):
        self.existing = existing
        self.registered = []
        self.retired = []

    def get(self, **kwargs):
        if self.existing is None:
            raise KeyError(kwargs)
        return self.existing

    def register(self, **kwargs):
        self.registered.append(kwargs)
        self.existing = record()
        return self.existing

    def retire(self, **kwargs):
        self.retired.append(kwargs)
        self.existing = record("retired")
        return self.existing


def install(monkeypatch, *, registry, store):
    monkeypatch.setattr(cli, "load_relation_review_actor", binding)
    monkeypatch.setattr(
        cli,
        "require_relation_review_actor",
        lambda requested, *, binding: binding,
    )
    monkeypatch.setattr(cli, "get_custody_signer_key_registry", lambda path: registry)
    monkeypatch.setattr(cli, "get_custody_signer_admin_use_store", lambda path: store)
    monkeypatch.setattr(cli, "_load_public", lambda path: object())
    monkeypatch.setattr(cli, "_public_fingerprint", lambda key: "5" * 64)
    monkeypatch.setattr(
        cli.CustodySignerAdminUse,
        "reserve",
        classmethod(lambda cls, **kwargs: use()),
    )


def test_registration_confirmation_fails_before_actor_or_stores(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_public", lambda path: object())
    monkeypatch.setattr(cli, "_public_fingerprint", lambda key: "5" * 64)
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(AssertionError("actor must not load")),
    )
    assert cli.main(
        [
            "register-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--issuer",
            "lab-security",
            "--public-key-path",
            "public.pem",
            "--confirm-public-key-sha256",
            "0" * 64,
        ]
    ) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }


def test_new_signed_registration_reserves_before_registry_and_commits(monkeypatch, capsys):
    registry = Registry()
    store = Store()
    install(monkeypatch, registry=registry, store=store)

    assert cli.main(
        [
            "register-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--issuer",
            "lab-security",
            "--public-key-path",
            "public.pem",
            "--confirm-public-key-sha256",
            "5" * 64,
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(store.reserved) == 1
    assert len(registry.registered) == 1
    assert len(store.committed) == 1
    assert payload["admin_use"]["state"] == "committed"
    assert payload["contains_source_text"] is False
    assert "reviewer" not in json.dumps(payload)


def test_existing_registration_without_prior_reservation_cannot_be_backfilled(
    monkeypatch,
    capsys,
):
    registry = Registry(existing=record())
    store = Store(prior=None)
    install(monkeypatch, registry=registry, store=store)

    assert cli.main(
        [
            "register-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--issuer",
            "lab-security",
            "--public-key-path",
            "public.pem",
            "--confirm-public-key-sha256",
            "5" * 64,
        ]
    ) == 1
    assert json.loads(capsys.readouterr().err) == {"error": "not_authorized"}
    assert store.reserved == []
    assert registry.registered == []


def test_existing_action_with_prior_reservation_recovers_and_commits(
    monkeypatch,
    capsys,
):
    registry = Registry(existing=record())
    store = Store(prior=use())
    install(monkeypatch, registry=registry, store=store)

    assert cli.main(
        [
            "register-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--issuer",
            "lab-security",
            "--public-key-path",
            "public.pem",
            "--confirm-public-key-sha256",
            "5" * 64,
        ]
    ) == 0
    json.loads(capsys.readouterr().out)
    assert store.reserved == []
    assert len(registry.registered) == 1
    assert len(store.committed) == 1


def test_signed_retirement_requires_confirmation_and_commits_use(monkeypatch, capsys):
    registry = Registry(existing=record())
    store = Store()
    install(monkeypatch, registry=registry, store=store)
    monkeypatch.setattr(
        cli.CustodySignerAdminUse,
        "reserve",
        classmethod(
            lambda cls, **kwargs: SimpleNamespace(
                **{
                    **use().__dict__,
                    "action": "retire",
                }
            )
        ),
    )

    assert cli.main(
        [
            "retire-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--confirm-key-id",
            "wrong",
        ]
    ) == 2
    capsys.readouterr()
    assert registry.retired == []

    assert cli.main(
        [
            "retire-signed",
            "--owner-id",
            "alice",
            "--key-id",
            "key-1",
            "--confirm-key-id",
            "key-1",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(store.reserved) == 1
    assert len(registry.retired) == 1
    assert len(store.committed) == 1
    assert payload["signer_record"]["state"] == "retired"
