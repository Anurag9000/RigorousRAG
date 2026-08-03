from __future__ import annotations

import os
from dataclasses import replace

import pytest

from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use import (
    CustodySignerAdminUse,
    CustodySignerAdminUseStore,
    deterministic_signer_admin_use_id,
)


def use(
    *,
    binding_digit: str = "1",
    action: str = "register",
    key_id: str = "key-1",
    action_digit: str = "2",
    state: str = "reserved",
):
    binding = binding_digit * 64
    action_digest = action_digit * 64
    reserved = 10.0
    return CustodySignerAdminUse(
        use_id=deterministic_signer_admin_use_id(
            binding_digest=binding,
            owner_id="alice",
            action=action,
            key_id=key_id,
            action_digest=action_digest,
        ),
        binding_digest=binding,
        assertion_digest="3" * 64,
        assertion_issuer="review-issuer",
        assertion_expires_at=100.0,
        actor_id="reviewer",
        binding_method="signed_assertion",
        owner_id="alice",
        action=action,
        key_id=key_id,
        action_digest=action_digest,
        state=state,
        reserved_at=reserved,
        committed_at=20.0 if state == "committed" else None,
    )


def test_signer_admin_use_identity_is_scope_bound_and_tamper_evident():
    first = use()
    second = use()
    assert first == second
    assert len(first.use_id) == 64
    assert first.use_id != use(action="retire").use_id
    assert first.use_id != use(key_id="key-2").use_id
    with pytest.raises(ValueError, match="use_id"):
        replace(first, key_id="other")
    with pytest.raises(ValueError, match="expired"):
        replace(first, assertion_expires_at=9.0)


def test_one_binding_digest_can_reserve_only_one_action(tmp_path):
    store = CustodySignerAdminUseStore(tmp_path / "uses.sqlite3")
    first = store.reserve(use())
    assert store.reserve(use()) == first
    with pytest.raises(RuntimeError, match="another action"):
        store.reserve(use(action="retire", action_digit="4"))


def test_commit_is_exact_monotonic_and_idempotent(tmp_path):
    store = CustodySignerAdminUseStore(tmp_path / "uses.sqlite3")
    reserved = store.reserve(use())
    with pytest.raises(ValueError, match="confirmation"):
        store.commit(
            reserved.use_id,
            confirm_use_id="f" * 64,
            now=20.0,
        )
    committed = store.commit(
        reserved.use_id,
        confirm_use_id=reserved.use_id,
        now=20.0,
    )
    assert committed.state == "committed"
    assert committed.committed_at == 20.0
    assert store.commit(
        reserved.use_id,
        confirm_use_id=reserved.use_id,
        now=30.0,
    ) == committed


def test_database_identity_and_payload_tampering_fail_closed(tmp_path):
    path = tmp_path / "uses.sqlite3"
    store = CustodySignerAdminUseStore(path)
    value = store.reserve(use())
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.get(value.use_id)

    second = CustodySignerAdminUseStore(tmp_path / "second.sqlite3")
    value = second.reserve(use())
    with second._lock, second._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_signer_admin_uses "
            "SET key_id=? WHERE use_id=?",
            ("other", value.use_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        second.get(value.use_id)
