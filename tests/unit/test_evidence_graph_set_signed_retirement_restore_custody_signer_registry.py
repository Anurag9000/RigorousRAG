from __future__ import annotations

import os
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_readonly import (
    ReadOnlyCustodySignerKeyRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_registry import (
    CustodySignerKeyRecord,
    CustodySignerKeyRegistry,
)


def actor(name="operator"):
    return ReviewActorBinding.create(
        actor_id=name,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def keys(tmp_path, name):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{name}.private.pem"
    public_path = tmp_path / f"{name}.public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    if os.name != "nt":
        private_path.chmod(0o600)
    return private_path, public_path


def test_registry_registers_idempotently_and_detects_identity_collisions(tmp_path):
    registry = CustodySignerKeyRegistry(tmp_path / "signers.sqlite3")
    _private, public = keys(tmp_path, "first")
    first = registry.register(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_path=public,
        actor=actor(),
        now=10.0,
    )
    replay = registry.register(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_path=public,
        actor=actor(),
        now=20.0,
    )
    assert replay == first
    assert first.state == "active"
    assert len(first.public_key_sha256) == 64
    assert first.registered_at == 10.0

    _other_private, other_public = keys(tmp_path, "other")
    with pytest.raises(RuntimeError, match="collision"):
        registry.register(
            owner_id="alice",
            key_id="key-1",
            issuer="lab-security",
            public_key_path=other_public,
            actor=actor(),
            now=30.0,
        )
    with pytest.raises(Exception):
        registry.register(
            owner_id="alice",
            key_id="key-alias",
            issuer="lab-security",
            public_key_path=public,
            actor=actor(),
            now=30.0,
        )


def test_registry_supports_overlap_and_monotonic_retirement(tmp_path):
    registry = CustodySignerKeyRegistry(tmp_path / "signers.sqlite3")
    _private1, public1 = keys(tmp_path, "first")
    _private2, public2 = keys(tmp_path, "second")
    first = registry.register(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_path=public1,
        actor=actor("registrar"),
        now=10.0,
    )
    second = registry.register(
        owner_id="alice",
        key_id="key-2",
        issuer="lab-security",
        public_key_path=public2,
        actor=actor("registrar"),
        now=11.0,
    )
    assert {value.key_id for value in registry.list(owner_id="alice", state="active")} == {
        first.key_id,
        second.key_id,
    }

    with pytest.raises(ValueError, match="confirmation"):
        registry.retire(
            owner_id="alice",
            key_id="key-1",
            confirm_key_id="key-2",
            actor=actor("retirer"),
            now=20.0,
        )
    retired = registry.retire(
        owner_id="alice",
        key_id="key-1",
        confirm_key_id="key-1",
        actor=actor("retirer"),
        now=20.0,
    )
    assert retired.state == "retired"
    assert retired.retired_at == 20.0
    assert registry.get(owner_id="alice", key_id="key-2").state == "active"
    assert registry.retire(
        owner_id="alice",
        key_id="key-1",
        confirm_key_id="key-1",
        actor=actor("retirer"),
        now=30.0,
    ) == retired
    with pytest.raises(RuntimeError, match="another actor"):
        registry.retire(
            owner_id="alice",
            key_id="key-1",
            confirm_key_id="key-1",
            actor=actor("other"),
            now=30.0,
        )


def test_record_and_database_tampering_fail_closed(tmp_path):
    registry = CustodySignerKeyRegistry(tmp_path / "signers.sqlite3")
    _private, public = keys(tmp_path, "first")
    value = registry.register(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_path=public,
        actor=actor(),
        now=10.0,
    )
    with pytest.raises(ValueError, match="record_digest"):
        replace(value, issuer="other")

    with registry._lock, registry._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_signers "
            "SET public_key_sha256=? WHERE owner_id=? AND key_id=?",
            ("0" * 64, "alice", "key-1"),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        registry.get(owner_id="alice", key_id="key-1")


def test_read_only_registry_requires_schema_and_rejects_writes(tmp_path):
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="not initialized"):
        ReadOnlyCustodySignerKeyRegistry(empty)

    registry = CustodySignerKeyRegistry(tmp_path / "signers.sqlite3")
    _private, public = keys(tmp_path, "first")
    value = registry.register(
        owner_id="alice",
        key_id="key-1",
        issuer="lab-security",
        public_key_path=public,
        actor=actor(),
        now=10.0,
    )
    read_only = ReadOnlyCustodySignerKeyRegistry(registry.path)
    assert read_only.get(owner_id="alice", key_id="key-1") == value
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_signers"
            )
