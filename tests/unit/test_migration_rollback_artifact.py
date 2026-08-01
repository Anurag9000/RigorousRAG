import base64
from types import SimpleNamespace

import pytest

from tools.migration_cutover_preflight import CutoverPreflight
from tools.migration_rollback_artifact import (
    RollbackEncryptionKey,
    capture_rollback_payload,
    rollback_key_from_environment,
    validate_rollback_payload,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def preflight():
    return CutoverPreflight(
        task_id=E,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=A,
        target_profile_fingerprint=B,
        source_content_sha256=C,
        validation_digest=D,
        promotion_report_digest=F,
        benchmark_fingerprint="1" * 64,
        vector_snapshot_digest="2" * 64,
        sparse_snapshot_digest="3" * 64,
        rollback_identity_digest="4" * 64,
        target_artifact_digest="5" * 64,
        source_vector_rows=1,
        source_sparse_generation=7,
        source_sparse_fields=1,
        target_vector_rows=1,
        target_sparse_rows=1,
        created_at=1,
    )


def snapshot_for(value):
    vector = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        ids=("v1",),
        documents=("one",),
        metadatas=({"owner_id": "alice", "doc_id": "doc-1"},),
    )
    field = SimpleNamespace(
        field_id="f1",
        field_type="body",
        text="one",
        position=0,
        token_count=1,
        page_number=1,
        section="A",
        metadata={},
    )
    sparse = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        generation=7,
        profile_fingerprint=A,
        metadata={},
        fields=(field,),
        schema_version=1,
    )
    generation = SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        sequence=4,
        state="active",
        content_sha256=C,
        profile_fingerprint=A,
        vector_rows=1,
        sparse_generation=7,
        committed_at=1.0,
        metadata={},
    )
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        stores=SimpleNamespace(vector=vector, sparse=sparse),
        generation=generation,
    )


def aligned_preflight():
    from tools.migration_cutover_preflight import (
        _sha256,
        _sparse_identity,
        _vector_identity,
    )

    value = preflight()
    snapshot = snapshot_for(value)
    vector_digest, vector_count = _vector_identity(
        snapshot.stores.vector,
        value.owner_id,
        value.doc_id,
    )
    sparse_digest, sparse_count = _sparse_identity(
        snapshot.stores.sparse,
        value.owner_id,
        value.doc_id,
        value.source_profile_fingerprint,
        value.source_sparse_generation,
    )
    rollback = _sha256(
        {
            "owner_id": value.owner_id,
            "doc_id": value.doc_id,
            "source_sequence": value.source_sequence,
            "source_profile_fingerprint": value.source_profile_fingerprint,
            "source_content_sha256": value.source_content_sha256,
            "vector_snapshot_digest": vector_digest,
            "sparse_snapshot_digest": sparse_digest,
            "vector_rows": vector_count,
            "sparse_generation": value.source_sparse_generation,
            "sparse_fields": sparse_count,
        }
    )
    return value.__class__(
        **{
            **value.__dict__,
            "vector_snapshot_digest": vector_digest,
            "sparse_snapshot_digest": sparse_digest,
            "rollback_identity_digest": rollback,
        }
    ), snapshot


def test_capture_and_validate_complete_payload():
    value, snapshot = aligned_preflight()
    payload = capture_rollback_payload(value, snapshot)
    assert payload["vector_rows"][0]["document"] == "one"
    assert payload["sparse_snapshot"]["fields"][0]["text"] == "one"
    assert validate_rollback_payload(value, payload) == payload


def test_payload_tamper_or_generation_change_is_refused():
    value, snapshot = aligned_preflight()
    payload = capture_rollback_payload(value, snapshot)
    payload["vector_rows"][0]["document"] = "changed"
    with pytest.raises(RuntimeError, match="digest"):
        validate_rollback_payload(value, payload)
    snapshot.generation.sequence = 5
    with pytest.raises(RuntimeError, match="generation changed"):
        capture_rollback_payload(value, snapshot)


def test_key_requires_exact_32_bytes_and_repr_hides_material(monkeypatch):
    key = RollbackEncryptionKey("key-1", b"x" * 32)
    assert "xxxxxxxx" not in repr(key)
    with pytest.raises(ValueError, match="32 bytes"):
        RollbackEncryptionKey("key-1", b"short")
    monkeypatch.setenv(
        "MIGRATION_ROLLBACK_KEY_B64",
        base64.b64encode(b"k" * 32).decode("ascii"),
    )
    monkeypatch.setenv("MIGRATION_ROLLBACK_KEY_ID", "key-2026-08")
    loaded = rollback_key_from_environment()
    assert loaded.key_id == "key-2026-08" and loaded.key == b"k" * 32


def test_missing_invalid_or_whitespace_key_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("MIGRATION_ROLLBACK_KEY_B64", raising=False)
    monkeypatch.delenv("MIGRATION_ROLLBACK_KEY_ID", raising=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        rollback_key_from_environment()
    monkeypatch.setenv("MIGRATION_ROLLBACK_KEY_B64", "not-base64")
    monkeypatch.setenv("MIGRATION_ROLLBACK_KEY_ID", "key")
    with pytest.raises(RuntimeError, match="encoding"):
        rollback_key_from_environment()
    monkeypatch.setenv(
        "MIGRATION_ROLLBACK_KEY_B64",
        base64.b64encode(b"k" * 32).decode("ascii") + " ",
    )
    with pytest.raises(RuntimeError, match="canonical"):
        rollback_key_from_environment()
