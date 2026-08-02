from __future__ import annotations

import json

import pytest

from tools.evidence_graph_relation_actor import load_relation_review_actor
from tools.evidence_graph_relation_actor_assertion import (
    sign_review_actor_assertion,
    verify_review_actor_assertion,
)

KEY = b"k" * 32


def write_assertion(tmp_path, *, key=KEY, **overrides):
    values = {
        "actor_id": "reviewer-1",
        "issuer": "review-control-plane",
        "issued_at": 100.0,
        "expires_at": 200.0,
        "nonce": "nonce-1",
        "key": key,
    }
    values.update(overrides)
    payload = sign_review_actor_assertion(**values)
    assertion_path = tmp_path / "actor-assertion.json"
    key_path = tmp_path / "actor-key.bin"
    assertion_path.write_text(json.dumps(payload), encoding="utf-8")
    key_path.write_bytes(key)
    return assertion_path, key_path, payload


def test_valid_signed_assertion_and_actor_binding(tmp_path):
    assertion_path, key_path, payload = write_assertion(tmp_path)

    assertion = verify_review_actor_assertion(
        assertion_path=assertion_path,
        key_path=key_path,
        expected_issuer="review-control-plane",
        now=150.0,
    )
    binding = load_relation_review_actor(
        assertion_path=assertion_path,
        key_path=key_path,
        expected_issuer="review-control-plane",
        loaded_at=150.0,
    )

    assert assertion.actor_id == payload["actor_id"]
    assert assertion.issuer == payload["issuer"]
    assert assertion.expires_at == 200.0
    assert len(assertion.assertion_digest) == 64
    assert len(assertion.signature_digest) == 64
    assert binding.actor_id == "reviewer-1"
    assert binding.binding_method == "hmac_assertion"
    assert binding.assertion_digest == assertion.assertion_digest
    assert binding.issuer == "review-control-plane"
    assert binding.expires_at == 200.0
    assert len(binding.binding_digest) == 64


def test_signature_tamper_and_wrong_key_fail_closed(tmp_path):
    assertion_path, key_path, payload = write_assertion(tmp_path)
    payload["actor_id"] = "reviewer-2"
    assertion_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="signature"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=150.0,
        )

    assertion_path, key_path, _payload = write_assertion(tmp_path)
    key_path.write_bytes(b"z" * 32)
    with pytest.raises(PermissionError, match="signature"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=150.0,
        )


def test_time_issuer_lifetime_and_key_bounds(tmp_path):
    assertion_path, key_path, _payload = write_assertion(tmp_path)
    with pytest.raises(PermissionError, match="expired"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=400.0,
            clock_skew_seconds=0.0,
        )
    with pytest.raises(PermissionError, match="future"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=0.0,
            clock_skew_seconds=0.0,
        )
    with pytest.raises(PermissionError, match="issuer"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="other-issuer",
            now=150.0,
        )
    with pytest.raises(ValueError, match="clock skew"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=150.0,
            clock_skew_seconds=301.0,
        )
    with pytest.raises(ValueError, match="lifetime"):
        sign_review_actor_assertion(
            actor_id="reviewer-1",
            issuer="review-control-plane",
            issued_at=0.0,
            expires_at=86_401.0,
            nonce="nonce",
            key=KEY,
        )
    with pytest.raises(ValueError, match="key"):
        sign_review_actor_assertion(
            actor_id="reviewer-1",
            issuer="review-control-plane",
            issued_at=0.0,
            expires_at=10.0,
            nonce="nonce",
            key=b"weak",
        )


def test_strict_schema_duplicate_keys_and_signed_source_requirements(tmp_path):
    assertion_path, key_path, payload = write_assertion(tmp_path)
    duplicate = (
        '{"schema_version":1,"actor_id":"reviewer-1",'
        '"actor_id":"reviewer-2","issuer":"review-control-plane",'
        '"issued_at":100.0,"expires_at":200.0,"nonce":"nonce-1",'
        f'"signature":"{payload["signature"]}"}}'
    )
    assertion_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer="review-control-plane",
            now=150.0,
        )

    assertion_path, key_path, _payload = write_assertion(tmp_path)
    with pytest.raises(RuntimeError, match="requires key path"):
        load_relation_review_actor(
            assertion_path=assertion_path,
            expected_issuer="review-control-plane",
            loaded_at=150.0,
        )
    with pytest.raises(RuntimeError, match="requires key path"):
        load_relation_review_actor(
            assertion_path=assertion_path,
            key_path=key_path,
            loaded_at=150.0,
        )
    with pytest.raises(RuntimeError, match="without an assertion"):
        load_relation_review_actor(
            actor_id="reviewer-1",
            key_path=key_path,
            expected_issuer="review-control-plane",
            loaded_at=150.0,
        )
