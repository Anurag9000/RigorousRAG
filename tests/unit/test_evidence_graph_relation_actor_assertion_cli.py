from __future__ import annotations

import json
import stat

from tools import evidence_graph_relation_actor_assertion_cli as cli


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_sign_and_verify_cli_never_return_key_material(
    tmp_path, monkeypatch, capsys
):
    key_path = tmp_path / "actor-key.bin"
    assertion_path = tmp_path / "actor-assertion.json"
    key_path.write_bytes(b"k" * 32)
    monkeypatch.setattr(cli.time, "time", lambda: 100.0)

    assert cli.main([
        "sign",
        "--actor-id", "reviewer-1",
        "--issuer", "review-control-plane",
        "--key-path", str(key_path),
        "--output", str(assertion_path),
        "--lifetime-seconds", "600",
        "--nonce", "nonce-1",
    ]) == 0
    created, error = read(capsys)
    assert error is None
    assert created["status"] == "created"
    assert created["actor_id"] == "reviewer-1"
    assert created["key_material_returned"] is False
    assert assertion_path.exists()
    if hasattr(stat, "S_IMODE"):
        assert stat.S_IMODE(assertion_path.stat().st_mode) & 0o077 == 0
    rendered = json.dumps(created)
    assert "kkkk" not in rendered

    monkeypatch.setattr(
        cli,
        "verify_review_actor_assertion",
        lambda **kwargs: __import__(
            "tools.evidence_graph_relation_actor_assertion",
            fromlist=["verify_review_actor_assertion"],
        ).verify_review_actor_assertion(now=150.0, **kwargs),
    )
    assert cli.main([
        "verify",
        "--assertion-path", str(assertion_path),
        "--key-path", str(key_path),
        "--expected-issuer", "review-control-plane",
    ]) == 0
    verified, error = read(capsys)
    assert error is None
    assert verified["status"] == "valid"
    assert verified["actor_id"] == "reviewer-1"
    assert verified["key_material_returned"] is False
    assert len(verified["assertion_digest"]) == 64
    assert len(verified["signature_digest"]) == 64


def test_sign_refuses_existing_or_concurrent_destination(
    tmp_path, monkeypatch, capsys
):
    key_path = tmp_path / "actor-key.bin"
    assertion_path = tmp_path / "actor-assertion.json"
    key_path.write_bytes(b"k" * 32)
    assertion_path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 100.0)

    assert cli.main([
        "sign",
        "--actor-id", "reviewer-1",
        "--issuer", "review-control-plane",
        "--key-path", str(key_path),
        "--output", str(assertion_path),
        "--lifetime-seconds", "600",
    ]) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert assertion_path.read_text(encoding="utf-8") == "existing"


def test_cli_rejects_weak_keys_and_invalid_lifetimes(
    tmp_path, monkeypatch, capsys
):
    key_path = tmp_path / "weak-key.bin"
    output_path = tmp_path / "assertion.json"
    key_path.write_bytes(b"weak")
    monkeypatch.setattr(cli.time, "time", lambda: 100.0)

    assert cli.main([
        "sign",
        "--actor-id", "reviewer-1",
        "--issuer", "review-control-plane",
        "--key-path", str(key_path),
        "--output", str(output_path),
        "--lifetime-seconds", "30",
    ]) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert not output_path.exists()
