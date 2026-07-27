import ipaddress
import json
import socket

import pytest

from tools.security import (
    SecurityError,
    hostname_matches,
    normalize_owner_id,
    parse_api_key_owners,
    safe_upload_suffix,
    validate_public_url,
)


def test_owner_identifier_validation():
    assert normalize_owner_id("team.alpha-1") == "team.alpha-1"
    for value in ("", "../other", "owner/child", " space", "a" * 129):
        with pytest.raises(SecurityError):
            normalize_owner_id(value)


def test_api_key_identity_mapping(monkeypatch):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"secret-a": "alice", "secret-b": "bob"}))
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    assert parse_api_key_owners() == {"secret-a": "alice", "secret-b": "bob"}


def test_legacy_keys_get_distinct_server_owned_principals(monkeypatch):
    monkeypatch.delenv("API_KEY_OWNERS_JSON", raising=False)
    monkeypatch.setenv("ALLOWED_API_KEYS", "alpha,beta")
    mapping = parse_api_key_owners()
    assert set(mapping) == {"alpha", "beta"}
    assert mapping["alpha"] != mapping["beta"]
    assert mapping["alpha"].startswith("api-")


def test_upload_suffix_is_allowlisted():
    assert safe_upload_suffix("paper.PDF") == ".pdf"
    for filename in ("payload.exe", "archive.pdf.exe", "no-extension"):
        with pytest.raises(SecurityError):
            safe_upload_suffix(filename)


def test_private_and_local_destinations_are_blocked(monkeypatch):
    def private_lookup(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.4", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", private_lookup)
    with pytest.raises(SecurityError):
        validate_public_url("https://internal.example/resource")
    for url in (
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost:8000",
        "file:///etc/passwd",
    ):
        with pytest.raises(SecurityError):
            validate_public_url(url)


def test_public_destination_and_hostname_matching(monkeypatch):
    def public_lookup(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_lookup)
    assert validate_public_url("https://example.com/paper") == "https://example.com/paper"
    assert hostname_matches("journals.example.com", ["example.com"])
    assert not hostname_matches("example.com.attacker.test", ["example.com"])
