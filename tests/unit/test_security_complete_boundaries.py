import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.security as security
from tools.security import (
    Principal,
    SecurityError,
    hostname_matches,
    parse_api_key_owners,
    safe_download,
    safe_upload_suffix,
    validate_public_url,
)


class PublicSocket:
    def getpeername(self):
        return ("93.184.216.34", 443)


class FakeResponse:
    def __init__(
        self,
        *,
        status=200,
        headers=None,
        chunks=None,
        peer=None,
    ):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/plain"}
        self._chunks = chunks if chunks is not None else [b"evidence"]
        socket = peer if peer is not None else PublicSocket()
        self.raw = SimpleNamespace(_connection=SimpleNamespace(sock=socket))
        self.closed = False

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.closed = False

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def _public_dns(*_args, **_kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def test_principal_and_upload_suffix_reject_non_string_inputs():
    with pytest.raises(SecurityError, match="strings"):
        Principal(owner_id=object(), authenticated=True)
    with pytest.raises(SecurityError, match="boolean"):
        Principal(owner_id="alice", authenticated="yes")
    with pytest.raises(SecurityError, match="filenames"):
        safe_upload_suffix(None)
    with pytest.raises(SecurityError, match="500"):
        safe_upload_suffix("x" * 501 + ".pdf")


def test_api_key_configuration_is_strict_bounded_and_nonstandard_json_free(monkeypatch):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", '{"key":NaN}')
    with pytest.raises(RuntimeError, match="valid JSON"):
        parse_api_key_owners()

    monkeypatch.setenv(
        "API_KEY_OWNERS_JSON",
        json.dumps({"x" * 4097: "alice"}),
    )
    with pytest.raises(RuntimeError, match="4096"):
        parse_api_key_owners()

    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"bad\nkey": "alice"}))
    with pytest.raises(RuntimeError, match="valid characters"):
        parse_api_key_owners()


def test_url_validation_rejects_controls_credentials_bad_hosts_and_private_dns(monkeypatch):
    for value in (
        object(),
        "https://alice:password@example.test/",
        "https://not a hostname.test/",
        "https://example.test/\r\nInjected: yes",
        "x" * 9000,
    ):
        with pytest.raises(SecurityError):
            validate_public_url(value)

    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("10.0.0.1", 443))],
    )
    with pytest.raises(SecurityError, match="Private"):
        validate_public_url("https://example.test/")


def test_public_url_canonicalizes_idna_ipv6_and_fragment(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    assert validate_public_url("https://EXAMPLE.test/a#fragment") == "https://example.test/a"
    assert validate_public_url("https://[2606:4700:4700::1111]/a") == (
        "https://[2606:4700:4700::1111]/a"
    )


def test_hostname_matching_is_exact_subdomain_and_iterable_bounded():
    assert hostname_matches("papers.example.test", ["example.test"])
    assert not hostname_matches("example.test.attacker.invalid", ["example.test"])
    assert not hostname_matches("example.test", "example.test")
    assert not hostname_matches("not a hostname", ["example.test"])


def test_safe_download_validates_direct_limits_headers_bodies_and_mime_before_request(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    session = FakeSession([])
    cases = [
        {"max_bytes": True},
        {"max_bytes": 0},
        {"timeout": float("nan")},
        {"method": object()},
        {"headers": {"Host": "attacker"}},
        {"headers": {"X-Test": "bad\r\nInjected: yes"}},
        {"headers": {"X" * 201: "value"}},
        {"data": b"x", "json_body": {}},
        {"data": object()},
        {"json_body": {"score": float("nan")}},
        {"allowed_content_types": "text/plain"},
        {"allowed_content_types": ["not-a-mime"]},
    ]
    for kwargs in cases:
        with pytest.raises((SecurityError, ValueError)):
            safe_download(
                "https://example.test/",
                session=session,
                **kwargs,
            )
    assert session.calls == []


def test_safe_download_bounds_outbound_json_and_requires_declared_allowed_mime(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    with patch.object(security, "MAX_REMOTE_REQUEST_BODY_BYTES", 10):
        with pytest.raises(SecurityError, match="request body"):
            safe_download(
                "https://example.test/",
                json_body={"query": "x" * 100},
                session=FakeSession([]),
            )

    missing_type = FakeResponse(headers={})
    session = FakeSession([missing_type])
    with pytest.raises(SecurityError, match="content type"):
        safe_download(
            "https://example.test/",
            allowed_content_types={"text/plain"},
            session=session,
        )
    assert missing_type.closed is True


def test_redirect_strips_secrets_and_does_not_replay_cross_origin_post(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    redirect = FakeResponse(
        status=302,
        headers={"Location": "https://other.test/final", "Content-Type": "text/plain"},
    )
    final = FakeResponse(headers={"Content-Type": "text/plain"})
    session = FakeSession([redirect, final])

    result = safe_download(
        "https://example.test/start",
        method="POST",
        headers={"Authorization": "Bearer secret", "X-API-Key": "secret"},
        json_body={"query": "evidence"},
        allowed_content_types={"text/plain"},
        session=session,
    )

    assert result.content == b"evidence"
    assert session.calls[0]["method"] == "POST"
    assert session.calls[1]["method"] == "GET"
    assert "Authorization" not in session.calls[1]["headers"]
    assert "X-API-Key" not in session.calls[1]["headers"]
    assert "Content-Type" not in session.calls[1]["headers"]
    assert session.calls[1]["data"] is None


def test_response_chunks_and_headers_are_bounded_and_sensitive_headers_removed(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
            "Set-Cookie": "secret=value",
            **{f"X-{index}": "v" for index in range(300)},
        },
        chunks=[b"a", bytearray(b"b"), memoryview(b"c")],
    )
    result = safe_download(
        "https://example.test/",
        allowed_content_types={"text/plain"},
        session=FakeSession([response]),
    )

    assert result.content == b"abc"
    assert "Set-Cookie" not in result.headers
    assert len(result.headers) <= security._MAX_RESPONSE_HEADERS
    assert response.closed is True


def test_nonbyte_response_chunk_and_private_connected_peer_fail_closed(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    bad_chunk = FakeResponse(chunks=["not bytes"])
    with pytest.raises(SecurityError, match="chunks"):
        safe_download(
            "https://example.test/",
            session=FakeSession([bad_chunk]),
        )

    private_socket = SimpleNamespace(getpeername=lambda: ("127.0.0.1", 443))
    private = FakeResponse(peer=private_socket)
    with pytest.raises(SecurityError, match="private"):
        safe_download(
            "https://example.test/",
            session=FakeSession([private]),
        )
