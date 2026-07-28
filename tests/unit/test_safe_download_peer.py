from types import SimpleNamespace

import pytest

from tools.security import SecurityError, safe_download


class FakeSocket:
    def __init__(self, host):
        self.host = host

    def getpeername(self):
        return (self.host, 443)


class FakeResponse:
    def __init__(
        self,
        *,
        peer="93.184.216.34",
        status=200,
        headers=None,
        url="https://example.com/final",
        chunks=(b"ok",),
        include_socket=True,
    ):
        connection = SimpleNamespace(sock=FakeSocket(peer)) if include_socket else None
        self.raw = SimpleNamespace(_connection=connection)
        self.status_code = status
        self.headers = dict(headers or {"Content-Type": "text/plain"})
        self.url = url
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield from self.chunks

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


def _public_dns(monkeypatch):
    monkeypatch.setattr(
        "tools.security.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    )


def test_private_connected_peer_is_blocked_after_public_dns_validation(monkeypatch):
    _public_dns(monkeypatch)
    response = FakeResponse(peer="127.0.0.1")
    session = FakeSession([response])

    with pytest.raises(SecurityError, match="connected remote address"):
        safe_download("https://example.com/data", session=session)

    assert response.closed is True
    assert session.trust_env is True


def test_unverifiable_connected_peer_fails_closed(monkeypatch):
    _public_dns(monkeypatch)
    response = FakeResponse(include_socket=False)
    session = FakeSession([response])

    with pytest.raises(SecurityError, match="Could not verify"):
        safe_download("https://example.com/data", session=session)

    assert response.closed is True


def test_post_body_is_not_replayed_after_302_redirect(monkeypatch):
    _public_dns(monkeypatch)
    redirect = FakeResponse(
        status=302,
        headers={"Location": "/final"},
        url="https://example.com/start",
        chunks=(),
    )
    final = FakeResponse(url="https://example.com/final", chunks=(b"evidence",))
    session = FakeSession([redirect, final])

    downloaded = safe_download(
        "https://example.com/start",
        method="POST",
        data=b"secret-request-body",
        session=session,
        allowed_content_types={"text/plain"},
    )

    assert downloaded.content == b"evidence"
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["data"] == b"secret-request-body"
    assert session.calls[1]["method"] == "GET"
    assert session.calls[1]["data"] is None
    assert session.calls[1]["json"] is None
    assert redirect.closed is True
    assert final.closed is True
    assert session.trust_env is True


def test_remote_method_allowlist_rejects_connect(monkeypatch):
    _public_dns(monkeypatch)
    with pytest.raises(SecurityError, match="not allowed"):
        safe_download("https://example.com", method="CONNECT", session=FakeSession([]))
