import threading
import time
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


class BlockingResponse(FakeResponse):
    def __init__(self, entered, release, **kwargs):
        super().__init__(**kwargs)
        self.entered = entered
        self.release = release

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        self.entered.set()
        assert self.release.wait(2.0)
        yield from self.chunks


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


def test_shared_injected_session_cannot_restore_proxies_mid_download(monkeypatch):
    _public_dns(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    first = BlockingResponse(
        entered,
        release,
        url="https://example.com/first",
        chunks=(b"first",),
    )
    second = FakeResponse(url="https://example.com/second", chunks=(b"second",))
    session = FakeSession([first, second])
    results = []
    errors = []

    def download(path):
        try:
            results.append(
                safe_download(f"https://example.com/{path}", session=session).content
            )
        except Exception as exc:  # pragma: no cover - assertion captures unexpected errors
            errors.append(exc)

    first_thread = threading.Thread(target=download, args=("first",))
    second_thread = threading.Thread(target=download, args=("second",))
    first_thread.start()
    assert entered.wait(1.0)
    assert session.trust_env is False

    second_thread.start()
    time.sleep(0.05)
    assert len(session.calls) == 1
    assert session.trust_env is False

    release.set()
    first_thread.join(2.0)
    second_thread.join(2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert sorted(results) == [b"first", b"second"]
    assert len(session.calls) == 2
    assert session.trust_env is True


def test_cross_origin_redirect_strips_api_keys_and_tokens(monkeypatch):
    _public_dns(monkeypatch)
    redirect = FakeResponse(
        status=302,
        headers={"Location": "https://other.example/final"},
        url="https://example.com/start",
        chunks=(),
    )
    final = FakeResponse(url="https://other.example/final", chunks=(b"ok",))
    session = FakeSession([redirect, final])

    safe_download(
        "https://example.com/start",
        headers={
            "Authorization": "Bearer secret",
            "X-API-KEY": "provider-secret",
            "X-Custom-Token": "another-secret",
            "Accept": "text/plain",
        },
        session=session,
    )

    first_headers = session.calls[0]["headers"]
    second_headers = session.calls[1]["headers"]
    assert first_headers["Authorization"] == "Bearer secret"
    assert "Authorization" not in second_headers
    assert "X-API-KEY" not in second_headers
    assert "X-Custom-Token" not in second_headers
    assert second_headers["Accept"] == "text/plain"


def test_cross_origin_307_does_not_replay_post_body(monkeypatch):
    _public_dns(monkeypatch)
    redirect = FakeResponse(
        status=307,
        headers={"Location": "https://other.example/final"},
        url="https://example.com/start",
        chunks=(),
    )
    session = FakeSession([redirect])

    with pytest.raises(SecurityError, match="may not replay"):
        safe_download(
            "https://example.com/start",
            method="POST",
            data=b"secret-request-body",
            session=session,
        )

    assert len(session.calls) == 1
    assert redirect.closed is True


def test_caller_cannot_override_host_or_hop_by_hop_headers(monkeypatch):
    _public_dns(monkeypatch)
    for header in ("Host", "Content-Length", "Transfer-Encoding", "Connection"):
        with pytest.raises(SecurityError, match="not allowed"):
            safe_download(
                "https://example.com",
                headers={header: "attacker-controlled"},
                session=FakeSession([]),
            )


def test_remote_method_allowlist_rejects_connect(monkeypatch):
    _public_dns(monkeypatch)
    with pytest.raises(SecurityError, match="not allowed"):
        safe_download("https://example.com", method="CONNECT", session=FakeSession([]))
