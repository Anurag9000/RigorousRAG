from types import SimpleNamespace

import pytest

import tools.security as security
from tools.security import SecurityError, safe_download


class FakeSocket:
    def getpeername(self):
        return ("93.184.216.34", 443)


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "text/plain"}
    url = "https://example.com/final"

    def __init__(self):
        self.raw = SimpleNamespace(_connection=SimpleNamespace(sock=FakeSocket()))
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield b"first"
        yield b"second"

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.trust_env = True

    def request(self, **_kwargs):
        return self.response

    def close(self):
        return None


def test_streaming_download_cannot_extend_total_deadline(monkeypatch):
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    timestamps = iter([0.0, 0.0, 0.4, 1.1])
    monkeypatch.setattr(security.time, "monotonic", lambda: next(timestamps))
    response = FakeResponse()

    with pytest.raises(SecurityError, match="time limit"):
        safe_download(
            "https://example.com/data",
            timeout=1.0,
            session=FakeSession(response),
        )

    assert response.closed is True


def test_url_fragments_are_removed_before_request(monkeypatch):
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    response = FakeResponse()
    calls = []
    session = FakeSession(response)
    session.request = lambda **kwargs: calls.append(kwargs) or response

    downloaded = safe_download(
        "https://example.com/data#not-sent",
        session=session,
    )

    assert calls[0]["url"] == "https://example.com/data"
    assert downloaded.final_url == "https://example.com/final"
