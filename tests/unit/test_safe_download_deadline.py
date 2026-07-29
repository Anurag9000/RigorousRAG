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
    url = "https://example.com/provider-controlled"

    def __init__(self, chunks=None):
        self.raw = SimpleNamespace(_connection=SimpleNamespace(sock=FakeSocket()))
        self.closed = False
        self._chunks = chunks if chunks is not None else [b"first", b"second"]

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield from self._chunks

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


def _public_dns(monkeypatch):
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


def test_streaming_download_cannot_extend_total_deadline(monkeypatch):
    _public_dns(monkeypatch)
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


def test_empty_chunks_are_also_charged_against_total_deadline(monkeypatch):
    _public_dns(monkeypatch)
    timestamps = iter([0.0, 0.0, 0.2, 1.1])
    monkeypatch.setattr(security.time, "monotonic", lambda: next(timestamps))
    response = FakeResponse(chunks=[b"", b""])

    with pytest.raises(SecurityError, match="time limit"):
        safe_download(
            "https://example.com/data",
            timeout=1.0,
            session=FakeSession(response),
        )

    assert response.closed is True


def test_url_fragments_are_removed_and_response_url_is_not_trusted(monkeypatch):
    _public_dns(monkeypatch)
    response = FakeResponse(chunks=[b"evidence"])
    calls = []
    session = FakeSession(response)
    session.request = lambda **kwargs: calls.append(kwargs) or response

    downloaded = safe_download(
        "https://example.com/data#not-sent",
        session=session,
    )

    assert calls[0]["url"] == "https://example.com/data"
    assert downloaded.final_url == "https://example.com/data"
