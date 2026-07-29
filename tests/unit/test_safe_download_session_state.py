import threading
import time
from types import SimpleNamespace

import pytest
import requests

import tools.security as security
from tools.security import SecurityError, safe_download


class PublicSocket:
    def getpeername(self):
        return ("93.184.216.34", 443)


class Response:
    status_code = 200
    headers = {"Content-Type": "text/plain"}

    def __init__(self, *, entered=None, release=None, fail=False):
        self.raw = SimpleNamespace(
            _connection=SimpleNamespace(sock=PublicSocket())
        )
        self.entered = entered
        self.release = release
        self.fail = fail
        self.closed = False

    def raise_for_status(self):
        if self.fail:
            raise RuntimeError("provider failed")

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(2.0)
        yield b"evidence"

    def close(self):
        self.closed = True


class StatefulSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.proxies = {"https": "http://proxy.example:8080"}
        self.auth = ("alice", "secret")
        self.headers = requests.structures.CaseInsensitiveDict(
            {
                "Authorization": "Bearer ambient-secret",
                "X-Ambient": "ambient-value",
            }
        )
        self.cookies = requests.cookies.RequestsCookieJar()
        self.cookies.set("session", "ambient-cookie")
        self.params = {"ambient": "secret"}
        self.hooks = {"response": [lambda response, *_args, **_kwargs: response]}
        self.verify = False
        self.cert = ("client-cert.pem", "client-key.pem")
        self.snapshots = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        self.snapshots.append(
            {
                "trust_env": self.trust_env,
                "proxies": dict(self.proxies),
                "auth": self.auth,
                "headers": dict(self.headers),
                "cookies": self.cookies.get_dict(),
                "params": dict(self.params),
                "hooks": dict(self.hooks),
                "verify": self.verify,
                "cert": self.cert,
            }
        )
        return self.responses.pop(0)

    def close(self):
        raise AssertionError("Injected sessions must not be closed")


def _public_dns(monkeypatch):
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    )


def _original_state(session):
    return {
        "trust_env": session.trust_env,
        "proxies": session.proxies,
        "auth": session.auth,
        "headers": session.headers,
        "cookies": session.cookies,
        "params": session.params,
        "hooks": session.hooks,
        "verify": session.verify,
        "cert": session.cert,
    }


def _assert_neutral(snapshot):
    assert snapshot == {
        "trust_env": False,
        "proxies": {},
        "auth": None,
        "headers": {},
        "cookies": {},
        "params": {},
        "hooks": {"response": []},
        "verify": True,
        "cert": None,
    }


def _assert_restored(session, original):
    for name, value in original.items():
        assert getattr(session, name) is value


def test_injected_session_ambient_authority_is_neutralized_and_restored(monkeypatch):
    _public_dns(monkeypatch)
    response = Response()
    session = StatefulSession([response])
    original = _original_state(session)

    result = safe_download(
        "https://example.test/resource",
        headers={"Authorization": "Bearer explicit-provider-key"},
        session=session,
        allowed_content_types={"text/plain"},
    )

    assert result.content == b"evidence"
    assert response.closed is True
    _assert_neutral(session.snapshots[0])
    assert session.calls[0]["headers"] == {
        "Authorization": "Bearer explicit-provider-key"
    }
    _assert_restored(session, original)


def test_injected_session_state_is_restored_after_request_failure(monkeypatch):
    _public_dns(monkeypatch)
    response = Response(fail=True)
    session = StatefulSession([response])
    original = _original_state(session)

    with pytest.raises(RuntimeError, match="provider failed"):
        safe_download("https://example.test/resource", session=session)

    _assert_neutral(session.snapshots[0])
    _assert_restored(session, original)
    assert response.closed is True


def test_shared_session_cannot_restore_proxy_or_credentials_mid_download(monkeypatch):
    _public_dns(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    session = StatefulSession(
        [
            Response(entered=entered, release=release),
            Response(),
        ]
    )
    original = _original_state(session)
    results = []
    errors = []

    def download(path):
        try:
            results.append(
                safe_download(
                    f"https://example.test/{path}",
                    session=session,
                ).content
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    first = threading.Thread(target=download, args=("first",))
    second = threading.Thread(target=download, args=("second",))
    first.start()
    assert entered.wait(1.0)
    _assert_neutral(
        {
            "trust_env": session.trust_env,
            "proxies": dict(session.proxies),
            "auth": session.auth,
            "headers": dict(session.headers),
            "cookies": session.cookies.get_dict(),
            "params": dict(session.params),
            "hooks": dict(session.hooks),
            "verify": session.verify,
            "cert": session.cert,
        }
    )

    second.start()
    time.sleep(0.05)
    assert len(session.calls) == 1
    assert session.proxies == {}
    assert session.auth is None
    assert session.cookies.get_dict() == {}

    release.set()
    first.join(2.0)
    second.join(2.0)

    assert errors == []
    assert sorted(results) == [b"evidence", b"evidence"]
    assert len(session.snapshots) == 2
    assert all(snapshot["proxies"] == {} for snapshot in session.snapshots)
    assert all(snapshot["auth"] is None for snapshot in session.snapshots)
    _assert_restored(session, original)
