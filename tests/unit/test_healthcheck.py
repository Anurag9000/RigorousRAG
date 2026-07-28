import json
import os
import sqlite3

import pytest

import tools.healthcheck as healthcheck


class FakeResponse:
    def __init__(self, *, status=200, body=None):
        self.status = status
        self.body = body if body is not None else json.dumps({"status": "ok"}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, amount=-1):
        return self.body if amount < 0 else self.body[:amount]


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return self.response


def test_sqlite_and_directory_checks(tmp_path):
    database = tmp_path / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample(value INTEGER)")
    directory = tmp_path / "storage"
    directory.mkdir()

    assert healthcheck.check_sqlite(database) is True
    assert healthcheck.check_sqlite(tmp_path / "missing.sqlite3") is False
    assert healthcheck.check_writable_directory(directory) is True
    assert not list(directory.iterdir())
    assert healthcheck.check_writable_directory(tmp_path / "missing") is False


def test_symlinked_state_paths_are_not_ready(tmp_path):
    database = tmp_path / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample(value INTEGER)")
    directory = tmp_path / "storage"
    directory.mkdir()
    database_link = tmp_path / "state-link.sqlite3"
    directory_link = tmp_path / "storage-link"
    try:
        os.symlink(database, database_link)
        os.symlink(directory, directory_link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")

    assert healthcheck.check_sqlite(database_link) is False
    assert healthcheck.check_writable_directory(directory_link) is False


def test_http_check_requires_small_loopback_ok_json(monkeypatch):
    opener = FakeOpener(FakeResponse())
    monkeypatch.setattr(healthcheck.urllib.request, "build_opener", lambda *_args: opener)

    assert healthcheck.check_http("http://127.0.0.1/health") is True
    assert len(opener.calls) == 1
    assert opener.calls[0][0].full_url == "http://127.0.0.1/health"

    monkeypatch.setattr(
        healthcheck.urllib.request,
        "build_opener",
        lambda *_args: FakeOpener(FakeResponse(status=503)),
    )
    assert healthcheck.check_http("http://127.0.0.1/health") is False


def test_http_check_rejects_remote_or_oversized_response(monkeypatch):
    calls = []

    def build_opener(*_args):
        calls.append(True)
        return FakeOpener(FakeResponse(body=b"x" * (healthcheck._MAX_HTTP_BYTES + 1)))

    monkeypatch.setattr(healthcheck.urllib.request, "build_opener", build_opener)

    assert healthcheck.check_http("https://example.test/health") is False
    assert calls == []
    assert healthcheck.check_http("http://localhost/health") is False
    assert calls == [True]


def test_malformed_healthcheck_timeout_uses_bounded_default(monkeypatch):
    opener = FakeOpener(FakeResponse())
    monkeypatch.setattr(healthcheck.urllib.request, "build_opener", lambda *_args: opener)

    assert healthcheck.check_http(
        "http://127.0.0.1/health",
        timeout=float("nan"),
    ) is True
    assert opener.calls[0][1] == 3.0
    assert healthcheck._finite_timeout("not-a-number") == 3.0
    assert healthcheck._finite_timeout(float("inf")) == 3.0
    assert healthcheck._finite_timeout(999) == 60.0


def test_run_checks_does_not_crash_on_malformed_timeout(monkeypatch):
    monkeypatch.setenv("HEALTHCHECK_TIMEOUT_SECONDS", "not-a-number")
    captured = []
    monkeypatch.setattr(
        healthcheck,
        "check_http",
        lambda _url, timeout: captured.append(timeout) or True,
    )
    monkeypatch.setattr(healthcheck, "check_sqlite", lambda _path: True)
    monkeypatch.setattr(healthcheck, "check_writable_directory", lambda _path: True)

    results = healthcheck.run_checks()

    assert all(results.values())
    assert captured == [3.0]


def test_main_returns_nonzero_when_any_dependency_is_unready(monkeypatch, capsys):
    monkeypatch.setattr(
        healthcheck,
        "run_checks",
        lambda: {
            "http": True,
            "jobs": True,
            "documents": False,
            "uploads": True,
            "vectors": True,
        },
    )
    assert healthcheck.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_ready"
    assert payload["checks"]["documents"] is False
