import json
import sqlite3
from types import SimpleNamespace

import tools.healthcheck as healthcheck


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


def test_http_check_requires_ok_json(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return json.dumps({"status": "ok"}).encode("utf-8")

    monkeypatch.setattr(
        healthcheck.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    assert healthcheck.check_http("http://127.0.0.1/health") is True

    monkeypatch.setattr(
        healthcheck.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: SimpleNamespace(status=503),
    )
    assert healthcheck.check_http("http://127.0.0.1/health") is False


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
