import importlib
import json
import sys

import pytest


def _clear_server_modules():
    for name in ("server", "server_app"):
        sys.modules.pop(name, None)


def _base_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"alice-key": "alice"}))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "vectors"))
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def _shutdown(module):
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    _clear_server_modules()


def test_malformed_numeric_environment_imports_with_bounded_defaults(monkeypatch, tmp_path):
    _base_environment(monkeypatch, tmp_path)
    malformed = {
        "MAX_UPLOAD_BYTES": "not-an-integer",
        "MAX_REMOTE_REDIRECTS": "nan",
        "JOB_TTL_SECONDS": "-999",
        "REQUESTS_PER_MINUTE": "infinity",
        "QUERY_WORKERS": "0",
        "QUERY_MAX_PENDING": "-1",
        "QUERY_TIMEOUT_SECONDS": "nan",
        "INGEST_WORKERS": "999999",
        "INGEST_MAX_ATTEMPTS": "bad",
        "INGEST_MAX_PENDING": "bad",
        "INGEST_ADMISSION_RETRY_SECONDS": "inf",
        "MAX_REQUEST_BODY_BYTES": "bad",
        "MAX_DOCX_COMPRESSION_RATIO": "nan",
        "PORT": "999999",
    }
    for name, value in malformed.items():
        monkeypatch.setenv(name, value)
    _clear_server_modules()

    module = importlib.import_module("server")
    try:
        assert module.DEFAULT_MAX_UPLOAD_BYTES == 50_000_000
        assert module.JOB_TTL_SECONDS == 60
        assert module.REQUESTS_PER_MINUTE == 60
        assert module.QUERY_WORKERS == 1
        assert module.QUERY_MAX_PENDING >= module.QUERY_WORKERS
        assert module.QUERY_TIMEOUT_SECONDS == 120.0
        assert module.INGEST_WORKERS == 16
        assert module.INGEST_MAX_ATTEMPTS == 3
        assert module.INGEST_MAX_PENDING >= module.INGEST_WORKERS
        assert module.INGEST_ADMISSION_RETRY_SECONDS == 1.0
        assert module.MAX_REQUEST_BODY_BYTES >= module.DEFAULT_MAX_UPLOAD_BYTES
        assert int(module.os.environ["PORT"]) == 65_535
        assert module.app.version == "4.4.0"
    finally:
        _shutdown(module)


def test_request_body_limit_is_never_below_upload_limit(monkeypatch, tmp_path):
    _base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "123456")
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1")
    _clear_server_modules()

    module = importlib.import_module("server")
    try:
        assert module.DEFAULT_MAX_UPLOAD_BYTES == 123_456
        assert module.MAX_REQUEST_BODY_BYTES == 123_456
        assert module.os.environ["MAX_REQUEST_BODY_BYTES"] == "123456"
    finally:
        _shutdown(module)


def test_symlinked_upload_root_is_rejected_before_server_app_import(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "uploads"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    _base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(link))
    _clear_server_modules()

    with pytest.raises(RuntimeError, match="UPLOAD_DIR"):
        importlib.import_module("server")

    assert "server_app" not in sys.modules
    assert list(outside.iterdir()) == []
    _clear_server_modules()
