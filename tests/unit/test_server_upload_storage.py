import importlib
import io
import json
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"alice-key": "alice"}))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "1000")
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    sys.modules.pop("server", None)


def test_ingest_refuses_symlinked_owner_directory_without_outside_write(
    server_module,
    tmp_path,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    owner_dir = server_module.UPLOAD_DIR / "alice"
    try:
        owner_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with TestClient(server_module.app) as client:
        response = client.post(
            "/ingest",
            headers={"X-API-Key": "alice-key"},
            files={"file": ("paper.txt", io.BytesIO(b"evidence"), "text/plain")},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Upload storage is unavailable."}
    assert list(outside.iterdir()) == []


def test_safe_unlink_cannot_follow_owner_directory_swap(server_module, tmp_path):
    owner_dir = server_module.UPLOAD_DIR / "alice"
    owner_dir.mkdir()
    inside = owner_dir / "stored.txt"
    inside.write_bytes(b"inside")
    original_owner = server_module.UPLOAD_DIR / "alice-original"
    owner_dir.rename(original_owner)

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / inside.name
    outside_file.write_bytes(b"outside")
    try:
        owner_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        original_owner.rename(owner_dir)
        pytest.skip("Symlinks are unavailable in this environment.")

    assert server_module._safe_unlink_upload(inside) is False
    assert outside_file.read_bytes() == b"outside"
    assert (original_owner / inside.name).read_bytes() == b"inside"
