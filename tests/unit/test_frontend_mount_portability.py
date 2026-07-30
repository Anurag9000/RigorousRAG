import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.staticfiles import StaticFiles

from tools.frontend_static import frontend_directory, install_portable_frontend_staticfiles


def test_legacy_frontend_sentinel_is_independent_of_working_directory(tmp_path, monkeypatch):
    install_portable_frontend_staticfiles()
    monkeypatch.chdir(tmp_path)

    mounted = StaticFiles(directory="frontend", html=True)

    assert Path(mounted.directory) == frontend_directory()
    assert (Path(mounted.directory) / "index.html").is_file()


def test_nonlegacy_static_directory_keeps_normal_semantics(tmp_path):
    install_portable_frontend_staticfiles()
    custom = tmp_path / "custom-assets"
    custom.mkdir()

    mounted = StaticFiles(directory=custom, html=True)

    assert Path(mounted.directory) == custom


def test_frontend_directory_rejects_missing_bundled_assets(tmp_path, monkeypatch):
    fake_tools = tmp_path / "package" / "tools"
    fake_tools.mkdir(parents=True)
    fake_module = fake_tools / "frontend_static.py"
    fake_module.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr("tools.frontend_static.__file__", str(fake_module))

    with pytest.raises(RuntimeError, match="frontend"):
        frontend_directory()


def test_server_import_mounts_frontend_from_arbitrary_cwd(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository_root),
            "SINGLE_USER_OWNER_ID": "frontend-test-user",
            "UPLOAD_DIR": str(state / "uploads"),
            "CHROMA_PATH": str(state / "vectors"),
            "JOB_DB_PATH": str(state / "jobs.sqlite3"),
            "DOCUMENT_DB_PATH": str(state / "documents.sqlite3"),
            "CLASSIC_STORAGE_DIR": str(state / "classic"),
            "USAGE_LOG_FILE": str(state / "usage.jsonl"),
            "ORPHAN_CLEANUP_ON_STARTUP": "false",
            "ENABLE_OCR": "false",
        }
    )
    code = """
from pathlib import Path
import server
mount = next(route for route in server.app.routes if getattr(route, 'name', None) == 'static')
expected = Path(server.__file__).resolve().parent / 'frontend'
assert Path(mount.app.directory) == expected
assert (Path(mount.app.directory) / 'index.html').is_file()
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
