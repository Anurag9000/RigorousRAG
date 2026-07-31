from pathlib import Path

import pytest

from tools import frontend_static


_REQUIRED = ("index.html", "app.js", "lifecycle.js", "preload.js")


def _write_frontend(root: Path) -> None:
    frontend = root / "frontend"
    frontend.mkdir(parents=True)
    for name in _REQUIRED:
        (frontend / name).write_text("fixture", encoding="utf-8")


def test_frontend_directory_rejects_symlinked_resolver_module(tmp_path, monkeypatch):
    package = tmp_path / "package"
    tools = package / "tools"
    tools.mkdir(parents=True)
    _write_frontend(package)
    target = tmp_path / "real-frontend-static.py"
    target.write_text("fixture", encoding="utf-8")
    resolver = tools / "frontend_static.py"
    try:
        resolver.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("File symlinks are unavailable in this environment.")
    monkeypatch.setattr(frontend_static, "__file__", str(resolver))

    with pytest.raises(RuntimeError, match="regular non-symlink file"):
        frontend_static.frontend_directory()


def test_frontend_directory_rejects_reparse_flagged_asset(tmp_path, monkeypatch):
    package = tmp_path / "package"
    tools = package / "tools"
    tools.mkdir(parents=True)
    resolver = tools / "frontend_static.py"
    resolver.write_text("fixture", encoding="utf-8")
    _write_frontend(package)
    original_lstat = frontend_static.os.lstat
    asset = package / "frontend" / "app.js"

    class ReparseInfo:
        def __init__(self, info):
            self.st_mode = info.st_mode
            self.st_file_attributes = frontend_static._FILE_ATTRIBUTE_REPARSE_POINT

    def fake_lstat(path):
        info = original_lstat(path)
        if Path(path) == asset:
            return ReparseInfo(info)
        return info

    monkeypatch.setattr(frontend_static, "__file__", str(resolver))
    monkeypatch.setattr(frontend_static.os, "lstat", fake_lstat)

    with pytest.raises(RuntimeError, match="regular non-symlink file"):
        frontend_static.frontend_directory()
