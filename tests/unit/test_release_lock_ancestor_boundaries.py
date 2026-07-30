from pathlib import Path

import pytest

from scripts import generate_release_lock


def test_github_output_refuses_symlinked_parent(tmp_path, monkeypatch):
    destination = tmp_path / "runtime-linux-py312.txt"
    destination.write_text("fixture", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-output"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")
    output = linked / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock._write_github_output(destination)
    assert list(outside.iterdir()) == []


def test_pip_compile_executable_refuses_symlinked_script(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = tmp_path / "real-pip-compile"
    target.write_text("fixture", encoding="utf-8")
    executable = scripts / ("pip-compile.exe" if generate_release_lock.os.name == "nt" else "pip-compile")
    try:
        executable.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    monkeypatch.setattr(generate_release_lock.sysconfig, "get_path", lambda name: str(scripts))

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock._pip_compile_executable()
