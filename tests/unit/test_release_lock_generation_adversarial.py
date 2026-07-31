import os
from pathlib import Path

import pytest

from scripts import generate_release_lock


def test_subprocess_environment_strips_case_variant_authority(monkeypatch):
    monkeypatch.setenv("PythonPath", "/tmp/injected")
    monkeypatch.setenv("https_proxy", "https://proxy.invalid")
    monkeypatch.setenv("Ssl_Cert_File", "/tmp/untrusted-ca.pem")
    monkeypatch.setenv("PiP_Extra_Index_Url", "https://packages.invalid/simple")

    environment = generate_release_lock._safe_subprocess_environment()

    normalized = {key.upper() for key in environment}
    assert "PYTHONPATH" not in normalized
    assert "HTTPS_PROXY" not in normalized
    assert "SSL_CERT_FILE" not in normalized
    assert "PIP_EXTRA_INDEX_URL" not in normalized
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PIP_KEYRING_PROVIDER"] == "disabled"
    assert environment["PIP_NO_CACHE_DIR"] == "1"


def test_github_output_nested_under_symlink_does_not_create_outside_directory(
    tmp_path,
    monkeypatch,
):
    lock = tmp_path / "runtime-linux-py312.txt"
    lock.write_text("fixture", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")
    monkeypatch.setenv(
        "GITHUB_OUTPUT",
        str(linked / "must-not-be-created" / "github-output.txt"),
    )

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock._write_github_output(lock)

    assert not (outside / "must-not-be-created").exists()


def test_requirements_source_rejects_relative_artifact_path():
    with pytest.raises(ValueError, match="URL or local-path"):
        generate_release_lock._validate_requirements_source(
            b"dist/local-package-1.0-py3-none-any.whl\n"
        )


def test_github_output_append_rejects_existing_oversized_file(tmp_path):
    output = tmp_path / "github-output.txt"
    output.write_bytes(b"x" * generate_release_lock._MAX_GITHUB_OUTPUT_BYTES)

    with pytest.raises(ValueError, match="1 MB"):
        generate_release_lock._append_github_output(output, b"path=x\n")


def test_existing_lock_replacement_during_resolution_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "requirements.txt"
    source.write_text("alpha>=1\n", encoding="utf-8")
    destination = tmp_path / "runtime-linux-py312.txt"
    destination.write_text("previous", encoding="utf-8")
    executable = tmp_path / "pip-compile"
    executable.write_text("fixture", encoding="utf-8")

    def fake_run(command, check, env):
        assert check is True
        destination.unlink()
        destination.write_text("replacement", encoding="utf-8")
        generated = Path(command[command.index("--output-file") + 1])
        generated.write_text(
            "alpha==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(generate_release_lock, "_pip_compile_executable", lambda: executable)
    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="identity changed"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=destination,
            upgrade=False,
        )

    assert destination.read_text(encoding="utf-8") == "replacement"
