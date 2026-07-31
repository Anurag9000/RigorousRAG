import os
from pathlib import Path

import pytest

from scripts import generate_release_lock, verify_release_lock


def _hashed_lock(package: str = "requests", version: str = "2.32.4") -> str:
    return (
        f"{package}=={version} \\\n"
        "    --hash=sha256:"
        + "a" * 64
        + "\n"
    )


def _write_compiled_output(command, content=None):
    output_index = command.index("--output-file") + 1
    output = Path(command[output_index])
    output.write_text(content or _hashed_lock(), encoding="utf-8")
    return output


def test_verify_release_lock_accepts_pinned_hashed_requirements(tmp_path):
    lock = tmp_path / "runtime-linux-py312.txt"
    lock.write_text(
        "alpha==1.2.3 \\\n"
        "    --hash=sha256:"
        + "a" * 64
        + " \\\n"
        "    --hash=sha256:"
        + "b" * 64
        + "\n"
        "beta==2.0.0 \\\n"
        "    --hash=sha256:"
        + "c" * 64
        + "\n",
        encoding="utf-8",
    )

    result = verify_release_lock.verify_lock(lock)

    assert result == {"requirements": 2, "hashes": 3}


@pytest.mark.parametrize(
    "content, message",
    [
        ("alpha>=1\n", "no SHA-256"),
        (
            "alpha>=1 --hash=sha256:"
            + "a" * 64
            + "\n",
            "not exactly pinned",
        ),
        (
            "--index-url https://packages.invalid/simple\n"
            "alpha==1 --hash=sha256:"
            + "a" * 64
            + "\n",
            "package-index authority",
        ),
        ("alpha==1\n", "no SHA-256"),
    ],
)
def test_verify_release_lock_rejects_unreproducible_files(tmp_path, content, message):
    lock = tmp_path / "bad.txt"
    lock.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_release_lock.verify_lock(lock)


def test_generate_release_lock_uses_staged_compile_only_entry_point(tmp_path, monkeypatch):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    destination = tmp_path / "locks" / "runtime-linux-py312.txt"
    executable = tmp_path / "pip-compile"
    executable.write_text("fixture", encoding="utf-8")
    calls = []

    def fake_run(command, check, env):
        calls.append((command, check, env))
        snapshot = Path(command[1])
        assert snapshot != source
        assert snapshot.read_text(encoding="utf-8") == "requests>=2,<3\n"
        _write_compiled_output(command)

    monkeypatch.setattr(
        generate_release_lock,
        "_pip_compile_executable",
        lambda: executable,
    )
    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://untrusted.invalid/simple")
    monkeypatch.setenv("PIP_TRUSTED_HOST", "untrusted.invalid")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "injected-python"))
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "untrusted-ca.pem"))

    returned = generate_release_lock.generate_lock(
        input_path=source,
        output_path=destination,
        upgrade=True,
    )

    command, check, environment = calls[0]
    assert check is True
    assert command[0] == str(executable)
    assert "-m" not in command
    assert "piptools" not in command
    assert "--generate-hashes" in command
    assert "--allow-unsafe" in command
    assert "--index-url" in command
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--no-emit-index-url" in command
    assert "--no-emit-trusted-host" in command
    assert command[-1] == "--upgrade"
    assert environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert environment["PIP_NO_INPUT"] == "1"
    assert environment["PIP_NO_CACHE_DIR"] == "1"
    assert environment["PIP_KEYRING_PROVIDER"] == "disabled"
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert "PIP_EXTRA_INDEX_URL" not in environment
    assert "PIP_TRUSTED_HOST" not in environment
    assert "PYTHONPATH" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "REQUESTS_CA_BUNDLE" not in environment
    assert returned == destination.resolve()
    assert destination.read_text(encoding="utf-8") == _hashed_lock()
    assert not list(destination.parent.glob(".rigorousrag-lock-*"))


def test_generate_release_lock_snapshot_is_immune_to_source_replacement(tmp_path, monkeypatch):
    source = tmp_path / "requirements.txt"
    source.write_text("alpha>=1\n", encoding="utf-8")
    destination = tmp_path / "locks" / "runtime-linux-py312.txt"
    executable = tmp_path / "pip-compile"
    executable.write_text("fixture", encoding="utf-8")

    def fake_run(command, check, env):
        assert check is True
        assert env["PIP_CONFIG_FILE"] == os.devnull
        source.write_text("attacker-package>=9\n", encoding="utf-8")
        snapshot = Path(command[1])
        assert snapshot.read_text(encoding="utf-8") == "alpha>=1\n"
        _write_compiled_output(command, _hashed_lock("alpha", "1.0"))

    monkeypatch.setattr(generate_release_lock, "_pip_compile_executable", lambda: executable)
    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    returned = generate_release_lock.generate_lock(
        input_path=source,
        output_path=destination,
        upgrade=False,
    )

    assert returned == destination.resolve()
    assert destination.read_text(encoding="utf-8") == _hashed_lock("alpha", "1.0")
    assert source.read_text(encoding="utf-8") == "attacker-package>=9\n"


@pytest.mark.parametrize(
    "directive",
    [
        "--index-url https://packages.invalid/simple\nrequests>=2\n",
        "--extra-index-url=https://packages.invalid/simple\nrequests>=2\n",
        "--trusted-host packages.invalid\nrequests>=2\n",
        "-r other-requirements.txt\n",
        "-c constraints.txt\nrequests>=2\n",
        "-e ../editable-package\n",
        "package @ https://packages.invalid/package.whl\n",
        "../local-package\n",
        "~/local-package\n",
        "\\\\server\\share\\package\n",
    ],
)
def test_generate_release_lock_rejects_external_authority_and_unsnapshotted_inputs(
    tmp_path,
    monkeypatch,
    directive,
):
    source = tmp_path / "requirements.txt"
    source.write_text(directive, encoding="utf-8")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="resolver options|URL or local-path|include another"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=tmp_path / "lock.txt",
            upgrade=False,
        )
    assert called is False


def test_github_output_contains_absolute_lock_path_and_artifact_name(tmp_path, monkeypatch):
    destination = (tmp_path / "locks" / "runtime-linux-py312.txt").resolve()
    destination.parent.mkdir(parents=True)
    destination.write_text("fixture", encoding="utf-8")
    output = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    generate_release_lock._write_github_output(destination)

    assert output.read_text(encoding="utf-8") == (
        f"path={destination.as_posix()}\nname=runtime-linux-py312\n"
    )


def test_github_output_refuses_symlink(tmp_path, monkeypatch):
    destination = tmp_path / "runtime-linux-py312.txt"
    destination.write_text("fixture", encoding="utf-8")
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "github-output.txt"
    try:
        output.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock._write_github_output(destination)
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_github_output_refuses_symlinked_ancestor(tmp_path, monkeypatch):
    destination = tmp_path / "runtime-linux-py312.txt"
    destination.write_text("fixture", encoding="utf-8")
    real = tmp_path / "real-output"
    real.mkdir()
    linked = tmp_path / "linked-output"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")
    monkeypatch.setenv("GITHUB_OUTPUT", str(linked / "github-output.txt"))

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock._write_github_output(destination)
    assert list(real.iterdir()) == []


def test_generate_release_lock_refuses_symlinked_output(tmp_path):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    destination = tmp_path / "lock.txt"
    try:
        destination.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=destination,
            upgrade=False,
        )
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_generate_release_lock_refuses_destination_replaced_during_resolution(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    destination = tmp_path / "locks" / "runtime-linux-py312.txt"
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    executable = tmp_path / "pip-compile"
    executable.write_text("fixture", encoding="utf-8")

    def fake_run(command, check, env):
        assert check is True
        _write_compiled_output(command)
        try:
            destination.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks are unavailable in this environment.")

    monkeypatch.setattr(generate_release_lock, "_pip_compile_executable", lambda: executable)
    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="appeared unexpectedly|identity changed"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=destination,
            upgrade=False,
        )
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_generate_release_lock_refuses_symlinked_generated_output(tmp_path, monkeypatch):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    destination = tmp_path / "locks" / "runtime-linux-py312.txt"
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    executable = tmp_path / "pip-compile"
    executable.write_text("fixture", encoding="utf-8")

    def fake_run(command, check, env):
        output_index = command.index("--output-file") + 1
        generated = Path(command[output_index])
        try:
            generated.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks are unavailable in this environment.")

    monkeypatch.setattr(generate_release_lock, "_pip_compile_executable", lambda: executable)
    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="symbolic-link|safe regular file"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=destination,
            upgrade=False,
        )
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert not destination.exists()


def test_generate_release_lock_refuses_symlinked_input_ancestor(tmp_path):
    real = tmp_path / "real-input"
    real.mkdir()
    source = real / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    linked = tmp_path / "linked-input"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock.generate_lock(
            input_path=linked / "requirements.txt",
            output_path=tmp_path / "lock.txt",
            upgrade=False,
        )


def test_generate_release_lock_refuses_symlinked_output_ancestor(tmp_path):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-output"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=linked / "runtime.txt",
            upgrade=False,
        )
    assert list(outside.iterdir()) == []
