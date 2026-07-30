from pathlib import Path

import pytest

from scripts import generate_release_lock, verify_release_lock


def test_verify_release_lock_accepts_pinned_hashed_requirements(tmp_path):
    lock = tmp_path / "runtime-linux-py312.txt"
    lock.write_text(
        """
# generated fixture
alpha==1.2.3 \\
    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \\
    --hash=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
beta==2.0.0 \\
    --hash=sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
""".lstrip(),
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


def test_generate_release_lock_builds_hashed_pip_tools_command(tmp_path, monkeypatch):
    source = tmp_path / "requirements.txt"
    source.write_text("requests>=2,<3\n", encoding="utf-8")
    destination = tmp_path / "locks" / "runtime-linux-py312.txt"
    calls = []

    def fake_run(command, check, env):
        calls.append((command, check, env))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "requests==2.32.4 \\\n"
            "    --hash=sha256:"
            + "a" * 64
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(generate_release_lock.subprocess, "run", fake_run)

    generate_release_lock.generate_lock(
        input_path=source,
        output_path=destination,
        upgrade=True,
    )

    command, check, environment = calls[0]
    assert check is True
    assert "--generate-hashes" in command
    assert "--no-emit-index-url" in command
    assert "--no-emit-trusted-host" in command
    assert "--upgrade" in command
    assert command[-2:] == ["--upgrade"] or command[-1] == "--upgrade"
    assert environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert destination.exists()


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

    with pytest.raises(ValueError, match="symbolic link"):
        generate_release_lock.generate_lock(
            input_path=source,
            output_path=destination,
            upgrade=False,
        )
    assert target.read_text(encoding="utf-8") == "unchanged"
