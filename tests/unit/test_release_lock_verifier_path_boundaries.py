from pathlib import Path

import pytest

from scripts import verify_release_lock


def _valid_lock_text() -> str:
    return (
        "alpha==1.2.3 \\\n"
        "    --hash=sha256:"
        + "a" * 64
        + "\n"
    )


def test_lock_verifier_refuses_symlinked_ancestor(tmp_path):
    real = tmp_path / "real-locks"
    real.mkdir()
    lock = real / "runtime-linux-py312.txt"
    lock.write_text(_valid_lock_text(), encoding="utf-8")
    linked = tmp_path / "linked-locks"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link"):
        verify_release_lock.verify_lock(linked / lock.name)

    assert lock.read_text(encoding="utf-8") == _valid_lock_text()


def test_lock_verifier_refuses_final_symlink_without_touching_target(tmp_path):
    target = tmp_path / "real-lock.txt"
    target.write_text(_valid_lock_text(), encoding="utf-8")
    link = tmp_path / "runtime-linux-py312.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("File symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link"):
        verify_release_lock.verify_lock(link)

    assert target.read_text(encoding="utf-8") == _valid_lock_text()


def test_lock_verifier_rejects_non_utf8_bytes(tmp_path):
    lock = tmp_path / "runtime-linux-py312.txt"
    lock.write_bytes(b"alpha==1.2.3 \\\n    --hash=sha256:" + b"a" * 64 + b"\n\xff")

    with pytest.raises(ValueError, match="UTF-8"):
        verify_release_lock.verify_lock(lock)
