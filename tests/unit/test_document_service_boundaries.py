import hashlib
import os

import pytest

from tools.document_service import _bounded_source_sha256


def test_bounded_source_hash_matches_regular_file(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")

    digest = _bounded_source_sha256(source, max_bytes=100)

    assert digest == hashlib.sha256(b"evidence").hexdigest()


def test_bounded_source_hash_rejects_oversized_replacement(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"x" * 101)

    with pytest.raises(ValueError, match="byte limit"):
        _bounded_source_sha256(source, max_bytes=100)


def test_bounded_source_hash_refuses_symlink_swap(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"private evidence")
    link = tmp_path / "paper.txt"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")

    with pytest.raises(ValueError, match="unavailable"):
        _bounded_source_sha256(link, max_bytes=100)


def test_bounded_source_hash_rejects_invalid_limit(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")

    with pytest.raises(ValueError, match="positive"):
        _bounded_source_sha256(source, max_bytes=0)
