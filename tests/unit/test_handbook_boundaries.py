import os

import pytest

import tools.handbook as handbook


def _reset_cache(monkeypatch):
    monkeypatch.setattr(
        handbook,
        "_CACHE",
        {"signature": None, "index": None, "chunks": None},
    )


def test_long_paragraph_is_split_into_hard_bounded_chunks(monkeypatch):
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_CHUNKS", 20)

    chunks = handbook._paragraph_chunks("x" * 2501)

    assert len(chunks) == 3
    assert all(1 <= len(text) <= handbook._CHUNK_CHARS for _chunk_id, text in chunks)
    assert "".join(text for _chunk_id, text in chunks) == "x" * 2501


def test_chunk_limit_fails_closed(monkeypatch):
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_CHUNKS", 2)

    with pytest.raises(ValueError, match="chunk limit"):
        handbook._paragraph_chunks("x" * 2501)


def test_direct_query_limit_is_checked_before_file_access(monkeypatch):
    monkeypatch.setattr(
        handbook,
        "_read_handbook",
        lambda _path: (_ for _ in ()).throw(AssertionError("file should not be read")),
    )

    with pytest.raises(ValueError, match="2,000"):
        handbook.search_handbook("q" * 2001)


def test_oversized_handbook_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("x" * 101, encoding="utf-8")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_BYTES", 100)
    _reset_cache(monkeypatch)

    with pytest.raises(ValueError, match="byte limit"):
        handbook.search_handbook("policy")


def test_symlinked_handbook_is_refused(monkeypatch, tmp_path):
    target = tmp_path / "target.md"
    target.write_text("privacy policy evidence", encoding="utf-8")
    link = tmp_path / "handbook.md"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", link)
    _reset_cache(monkeypatch)

    with pytest.raises(FileNotFoundError, match="unavailable"):
        handbook.search_handbook("privacy")


def test_changed_handbook_signature_rebuilds_index(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("alpha privacy policy", encoding="utf-8")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    _reset_cache(monkeypatch)

    first = handbook.search_handbook("alpha")
    path.write_text("beta retention policy with additional content", encoding="utf-8")
    second = handbook.search_handbook("beta")

    assert "alpha privacy policy" in first
    assert "beta retention policy" in second
    assert "alpha privacy policy" not in second


def test_same_size_same_mtime_replacement_rebuilds_cache(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("alpha policy", encoding="utf-8")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    _reset_cache(monkeypatch)

    first = handbook.search_handbook("alpha")
    original = path.stat()
    replacement = tmp_path / "replacement.md"
    replacement.write_text("bravo policy", encoding="utf-8")
    assert replacement.stat().st_size == original.st_size
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_mtime_ns == original.st_mtime_ns

    second = handbook.search_handbook("bravo")

    assert "alpha policy" in first
    assert "bravo policy" in second
    assert "alpha policy" not in second
