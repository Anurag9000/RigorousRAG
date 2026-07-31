import os
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

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


def test_chunk_limit_and_non_text_content_fail_closed(monkeypatch):
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_CHUNKS", 2)

    with pytest.raises(ValueError, match="chunk limit"):
        handbook._paragraph_chunks("x" * 2501)
    with pytest.raises(ValueError, match="must be text"):
        handbook._paragraph_chunks(object())


def test_direct_query_limit_and_type_are_checked_before_file_access(monkeypatch):
    monkeypatch.setattr(
        handbook,
        "_read_handbook",
        lambda _path: (_ for _ in ()).throw(AssertionError("file should not be read")),
    )

    for query in (
        object(),
        "",
        "   ",
        "q" * 2001,
        "bad\x00query",
        "bad\nquery",
        "bad\rquery",
        "bad\tquery",
        "bad\x7fquery",
    ):
        with pytest.raises(ValueError):
            handbook.search_handbook(query)


def test_oversized_handbook_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("x" * 101, encoding="utf-8")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_BYTES", 100)
    _reset_cache(monkeypatch)

    with pytest.raises(ValueError, match="byte limit"):
        handbook.search_handbook("policy")


def test_symlinked_handbook_and_parent_are_refused(monkeypatch, tmp_path):
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

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "handbook.md").write_text("policy", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform.")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", linked_parent / "handbook.md")
    _reset_cache(monkeypatch)

    with pytest.raises(FileNotFoundError, match="unavailable"):
        handbook.search_handbook("policy")


def test_reparse_flagged_handbook_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("privacy policy evidence", encoding="utf-8")
    original_lstat = handbook.os.lstat

    class ReparseInfo:
        def __init__(self, info):
            self.st_mode = info.st_mode
            self.st_dev = info.st_dev
            self.st_ino = info.st_ino
            self.st_ctime_ns = info.st_ctime_ns
            self.st_mtime_ns = info.st_mtime_ns
            self.st_size = info.st_size
            self.st_file_attributes = handbook._FILE_ATTRIBUTE_REPARSE_POINT

    def fake_lstat(value):
        info = original_lstat(value)
        return ReparseInfo(info) if Path(value) == path else info

    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    monkeypatch.setattr(handbook.os, "lstat", fake_lstat)
    _reset_cache(monkeypatch)

    with pytest.raises(FileNotFoundError, match="unavailable"):
        handbook.search_handbook("privacy")


def test_fifo_handbook_is_refused_without_blocking(monkeypatch, tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform.")
    path = tmp_path / "handbook.md"
    os.mkfifo(path)
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    _reset_cache(monkeypatch)

    with pytest.raises(ValueError, match="regular file"):
        handbook.search_handbook("policy")


def test_invalid_utf8_handbook_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_bytes(b"\xff\xfe")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    _reset_cache(monkeypatch)

    with pytest.raises(ValueError, match="UTF-8"):
        handbook.search_handbook("policy")


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


def test_in_place_mutation_during_read_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text("a" * 70_000, encoding="utf-8")
    monkeypatch.setattr(handbook, "HANDBOOK_PATH", path)
    monkeypatch.setattr(handbook, "HANDBOOK_MAX_BYTES", 100_000)
    original_read = handbook.os.read
    mutated = False

    def mutating_read(descriptor, amount):
        nonlocal mutated
        chunk = original_read(descriptor, amount)
        if chunk and not mutated:
            mutated = True
            path.write_text("b" * 70_000, encoding="utf-8")
        return chunk

    monkeypatch.setattr(handbook.os, "read", mutating_read)
    _reset_cache(monkeypatch)

    with pytest.raises(ValueError, match="changed while it was being read"):
        handbook.search_handbook("policy")


def test_search_validates_top_k_and_ignores_nonfinite_weights():
    index = SimpleNamespace(
        idf={"policy": 1.0, "bad": float("nan")},
        index={
            "policy": {"handbook-1": 2.0, "handbook-2": float("inf")},
            "bad": {"handbook-2": 1.0},
        },
    )
    chunks = [("handbook-1", "policy evidence"), ("handbook-2", "bad evidence")]

    assert handbook._search("policy bad", index, chunks, top_k=1) == [
        ("handbook-1", "policy evidence")
    ]
    for invalid in (
        True,
        0,
        11,
        1.5,
        Decimal("1.5"),
        Fraction(3, 2),
        "bad",
    ):
        with pytest.raises(ValueError, match="top_k"):
            handbook._search("policy", index, chunks, top_k=invalid)


def test_search_accepts_exact_index_protocol_top_k():
    class ExactInteger:
        def __index__(self):
            return 1

    index = SimpleNamespace(
        idf={"policy": 1.0},
        index={"policy": {"handbook-1": 2.0}},
    )
    chunks = [("handbook-1", "policy evidence")]

    assert handbook._search(
        "policy",
        index,
        chunks,
        top_k=ExactInteger(),
    ) == [("handbook-1", "policy evidence")]
