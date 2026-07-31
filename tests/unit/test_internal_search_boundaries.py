import json
import os
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.internal_search as internal_search
from Searching import SearchHit


def test_direct_arguments_are_validated_before_engine_initialization(monkeypatch):
    initializer = MagicMock(side_effect=AssertionError("engine should not initialize"))
    monkeypatch.setattr(internal_search, "get_engine", initializer)

    for query, limit in (
        (object(), 5),
        ("q" * 2001, 5),
        ("bad\x00query", 5),
        ("bad\nquery", 5),
        ("bad\rquery", 5),
        ("bad\tquery", 5),
        ("bad\x7fquery", 5),
        ("query", "bad"),
        ("query", True),
        ("query", 1.5),
        ("query", Decimal("1.5")),
        ("query", Fraction(3, 2)),
        ("query", 0),
        ("query", 21),
    ):
        with pytest.raises(ValueError):
            internal_search.search_internal(query, limit=limit)

    assert internal_search.search_internal("   ") == []
    initializer.assert_not_called()


def test_exact_index_protocol_limit_is_accepted(monkeypatch):
    class ExactInteger:
        def __index__(self):
            return 3

    engine = MagicMock()
    engine.search.return_value = []
    monkeypatch.setattr(internal_search, "get_engine", lambda: engine)

    assert internal_search.search_internal("question", limit=ExactInteger()) == []
    engine.search.assert_called_once_with("question", limit=3)


def test_search_filters_malformed_hits_and_bounds_results(monkeypatch):
    hits = [
        SearchHit(1, "https://a.test", "A", "evidence", 0.9, 0.8, 0.1, 10),
        object(),
        SearchHit(2, "https://b.test", "B", "evidence", 0.8, 0.7, 0.1, 10),
    ]
    engine = MagicMock()
    engine.search.return_value = hits
    monkeypatch.setattr(internal_search, "get_engine", lambda: engine)

    citations = internal_search.search_internal("question", limit=2)

    assert len(citations) == 1
    assert citations[0].url == "https://a.test"
    assert citations[0].metadata["combined_score"] == 0.9


def test_invalid_citation_hit_does_not_abort_later_valid_hit(monkeypatch):
    hits = [
        SearchHit(
            1,
            "https://alice:password@example.test/private",
            "Unsafe",
            "evidence",
            0.9,
            0.8,
            0.1,
            10,
        ),
        SearchHit(2, "https://b.test", "B", "evidence", 0.8, 0.7, 0.1, 10),
    ]
    engine = MagicMock()
    engine.search.return_value = hits
    monkeypatch.setattr(internal_search, "get_engine", lambda: engine)

    citations = internal_search.search_internal("question", limit=2)

    assert [citation.url for citation in citations] == ["https://b.test"]


def test_strict_manifest_reader_rejects_nonstandard_json(tmp_path):
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text('{"generation":NaN}', encoding="utf-8")

    assert internal_search._read_manifest(manifest) is None
    assert internal_search._manifest_member_paths(tmp_path, manifest) == []


def test_manifest_reader_rejects_replacement_during_read(tmp_path, monkeypatch):
    generation = "a" * 32
    manifest = tmp_path / "snapshot_manifest.json"
    payload = json.dumps(
        {
            "generation": generation,
            "files": {
                "crawl": {"name": f"crawl_state.{generation}.json"},
                "index": {"name": f"index.{generation}.json"},
                "pagerank": {"name": f"pagerank.{generation}.json"},
            },
            "padding": "x" * 70_000,
        }
    )
    manifest.write_text(payload, encoding="utf-8")
    original_read = internal_search.os.read
    replaced = False

    def replacing_read(descriptor, amount):
        nonlocal replaced
        chunk = original_read(descriptor, amount)
        if chunk and not replaced:
            replaced = True
            replacement = tmp_path / "replacement.json"
            replacement.write_text(payload, encoding="utf-8")
            replacement.replace(manifest)
        return chunk

    monkeypatch.setattr(internal_search.os, "read", replacing_read)

    assert internal_search._read_manifest(manifest) is None


def test_storage_signature_rejects_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "classic"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link or reparse-point"):
        internal_search._storage_signature(link / "nested")


def test_storage_signature_rejects_reparse_flagged_root(tmp_path, monkeypatch):
    root = tmp_path / "classic"
    root.mkdir()
    original_lstat = internal_search.os.lstat

    class ReparseInfo:
        def __init__(self, info):
            self.st_mode = info.st_mode
            self.st_dev = info.st_dev
            self.st_ino = info.st_ino
            self.st_ctime_ns = info.st_ctime_ns
            self.st_mtime_ns = info.st_mtime_ns
            self.st_size = info.st_size
            self.st_file_attributes = internal_search._FILE_ATTRIBUTE_REPARSE_POINT

    def fake_lstat(path):
        info = original_lstat(path)
        return ReparseInfo(info) if Path(path) == root else info

    monkeypatch.setattr(internal_search.os, "lstat", fake_lstat)

    with pytest.raises(ValueError, match="reparse-point"):
        internal_search._storage_signature(root)


def test_engine_reload_closes_superseded_instance(monkeypatch, tmp_path):
    first = MagicMock()
    second = MagicMock()
    engines = iter([first, second])
    signatures = iter([
        (str(tmp_path), (("a", 1, 1, 1, 1, 1),)),
        (str(tmp_path), (("a", 1, 1, 1, 1, 1),)),
        (str(tmp_path), (("a", 2, 2, 2, 2, 2),)),
        (str(tmp_path), (("a", 2, 2, 2, 2, 2),)),
    ])
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(internal_search, "_ENGINE_INSTANCE", None)
    monkeypatch.setattr(internal_search, "_ENGINE_SIGNATURE", None)
    monkeypatch.setattr(
        internal_search,
        "_storage_signature",
        lambda _path: next(signatures),
    )
    monkeypatch.setattr(
        internal_search,
        "AcademicSearchEngine",
        lambda **_kwargs: next(engines),
    )

    assert internal_search.get_engine() is first
    assert internal_search.get_engine() is second
    first.close.assert_called_once()
    second.close.assert_not_called()


def test_unstable_reload_closes_candidates_and_preserves_previous_engine(
    monkeypatch,
    tmp_path,
):
    previous = MagicMock()
    candidate_one = MagicMock()
    candidate_two = MagicMock()
    candidate_three = MagicMock()
    engines = iter([candidate_one, candidate_two, candidate_three])
    signatures = iter(
        [
            (str(tmp_path), (("a", 2, 2, 2, 2, 2),)),
            (str(tmp_path), (("a", 3, 3, 3, 3, 3),)),
            (str(tmp_path), (("a", 4, 4, 4, 4, 4),)),
            (str(tmp_path), (("a", 5, 5, 5, 5, 5),)),
        ]
    )
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(internal_search, "_ENGINE_INSTANCE", previous)
    monkeypatch.setattr(
        internal_search,
        "_ENGINE_SIGNATURE",
        (str(tmp_path), (("a", 1, 1, 1, 1, 1),)),
    )
    monkeypatch.setattr(
        internal_search,
        "_storage_signature",
        lambda _path: next(signatures),
    )
    monkeypatch.setattr(
        internal_search,
        "AcademicSearchEngine",
        lambda **_kwargs: next(engines),
    )

    with pytest.raises(RuntimeError, match="changed repeatedly"):
        internal_search.get_engine()

    assert internal_search._ENGINE_INSTANCE is previous
    candidate_one.close.assert_called_once()
    candidate_two.close.assert_called_once()
    candidate_three.close.assert_called_once()
    previous.close.assert_not_called()


def test_manifest_member_names_must_match_generation_exactly(tmp_path):
    generation = "a" * 32
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "generation": generation,
                "files": {
                    "crawl": {"name": f"crawl_state.{generation}.json"},
                    "index": {"name": "../outside.json"},
                    "pagerank": {"name": f"pagerank.{generation}.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert internal_search._manifest_member_paths(tmp_path, manifest) == []


def test_nonregular_generation_member_has_invalid_signature(tmp_path):
    member = tmp_path / "index.json"
    member.mkdir()

    identity = internal_search._file_identity(member)

    assert identity[1:] == (-2, -2, -2, -2, -2)
