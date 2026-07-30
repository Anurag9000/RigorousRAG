import json
import os
from types import SimpleNamespace

import pytest

import tools.internal_search as internal_search
from Searching import SearchHit


def _reset_engine(monkeypatch):
    monkeypatch.setattr(internal_search, "_ENGINE_INSTANCE", None)
    monkeypatch.setattr(internal_search, "_ENGINE_SIGNATURE", None)


def test_engine_is_reused_until_storage_signature_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path))
    _reset_engine(monkeypatch)
    constructed = []

    class FakeEngine:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir
            constructed.append(self)

        def search(self, _query, *, limit):
            return []

    monkeypatch.setattr(internal_search, "AcademicSearchEngine", FakeEngine)

    first = internal_search.get_engine()
    second = internal_search.get_engine()
    assert first is second
    assert len(constructed) == 1

    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text('{"generation":"one"}', encoding="utf-8")

    third = internal_search.get_engine()
    assert third is not first
    assert len(constructed) == 2
    assert third.storage_dir == str(tmp_path)


def test_manifest_replacement_triggers_reload_even_when_size_and_mtime_match(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path))
    _reset_engine(monkeypatch)
    constructed = []

    class FakeEngine:
        def __init__(self, *, storage_dir):
            constructed.append(storage_dir)

        def search(self, _query, *, limit):
            return []

    monkeypatch.setattr(internal_search, "AcademicSearchEngine", FakeEngine)
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text("generation-a", encoding="utf-8")
    original = manifest.stat()
    first = internal_search.get_engine()

    replacement = tmp_path / "replacement.tmp"
    replacement.write_text("generation-b", encoding="utf-8")
    os.utime(
        replacement,
        ns=(original.st_atime_ns, original.st_mtime_ns),
    )
    replacement.replace(manifest)
    assert manifest.stat().st_size == original.st_size
    assert manifest.stat().st_mtime_ns == original.st_mtime_ns

    second = internal_search.get_engine()
    assert second is not first
    assert len(constructed) == 2


def test_referenced_generation_member_replacement_triggers_reload(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path))
    _reset_engine(monkeypatch)
    constructed = []

    class FakeEngine:
        def __init__(self, *, storage_dir):
            constructed.append(storage_dir)

        def search(self, _query, *, limit):
            return []

    monkeypatch.setattr(internal_search, "AcademicSearchEngine", FakeEngine)
    generation = "a" * 32
    names = {
        "crawl": f"crawl_state.{generation}.json",
        "index": f"index.{generation}.json",
        "pagerank": f"pagerank.{generation}.json",
    }
    for name in names.values():
        (tmp_path / name).write_text("member-a", encoding="utf-8")
    manifest = {
        "generation": generation,
        "files": {
            key: {"name": name}
            for key, name in names.items()
        },
    }
    (tmp_path / "snapshot_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    first = internal_search.get_engine()

    member = tmp_path / names["index"]
    original = member.stat()
    replacement = tmp_path / "index-replacement.tmp"
    replacement.write_text("member-b", encoding="utf-8")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(member)
    assert member.stat().st_size == original.st_size
    assert member.stat().st_mtime_ns == original.st_mtime_ns

    second = internal_search.get_engine()
    assert second is not first
    assert len(constructed) == 2


def test_symlinked_classic_storage_root_is_refused(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "classic"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(link))
    _reset_engine(monkeypatch)

    with pytest.raises(ValueError, match="CLASSIC_STORAGE_DIR"):
        internal_search.get_engine()


def test_direct_internal_search_bounds_query_before_engine_lookup(monkeypatch):
    _reset_engine(monkeypatch)
    monkeypatch.setattr(
        internal_search,
        "get_engine",
        lambda: (_ for _ in ()).throw(AssertionError("engine should not load")),
    )

    with pytest.raises(ValueError, match="2,000"):
        internal_search.search_internal("q" * 2001)


def test_internal_search_maps_hits_to_bounded_citations(monkeypatch):
    hit = SearchHit(
        rank=1,
        url="https://example.test/paper",
        title="Paper",
        snippet="Relevant evidence",
        score=0.9,
        cosine=0.8,
        pagerank=0.7,
        length=1234,
    )
    engine = SimpleNamespace(search=lambda query, limit: [hit])
    monkeypatch.setattr(internal_search, "get_engine", lambda: engine)

    citations = internal_search.search_internal("evidence", limit=20)

    assert len(citations) == 1
    citation = citations[0]
    assert citation.label == "[1]"
    assert citation.url == hit.url
    assert citation.metadata == {
        "combined_score": 0.9,
        "cosine": 0.8,
        "pagerank": 0.7,
    }
