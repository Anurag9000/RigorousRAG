import json
from types import SimpleNamespace
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
        ("query", "bad"),
        ("query", 0),
        ("query", 21),
    ):
        with pytest.raises(ValueError):
            internal_search.search_internal(query, limit=limit)

    assert internal_search.search_internal("   ") == []
    initializer.assert_not_called()


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


def test_strict_manifest_reader_rejects_nonstandard_json(tmp_path):
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text('{"generation":NaN}', encoding="utf-8")

    assert internal_search._read_manifest(manifest) is None
    assert internal_search._manifest_member_paths(tmp_path, manifest) == []


def test_storage_signature_rejects_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "classic"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="symbolic-link components"):
        internal_search._storage_signature(link / "nested")


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
