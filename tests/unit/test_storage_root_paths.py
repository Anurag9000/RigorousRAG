import json
import os

import pytest

from storage import CrawlState, StorageManager


def _symlink_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")


def test_classic_storage_rejects_symlinked_root(tmp_path):
    target = tmp_path / "real-data"
    target.mkdir()
    link = tmp_path / "data"
    _symlink_or_skip(link, target)

    with pytest.raises(ValueError, match="CLASSIC_STORAGE_DIR"):
        StorageManager(link)


def test_classic_storage_rejects_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _symlink_or_skip(linked_parent, outside)

    with pytest.raises(ValueError, match="symbolic-link components"):
        StorageManager(linked_parent / "data")


def test_classic_storage_detects_root_swap_before_read(tmp_path):
    root = tmp_path / "data"
    store = StorageManager(root)
    store.save_crawl_state(CrawlState.empty())
    moved = tmp_path / "data-original"
    root.rename(moved)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(root, outside)

    with pytest.raises(OSError, match="CLASSIC_STORAGE_DIR"):
        store.load_crawl_state()


def test_classic_storage_detects_parent_swap_after_initialization(tmp_path):
    parent = tmp_path / "parent"
    root = parent / "data"
    store = StorageManager(root)
    store.save_crawl_state(CrawlState.empty())

    moved_parent = tmp_path / "parent-original"
    parent.rename(moved_parent)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(parent, outside)

    with pytest.raises(OSError, match="CLASSIC_STORAGE_DIR"):
        store.load_crawl_state()


def test_member_paths_must_be_direct_children_of_bound_root(tmp_path):
    store = StorageManager(tmp_path / "data")
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="direct children"):
        store._write_json(outside, {"value": 1})
    with pytest.raises(ValueError, match="direct children"):
        store._read_json(outside)


def test_nonstandard_json_is_quarantined(tmp_path):
    store = StorageManager(tmp_path / "data")
    path = store.base_dir / "invalid.json"
    path.write_text('{"value":NaN}', encoding="utf-8")

    assert store._read_json(path) is None
    assert not path.exists()
    quarantined = list(store.base_dir.glob("invalid.json.corrupt-*"))
    assert len(quarantined) == 1


def test_atomic_write_replaces_final_symlink_without_touching_target(tmp_path):
    store = StorageManager(tmp_path / "data")
    target = tmp_path / "outside.json"
    target.write_text("unchanged", encoding="utf-8")
    link = store.base_dir / "member.json"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable in this environment.")

    store._write_json(link, {"value": 1})

    assert target.read_text(encoding="utf-8") == "unchanged"
    assert not link.is_symlink()
    assert json.loads(link.read_text(encoding="utf-8")) == {"value": 1}


def test_normal_classic_storage_still_round_trips(tmp_path):
    store = StorageManager(tmp_path / "data")
    state = CrawlState.empty()

    store.save_crawl_state(state)

    loaded = store.load_crawl_state()
    assert loaded.pages == {}
    assert loaded.graph == {}
    assert loaded.visited == set()
    assert loaded.frontier == []
