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


def test_normal_classic_storage_still_round_trips(tmp_path):
    store = StorageManager(tmp_path / "data")
    state = CrawlState.empty()

    store.save_crawl_state(state)

    loaded = store.load_crawl_state()
    assert loaded.pages == {}
    assert loaded.graph == {}
    assert loaded.visited == set()
    assert loaded.frontier == []
