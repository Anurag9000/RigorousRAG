import threading
import time

import pytest

from Crawler import Page
from Indexer import InvertedIndex
from storage import CrawlState, StorageManager


def _snapshot(label: str):
    url = f"https://{label}.test/"
    state = CrawlState(
        pages={url: Page(url, label, f"{label} evidence", [], "text/html", 20)},
        graph={url: set()},
        visited={url},
        frontier=[],
    )
    index = InvertedIndex()
    index.build(state.pages)
    return state, index, {url: 1.0}


def test_snapshot_reader_blocks_cleanup_until_generation_is_loaded(tmp_path, monkeypatch):
    initial_state, initial_index, initial_rank = _snapshot("initial")
    writer_state, writer_index, writer_rank = _snapshot("replacement")
    StorageManager(tmp_path).save_snapshot(initial_state, initial_index, initial_rank)

    reader = StorageManager(tmp_path)
    writer = StorageManager(tmp_path)
    entered_member_read = threading.Event()
    release_reader = threading.Event()
    reader_done = threading.Event()
    writer_done = threading.Event()
    errors = []
    loaded = []
    original_read_member = reader._read_manifest_member
    first_call = True

    def blocking_read_member(**kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            entered_member_read.set()
            if not release_reader.wait(timeout=5):
                raise AssertionError("Reader release was not signalled.")
        return original_read_member(**kwargs)

    monkeypatch.setattr(reader, "_read_manifest_member", blocking_read_member)

    def read_snapshot():
        try:
            loaded.append(reader.load_snapshot())
        except Exception as exc:  # pragma: no cover - assertion reports detail
            errors.append(exc)
        finally:
            reader_done.set()

    def write_snapshot():
        try:
            writer.save_snapshot(writer_state, writer_index, writer_rank)
        except Exception as exc:  # pragma: no cover - assertion reports detail
            errors.append(exc)
        finally:
            writer_done.set()

    reader_thread = threading.Thread(target=read_snapshot)
    reader_thread.start()
    assert entered_member_read.wait(timeout=5)

    writer_thread = threading.Thread(target=write_snapshot)
    writer_thread.start()
    time.sleep(0.1)
    assert not writer_done.is_set(), "Writer bypassed the snapshot reader lock."

    release_reader.set()
    reader_thread.join(timeout=5)
    writer_thread.join(timeout=5)

    assert reader_done.is_set() and writer_done.is_set()
    assert errors == []
    assert len(loaded) == 1
    loaded_state, loaded_index, loaded_rank = loaded[0]
    assert set(loaded_state.pages) == set(initial_state.pages)
    assert set(loaded_index.documents) == set(initial_index.documents)
    assert loaded_rank == initial_rank

    final_state, final_index, final_rank = StorageManager(tmp_path).load_snapshot()
    assert set(final_state.pages) == set(writer_state.pages)
    assert set(final_index.documents) == set(writer_index.documents)
    assert final_rank == writer_rank


def test_snapshot_lock_path_cannot_be_a_symlink(tmp_path):
    manager = StorageManager(tmp_path)
    target = tmp_path / "lock-target"
    target.write_bytes(b"x")
    try:
        manager.snapshot_lock_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    state, index, pagerank = _snapshot("blocked")
    with pytest.raises(OSError, match="symbolic link"):
        manager.save_snapshot(state, index, pagerank)
